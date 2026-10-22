"""SLAM-ASR Core Model Architecture (t4-ng-ta-gormez-lora: Görmez Hybrid Projector + PEFT LoRA).

Combines:
- Speech Foundation Encoder (Whisper Large v3) [FROZEN]
- Görmez Hybrid Attention Projector (multi-order n-gram + BiGRU temporal attention pooling) [TRAINABLE]
- Large Language Model Backbone (Qwen2.5-7B) [PEFT LoRA TRAINABLE]
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from slam_asr.models.projector import SpeechProjector
from slam_asr.models.speech_encoder import SpeechEncoderWrapper


@dataclass
class SLAMASRConfig:
    """Configuration class for SLAM-ASR (t4-ng-ta-gormez-lora)."""

    speech_encoder_name_or_path: str = "facebook/hubert-large-ls960-ft"
    llm_name_or_path: str = "lmsys/vicuna-7b-v1.5"
    encoder_dim: Optional[int] = None
    llm_dim: Optional[int] = None
    downsample_rate: int = 5
    projector_type: str = "gormez_hybrid"  # 'gormez_hybrid', 'ngram_temporal_hybrid', 'linear', 'mlp'
    ngram_orders: tuple[int, ...] = (1, 3, 5, 9)
    ngram_window_size: int = 5
    temporal_pooling_mode: str = "hybrid"  # 'hybrid', 'attention', or 'weighted_concat'
    hybrid_attention_mode: str = "split_heads"  # 'split_heads', 'sequential', or 'global'
    hybrid_attention_heads: int = 4
    hybrid_dropout: float = 0.1
    projector_hidden_dim: int = 2048
    projector_padding_mode: str = "discard"  # 'discard' or 'pad'

    # Adaptation & Fine-Tuning (PEFT / LoRA)
    freeze_encoder: bool = True
    freeze_llm: bool = True
    use_lora: bool = True  # Enable LoRA / QLoRA on LLM
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")

    default_prompt: str = "Konuşmayı metne dönüştürün."
    user_prefix: str = "USER: "
    assistant_prefix: str = " ASSISTANT: "
    torch_dtype: str = "float32"  # 'float32', 'float16', 'bfloat16'
    dtype: Optional[str] = None  # Modern transformers standard parameter alias
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: Optional[str] = None


class SLAM_ASR(nn.Module):
    """SLAM-ASR Model.

    Combines an off-the-shelf speech encoder and a frozen large language model
    via a lightweight, trainable projector with temporal downsampling.
    """

    def __init__(
        self,
        config: SLAMASRConfig,
        speech_encoder: Optional[SpeechEncoderWrapper] = None,
        projector: Optional[SpeechProjector] = None,
        language_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__()
        self.config = config

        # 1. Initialize or assign speech encoder
        if speech_encoder is not None:
            self.speech_encoder = speech_encoder
        else:
            self.speech_encoder = SpeechEncoderWrapper(
                encoder_name_or_path=config.speech_encoder_name_or_path,
                freeze_encoder=config.freeze_encoder,
            )

        encoder_dim = config.encoder_dim or self.speech_encoder.hidden_size

        # 2. Initialize or assign LLM and tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.llm_name_or_path,
                use_fast=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        if language_model is not None:
            self.language_model = language_model
        else:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            model_dtype_str = getattr(config, "dtype", None) or getattr(config, "torch_dtype", "float32")
            torch_dtype = dtype_map.get(model_dtype_str, torch.float32)
            llm_kwargs = {}

            if config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                llm_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype if torch_dtype != torch.float32 else torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                llm_kwargs["device_map"] = config.device_map or "auto"
            elif config.load_in_8bit:
                from transformers import BitsAndBytesConfig
                llm_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                llm_kwargs["device_map"] = config.device_map or "auto"
            elif config.device_map:
                llm_kwargs["device_map"] = config.device_map

            try:
                # Modern transformers (>=4.45) prefers 'dtype' and deprecates 'torch_dtype'
                self.language_model = AutoModelForCausalLM.from_pretrained(
                    config.llm_name_or_path,
                    dtype=torch_dtype,
                    **llm_kwargs,
                )
            except (TypeError, ValueError):
                # Backwards compatibility for older transformers versions
                self.language_model = AutoModelForCausalLM.from_pretrained(
                    config.llm_name_or_path,
                    torch_dtype=torch_dtype,
                    **llm_kwargs,
                )

        llm_dim = config.llm_dim or self.language_model.config.hidden_size

        # Freeze LLM if requested (default True in SLAM-ASR)
        if config.freeze_llm:
            self.freeze_llm()

        # Attach LoRA adapters to LLM if requested
        if config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
                if getattr(config, "load_in_4bit", False) or getattr(config, "load_in_8bit", False):
                    try:
                        self.language_model = prepare_model_for_kbit_training(self.language_model)
                    except Exception as prep_err:
                        print(f"[*] Note: prepare_model_for_kbit_training skipped: {prep_err}")
                lora_cfg = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    target_modules=list(config.lora_target_modules),
                    lora_dropout=config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self.language_model = get_peft_model(self.language_model, lora_cfg)
                print("[*] Successfully attached LoRA adapters to LLM backbone.")
            except Exception as e:
                print(f"[!] Warning: Could not initialize PEFT/LoRA ({type(e).__name__}: {e}). Training with frozen LLM.")

        # 3. Initialize or assign Projector
        if projector is not None:
            self.projector = projector
        else:
            self.projector = SpeechProjector(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                downsample_rate=config.downsample_rate,
                projector_type=config.projector_type,
                hidden_dim=config.projector_hidden_dim,
                padding_mode=config.projector_padding_mode,
                ngram_orders=config.ngram_orders,
                ngram_window_size=config.ngram_window_size,
                temporal_pooling_mode=config.temporal_pooling_mode,
                hybrid_attention_mode=config.hybrid_attention_mode,
                num_attention_heads=config.hybrid_attention_heads,
                attention_dropout=config.hybrid_dropout,
            )

    def freeze_llm(self):
        """Freeze LLM backbone."""
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.language_model.eval()

    def unfreeze_llm(self):
        """Unfreeze LLM backbone for fine-tuning exploration."""
        for param in self.language_model.parameters():
            param.requires_grad = True
        self.language_model.train()

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return list of trainable parameters (projector + LoRA adapters if active)."""
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def llm_dtype(self) -> torch.dtype:
        """Get the expected computation dtype for the LLM inputs."""
        if hasattr(self.language_model, "dtype") and self.language_model.dtype not in (torch.int8, torch.uint8):
            return self.language_model.dtype
        embed = self.language_model.get_input_embeddings()
        if hasattr(embed, "weight") and embed.weight.dtype not in (torch.int8, torch.uint8):
            return embed.weight.dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_dtype_str = getattr(self.config, "dtype", None) or getattr(self.config, "torch_dtype", "float32")
        return dtype_map.get(model_dtype_str, torch.float32)

    def encode_speech(
        self,
        speech_inputs: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pass speech through encoder and projector.

        Returns:
            speech_embeds: (batch_size, downsampled_frames, llm_dim)
            speech_mask: (batch_size, downsampled_frames) or None
        """
        if self.config.freeze_encoder:
            with torch.no_grad():
                features, mask = self.speech_encoder(speech_inputs, speech_attention_mask)
        else:
            features, mask = self.speech_encoder(speech_inputs, speech_attention_mask)
        projected, downsampled_mask = self.projector(features, mask)
        return projected, downsampled_mask

    def prepare_multimodal_embeddings(
        self,
        speech_embeds: torch.Tensor,
        speech_mask: Optional[torch.Tensor],
        transcripts: Optional[List[str]] = None,
        prompts: Optional[Union[str, List[str]]] = None,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compose the full input sequence embeddings and compute label masks.

        Format:
            Training: USER: <S> <P> ASSISTANT: <T><eos>
            Inference: USER: <S> <P> ASSISTANT:

        Labels are -100 for prefix, speech frames, prompt, and padding.
        Loss is ONLY computed on transcript tokens <T> and <eos>.

        Returns:
            inputs_embeds: (batch_size, max_seq_len, llm_dim)
            attention_mask: (batch_size, max_seq_len)
            labels: (batch_size, max_seq_len) or None (during inference)
        """
        batch_size = speech_embeds.size(0)
        target_dtype = self.llm_dtype

        # Normalize prompt input
        if prompts is None:
            prompt_list = [self.config.default_prompt] * batch_size
        elif isinstance(prompts, str):
            prompt_list = [prompts] * batch_size
        else:
            prompt_list = prompts

        embed_tokens = self.language_model.get_input_embeddings()
        embed_weight = getattr(embed_tokens, "weight", None)
        device = embed_weight.device if embed_weight is not None else speech_embeds.device
        speech_embeds = speech_embeds.to(device=device, dtype=target_dtype)

        sample_embeds_list = []
        sample_labels_list = []

        user_prefix = self.config.user_prefix
        assistant_prefix = self.config.assistant_prefix

        # Tokenize user prefix once
        user_prefix_ids = self.tokenizer(
            user_prefix,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.to(device)
        user_prefix_embeds = embed_tokens(user_prefix_ids)[0].to(dtype=target_dtype, device=device)  # (p1_len, llm_dim)
        p1_len = user_prefix_embeds.size(0)

        for i in range(batch_size):
            # 1. Speech embeddings for sample i
            if speech_mask is not None:
                valid_speech_len = speech_mask[i].sum().item()
                cur_speech_embeds = speech_embeds[i, :valid_speech_len]
            else:
                cur_speech_embeds = speech_embeds[i]
            cur_speech_embeds = cur_speech_embeds.to(dtype=target_dtype, device=device)
            s_len = cur_speech_embeds.size(0)

            # 2. Prompt + assistant prefix: " <P> ASSISTANT: "
            prompt_str = f" {prompt_list[i]}{assistant_prefix}"
            prompt_ids = self.tokenizer(
                prompt_str,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(device)
            prompt_embeds = embed_tokens(prompt_ids)[0].to(dtype=target_dtype, device=device)
            p2_len = prompt_embeds.size(0)

            if is_training and transcripts is not None:
                # 3. Target transcript: "<T><eos>"
                target_str = f"{transcripts[i]}{self.tokenizer.eos_token}"
                target_ids = self.tokenizer(
                    target_str,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(device)[0]
                target_embeds = embed_tokens(target_ids).to(dtype=target_dtype, device=device)
                t_len = target_ids.size(0)

                # Concatenate embeddings: [prefix, speech, prompt_suffix, target]
                full_embeds = torch.cat(
                    [user_prefix_embeds, cur_speech_embeds, prompt_embeds, target_embeds],
                    dim=0,
                )

                # Labels: -100 for non-target tokens, target_ids for target
                non_target_len = p1_len + s_len + p2_len
                labels = torch.full(
                    (non_target_len + t_len,),
                    -100,
                    dtype=torch.long,
                    device=device,
                )
                labels[non_target_len:] = target_ids

                sample_labels_list.append(labels)
            else:
                # Inference mode: only prefix, speech, prompt_suffix
                full_embeds = torch.cat(
                    [user_prefix_embeds, cur_speech_embeds, prompt_embeds],
                    dim=0,
                )

            sample_embeds_list.append(full_embeds)

        # Pad to max length in batch (right padding)
        max_len = max(emb.size(0) for emb in sample_embeds_list)
        llm_dim = sample_embeds_list[0].size(1)

        padded_embeds = torch.zeros(
            batch_size, max_len, llm_dim, dtype=target_dtype, device=device
        )
        attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=device
        )

        padded_labels = None
        if is_training and transcripts is not None:
            padded_labels = torch.full(
                (batch_size, max_len), -100, dtype=torch.long, device=device
            )

        for i in range(batch_size):
            cur_len = sample_embeds_list[i].size(0)
            if not is_training:
                # Left-padding for decoder autoregressive generation (prevents prompt offset/corruption)
                padded_embeds[i, -cur_len:] = sample_embeds_list[i]
                attention_mask[i, -cur_len:] = 1
            else:
                # Right-padding for training loss calculation
                padded_embeds[i, :cur_len] = sample_embeds_list[i]
                attention_mask[i, :cur_len] = 1
                if padded_labels is not None:
                    padded_labels[i, :cur_len] = sample_labels_list[i]

        return padded_embeds, attention_mask, padded_labels

    def forward(
        self,
        speech_inputs: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
        transcripts: Optional[List[str]] = None,
        prompts: Optional[Union[str, List[str]]] = None,
    ) -> dict:
        """Forward pass during training.

        Args:
            speech_inputs: Raw audio or mel spectrogram batch (batch_size, time).
            speech_attention_mask: Optional speech mask.
            transcripts: List of target transcript strings for the batch.
            prompts: Optional prompt or list of prompts (default: "Transcribe speech to text.").

        Returns:
            dict containing:
                loss: Cross-entropy loss computed strictly on transcript tokens.
                logits: Output logits from the LLM.
        """
        speech_embeds, speech_mask = self.encode_speech(
            speech_inputs, speech_attention_mask
        )

        inputs_embeds, attention_mask, labels = self.prepare_multimodal_embeddings(
            speech_embeds=speech_embeds,
            speech_mask=speech_mask,
            transcripts=transcripts,
            prompts=prompts,
            is_training=True,
        )

        target_dtype = self.llm_dtype
        inputs_embeds = inputs_embeds.to(dtype=target_dtype)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        speech_inputs: torch.Tensor,
        speech_attention_mask: Optional[torch.Tensor] = None,
        prompts: Optional[Union[str, List[str]]] = None,
        num_beams: int = 4,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        """Autoregressive ASR decoding using beam search (beam=4 by default in the paper).

        Args:
            speech_inputs: Raw audio or mel spectrogram batch (batch_size, time).
            speech_attention_mask: Optional speech mask.
            prompts: Optional prompt string or list.
            num_beams: Beam search width (default: 4, per paper Section 3.4).
            max_new_tokens: Maximum tokens to generate.
            repetition_penalty: Penalty for repeating tokens (default: 1.15).
            no_repeat_ngram_size: Avoid repeating n-grams of this size (default: 4).

        Returns:
            List of generated hypothesis transcript strings.
        """
        speech_embeds, speech_mask = self.encode_speech(
            speech_inputs, speech_attention_mask
        )

        inputs_embeds, attention_mask, _ = self.prepare_multimodal_embeddings(
            speech_embeds=speech_embeds,
            speech_mask=speech_mask,
            transcripts=None,
            prompts=prompts,
            is_training=False,
        )

        target_dtype = self.llm_dtype
        inputs_embeds = inputs_embeds.to(dtype=target_dtype)

        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        hypotheses = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return [hyp.strip() for hyp in hypotheses]
