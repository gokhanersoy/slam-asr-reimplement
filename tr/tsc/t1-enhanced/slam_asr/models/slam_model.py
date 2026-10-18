"""SLAM-ASR Core Model Architecture (t1-enhanced: Enhanced Baseline).

Upgrades the Baseline (t0) with:
1. 2-Layer Non-Linear MLP Projector (GELU, SwiGLU, or ReLU) + Conv1d Downsampling
2. Parameter-Efficient Fine-Tuning (PEFT / LoRA / QLoRA) on the LLM backbone
3. Auxiliary CTC Loss for monotonic phonetic alignment and hallucination suppression
4. Optional unfreezing of speech encoder top layers
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Configuration class for SLAM-ASR (t1-enhanced)."""

    speech_encoder_name_or_path: str = "facebook/hubert-large-ls960-ft"
    llm_name_or_path: str = "lmsys/vicuna-7b-v1.5"
    encoder_dim: Optional[int] = None
    llm_dim: Optional[int] = None
    downsample_rate: int = 5
    projector_type: str = "mlp"  # 'mlp', 'conv_mlp', 'swiglu', or 'linear'
    projector_hidden_dim: int = 2048
    projector_padding_mode: str = "discard"  # 'discard' or 'pad'
    mlp_activation: str = "gelu"  # 'gelu', 'swiglu', or 'relu'
    mlp_dropout: float = 0.1
    downsampler_type: str = "concat"  # 'concat' or 'conv1d'

    # Adaptation & Fine-Tuning
    freeze_encoder: bool = True
    unfreeze_encoder_layers: int = 0  # Number of top encoder layers to unfreeze (0 = fully frozen)
    freeze_llm: bool = True
    use_lora: bool = True  # Enable LoRA / QLoRA on LLM
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")

    # Multi-task Auxiliary CTC Loss
    use_ctc: bool = True  # Auxiliary CTC loss on projector outputs
    ctc_weight: float = 0.2  # Lambda weighting for CTC loss

    default_prompt: str = "Konuşmayı metne dönüştürün."
    user_prefix: str = "USER: "
    assistant_prefix: str = " ASSISTANT: "
    torch_dtype: str = "float32"  # 'float32', 'float16', 'bfloat16'
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: Optional[str] = None


class SLAM_ASR(nn.Module):
    """SLAM-ASR Model with Enhanced Baseline capabilities (t1-enhanced)."""

    def __init__(
        self,
        config: SLAMASRConfig,
        speech_encoder: Optional[SpeechEncoderWrapper] = None,
        projector: Optional[SpeechProjector] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        language_model: Optional[PreTrainedModel] = None,
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

        # Unfreeze top encoder layers if requested
        if config.unfreeze_encoder_layers > 0:
            self._unfreeze_encoder_top_layers(config.unfreeze_encoder_layers)

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
            torch_dtype = dtype_map.get(config.torch_dtype, torch.float32)
            llm_kwargs = {"torch_dtype": torch_dtype}

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

            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.llm_name_or_path,
                **llm_kwargs,
            )

        llm_dim = config.llm_dim or self.language_model.config.hidden_size

        # Freeze base LLM
        if config.freeze_llm:
            self.freeze_llm()

        # Attach LoRA adapters to LLM if requested
        if config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
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
            except ImportError:
                print("[!] Warning: peft library not available. Training with frozen LLM.")

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
                mlp_activation=config.mlp_activation,
                mlp_dropout=config.mlp_dropout,
                downsampler_type=config.downsampler_type,
            )

        # 4. Initialize Auxiliary CTC Head if enabled
        if config.use_ctc and config.ctc_weight > 0:
            self.ctc_chars = [
                "<blank>", " ", "a", "b", "c", "ç", "d", "e", "f", "g", "ğ",
                "h", "ı", "i", "j", "k", "l", "m", "n", "o", "ö", "p", "r",
                "s", "ş", "t", "u", "ü", "v", "y", "z", "0", "1", "2", "3",
                "4", "5", "6", "7", "8", "9", "<unk>",
            ]
            self.ctc_char2id = {ch: i for i, ch in enumerate(self.ctc_chars)}
            self.ctc_unk_id = self.ctc_char2id["<unk>"]
            self.ctc_head = nn.Linear(llm_dim, len(self.ctc_chars))
        else:
            self.ctc_head = None

    def _unfreeze_encoder_top_layers(self, num_layers: int):
        """Unfreeze the top N layers of speech encoder for acoustic fine-tuning."""
        enc = getattr(self.speech_encoder, "encoder", None)
        if enc is None:
            return
        layers = None
        if hasattr(enc, "layers"):
            layers = enc.layers
        elif hasattr(enc, "encoder") and hasattr(enc.encoder, "layers"):
            layers = enc.encoder.layers
        if layers is not None:
            total_layers = len(layers)
            for idx in range(max(0, total_layers - num_layers), total_layers):
                for param in layers[idx].parameters():
                    param.requires_grad = True
            print(f"[*] Unfroze top {num_layers} layers of speech encoder ({total_layers - num_layers}..{total_layers - 1}).")

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
        """Return list of trainable parameters (projector + LoRA adapters + CTC head)."""
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
        return dtype_map.get(getattr(self.config, "torch_dtype", "float32"), torch.float32)

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
        features, mask = self.speech_encoder(speech_inputs, speech_attention_mask)
        projected, downsampled_mask = self.projector(features, mask)
        return projected, downsampled_mask

    def prepare_multimodal_embeddings(
        self,
        speech_embeds: torch.Tensor,
        speech_mask: Optional[torch.Tensor] = None,
        transcripts: Optional[List[str]] = None,
        prompts: Optional[Union[str, List[str]]] = None,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = speech_embeds.size(0)
        target_dtype = self.llm_dtype

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

        user_prefix_ids = self.tokenizer(
            user_prefix,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.to(device)
        user_prefix_embeds = embed_tokens(user_prefix_ids)[0].to(dtype=target_dtype, device=device)
        p1_len = user_prefix_embeds.size(0)

        for i in range(batch_size):
            if speech_mask is not None:
                valid_speech_len = speech_mask[i].sum().item()
                cur_speech_embeds = speech_embeds[i, :valid_speech_len]
            else:
                cur_speech_embeds = speech_embeds[i]
            cur_speech_embeds = cur_speech_embeds.to(dtype=target_dtype, device=device)
            s_len = cur_speech_embeds.size(0)

            prompt_str = f" {prompt_list[i]}{assistant_prefix}"
            prompt_ids = self.tokenizer(
                prompt_str,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(device)
            prompt_embeds = embed_tokens(prompt_ids)[0].to(dtype=target_dtype, device=device)
            p2_len = prompt_embeds.size(0)

            if is_training and transcripts is not None:
                target_str = f"{transcripts[i]}{self.tokenizer.eos_token}"
                target_ids = self.tokenizer(
                    target_str,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(device)[0]
                target_embeds = embed_tokens(target_ids).to(dtype=target_dtype, device=device)
                t_len = target_ids.size(0)

                full_embeds = torch.cat(
                    [user_prefix_embeds, cur_speech_embeds, prompt_embeds, target_embeds],
                    dim=0,
                )

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
                full_embeds = torch.cat(
                    [user_prefix_embeds, cur_speech_embeds, prompt_embeds],
                    dim=0,
                )

            sample_embeds_list.append(full_embeds)

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
                padded_embeds[i, -cur_len:] = sample_embeds_list[i]
                attention_mask[i, -cur_len:] = 1
            else:
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
        """Forward pass during training with optional auxiliary CTC loss."""
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

        lm_loss = outputs.loss
        ctc_loss = torch.tensor(0.0, device=speech_embeds.device)

        # Auxiliary CTC loss calculation
        if (
            self.config.use_ctc
            and self.config.ctc_weight > 0
            and self.ctc_head is not None
            and transcripts is not None
        ):
            ctc_logits = self.ctc_head(speech_embeds)  # (B, T_down, V)
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T_down, B, V)

            batch_target_ids = []
            target_lengths = []
            for t in transcripts:
                t_clean = t.lower().strip()
                ids = [self.ctc_char2id.get(ch, self.ctc_unk_id) for ch in t_clean]
                batch_target_ids.extend(ids)
                target_lengths.append(len(ids))

            targets = torch.tensor(batch_target_ids, dtype=torch.long, device=speech_embeds.device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=speech_embeds.device)

            if speech_mask is not None:
                input_lengths = speech_mask.sum(dim=-1).long()
            else:
                input_lengths = torch.full(
                    (speech_embeds.size(0),), speech_embeds.size(1), dtype=torch.long, device=speech_embeds.device
                )

            # Clamp target lengths to at most input length for CTC valid path
            target_lengths = torch.clamp(target_lengths, min=1, max=input_lengths)

            ctc_loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                zero_infinity=True,
            )

        total_loss = lm_loss + (self.config.ctc_weight * ctc_loss if self.config.use_ctc else 0.0)

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "ctc_loss": ctc_loss,
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

        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "temperature": temperature,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        gen_kwargs.update(kwargs)

        generated_ids = self.language_model.generate(**gen_kwargs)
        transcriptions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [t.strip() for t in transcriptions]
