"""Unit tests for SLAM-ASR model forward pass, label masking, and parameter freezing."""

import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
    from slam_asr.models.slam_model import SLAM_ASR, SLAMASRConfig
    from slam_asr.models.projector import SpeechProjector
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


if TORCH_AVAILABLE:
    class DummySpeechEncoder(nn.Module):
        def __init__(self, hidden_size=64):
            super().__init__()
            self.hidden_size = hidden_size
            self.dummy_layer = nn.Linear(hidden_size, hidden_size)

        def forward(self, x, mask=None):
            # x is (batch, samples), return features (batch, 20, hidden_size)
            batch = x.size(0)
            return torch.randn(batch, 20, self.hidden_size), None


    class DummyTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"

        def __call__(self, text, add_special_tokens=True, return_tensors=None):
            tokens = [abs(hash(w)) % 100 + 3 for w in text.split()]
            if not tokens:
                tokens = [3]
            if return_tensors == "pt":
                return SimpleNamespace(input_ids=torch.tensor([tokens], dtype=torch.long))
            return SimpleNamespace(input_ids=tokens)

        def batch_decode(self, token_ids, skip_special_tokens=True):
            return ["dummy transcription" for _ in token_ids]


    class DummyLanguageModel(nn.Module):
        def __init__(self, vocab_size=200, hidden_size=128):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def get_input_embeddings(self):
            return self.embeddings

        def forward(self, inputs_embeds=None, attention_mask=None, labels=None):
            logits = self.lm_head(inputs_embeds)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            return SimpleNamespace(loss=loss, logits=logits)

        def generate(self, inputs_embeds=None, **kwargs):
            batch = inputs_embeds.size(0)
            return torch.tensor([[10, 11, 2]] * batch, dtype=torch.long)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for SLAM_ASR tests")
class TestSLAMASR(unittest.TestCase):
    def setUp(self):
        self.config = SLAMASRConfig(
            encoder_dim=64,
            llm_dim=128,
            downsample_rate=5,
            projector_type="linear",
            freeze_encoder=True,
            freeze_llm=True,
            use_lora=False,
        )
        self.speech_encoder = DummySpeechEncoder(hidden_size=64)
        self.tokenizer = DummyTokenizer()
        self.language_model = DummyLanguageModel(hidden_size=128)
        self.projector = SpeechProjector(
            encoder_dim=64,
            llm_dim=128,
            downsample_rate=5,
            projector_type="linear",
        )

        self.model = SLAM_ASR(
            config=self.config,
            speech_encoder=self.speech_encoder,
            projector=self.projector,
            language_model=self.language_model,
            tokenizer=self.tokenizer,
        )

    def test_trainable_parameters_isolation(self):
        """Verify that in SLAM-ASR, only projector parameters are trainable."""
        for p in self.language_model.parameters():
            p.requires_grad = False
        for p in self.speech_encoder.parameters():
            p.requires_grad = False

        trainable = self.model.get_trainable_parameters()
        projector_params = list(self.model.projector.parameters())

        self.assertEqual(len(trainable), len(projector_params))
        for p1, p2 in zip(trainable, projector_params):
            self.assertTrue(torch.equal(p1, p2))

    def test_forward_pass_and_loss(self):
        batch_size = 2
        speech_inputs = torch.randn(batch_size, 16000)
        transcripts = ["the quick brown fox", "jumps over the lazy dog"]

        outputs = self.model(
            speech_inputs=speech_inputs,
            transcripts=transcripts,
        )

        self.assertIn("loss", outputs)
        self.assertIn("logits", outputs)
        self.assertIsNotNone(outputs["loss"])
        self.assertTrue(outputs["loss"].item() > 0)

        # Check backward pass on loss
        outputs["loss"].backward()
        for p in self.model.projector.parameters():
            self.assertIsNotNone(p.grad)

    def test_label_masking(self):
        """Verify that prefix and speech tokens have label = -100, and target tokens are retained."""
        speech_embeds = torch.randn(2, 4, 128)
        transcripts = ["hello", "world"]

        _, _, labels = self.model.prepare_multimodal_embeddings(
            speech_embeds=speech_embeds,
            speech_mask=None,
            transcripts=transcripts,
            prompts="Transcribe speech to text.",
            is_training=True,
        )

        self.assertTrue((labels[:, :4] == -100).all())
        self.assertTrue((labels != -100).any())

    def test_generate(self):
        batch_size = 2
        speech_inputs = torch.randn(batch_size, 16000)
        hypotheses = self.model.generate(
            speech_inputs=speech_inputs,
            prompts="Transcribe speech to text.",
            num_beams=4,
        )
        self.assertEqual(len(hypotheses), batch_size)
        for hyp in hypotheses:
            self.assertEqual(hyp, "dummy transcription")


    def test_lora_config_parameters(self):
        """Verify LoRA configuration fields exist on SLAMASRConfig."""
        config = SLAMASRConfig(use_lora=True, lora_r=8, lora_alpha=16)
        self.assertTrue(config.use_lora)
        self.assertEqual(config.lora_r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertIn("q_proj", config.lora_target_modules)


if __name__ == "__main__":
    unittest.main()
