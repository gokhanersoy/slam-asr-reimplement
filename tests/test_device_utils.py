"""Unit tests for hardware-aware device_utils."""

import unittest
from slam_asr.utils.device_utils import (
    get_device_specs,
    get_recommended_hyperparameters,
    resolve_auto_hyperparameters,
)


class TestDeviceUtils(unittest.TestCase):
    def test_get_device_specs(self):
        specs = get_device_specs()
        self.assertIn("device_type", specs)
        self.assertIn("device_name", specs)
        self.assertIn("vram_gb", specs)
        self.assertIn("supports_bf16", specs)
        self.assertIn("supports_fp16", specs)

    def test_h100_tier_recommendations(self):
        rec = get_recommended_hyperparameters(
            vram_gb=80.0,
            compute_capability=(9, 0),
            supports_bf16=True,
            device_type="cuda",
        )
        self.assertFalse(rec["load_in_4bit"])
        self.assertEqual(rec["torch_dtype"], "bfloat16")
        self.assertEqual(rec["batch_size"], 16)
        self.assertEqual(rec["gradient_accumulation_steps"], 1)
        self.assertTrue(rec["bf16"])
        self.assertFalse(rec["fp16"])

    def test_a100_40gb_tier_recommendations(self):
        rec = get_recommended_hyperparameters(
            vram_gb=40.0,
            compute_capability=(8, 0),
            supports_bf16=True,
            device_type="cuda",
        )
        self.assertFalse(rec["load_in_4bit"])
        self.assertEqual(rec["torch_dtype"], "bfloat16")
        self.assertEqual(rec["batch_size"], 8)
        self.assertEqual(rec["gradient_accumulation_steps"], 2)
        self.assertTrue(rec["bf16"])

    def test_l4_tier_recommendations(self):
        rec = get_recommended_hyperparameters(
            vram_gb=24.0,
            compute_capability=(8, 9),
            supports_bf16=True,
            device_type="cuda",
        )
        self.assertTrue(rec["load_in_4bit"])
        self.assertEqual(rec["torch_dtype"], "bfloat16")
        self.assertEqual(rec["batch_size"], 8)
        self.assertEqual(rec["gradient_accumulation_steps"], 2)
        self.assertTrue(rec["bf16"])

    def test_t4_tier_recommendations(self):
        rec = get_recommended_hyperparameters(
            vram_gb=15.0,
            compute_capability=(7, 5),
            supports_bf16=False,
            device_type="cuda",
        )
        self.assertTrue(rec["load_in_4bit"])
        self.assertEqual(rec["torch_dtype"], "float16")
        self.assertEqual(rec["batch_size"], 4)
        self.assertEqual(rec["gradient_accumulation_steps"], 4)
        self.assertTrue(rec["fp16"])
        self.assertFalse(rec["bf16"])

    def test_cpu_tier_recommendations(self):
        rec = get_recommended_hyperparameters(
            vram_gb=0.0,
            device_type="cpu",
        )
        self.assertFalse(rec["load_in_4bit"])
        self.assertEqual(rec["torch_dtype"], "float32")
        self.assertEqual(rec["batch_size"], 2)
        self.assertFalse(rec["fp16"])
        self.assertFalse(rec["bf16"])

    def test_resolve_auto_hyperparameters(self):
        model_cfg = {"torch_dtype": "auto", "load_in_4bit": "auto"}
        data_cfg = {"batch_size": "auto", "num_workers": "auto"}
        train_cfg = {"gradient_accumulation_steps": "auto", "fp16": "auto", "bf16": "auto"}

        m_out, d_out, t_out = resolve_auto_hyperparameters(
            model_cfg, data_cfg, train_cfg, verbose=False
        )

        self.assertNotEqual(m_out["torch_dtype"], "auto")
        self.assertIsInstance(m_out["load_in_4bit"], bool)
        self.assertIsInstance(d_out["batch_size"], int)
        self.assertIsInstance(t_out["gradient_accumulation_steps"], int)


if __name__ == "__main__":
    unittest.main()
