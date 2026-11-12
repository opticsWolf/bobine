"""The shared device/precision vocabulary (bobine 0.9, plan B3).

The mapping itself lives in Rust (`src/config.rs` `runtime_plan` /
`formula_variant`, unit-tested there). These tests cover the Python
surface: the exposed words, what a word resolves to, the refusals, and
`describe()`.
"""
from __future__ import annotations

import unittest

import bobine


class Words(unittest.TestCase):
    def test_exposed_tuples_are_the_source_of_truth(self):
        """okfgraph validates against these, so they must not drift."""
        self.assertEqual(bobine.DEVICES, ("auto", "cpu", "cuda"))
        self.assertEqual(bobine.PRECISIONS, ("auto", "fp32", "fp16", "int8"))
        self.assertNotIn("q8", bobine.PRECISIONS)

    def test_unknown_device_names_the_set(self):
        with self.assertRaises(ValueError) as err:
            bobine.ConverterConfig(device="gpu2")
        self.assertIn("auto, cpu, cuda", str(err.exception))
        self.assertIn("gpu2", str(err.exception))

    def test_q8_is_refused_by_name(self):
        with self.assertRaises(ValueError) as err:
            bobine.ConverterConfig(precision="q8")
        self.assertIn("no q8 artifact", str(err.exception))
        self.assertIn("int8", str(err.exception))


class Resolution(unittest.TestCase):
    def test_defaults_are_auto_and_keep_the_legacy_variant(self):
        cfg = bobine.ConverterConfig()
        self.assertEqual(cfg.device, "auto")
        self.assertEqual(cfg.precision, "auto")
        self.assertEqual(cfg.model_precision, bobine.ModelPrecision.Fp32)
        self.assertEqual(cfg.model_quantization, bobine.ModelQuantization.Fp32)

    def test_precision_resolves_the_tex_teller_variant(self):
        fp16 = bobine.ConverterConfig(precision="fp16")
        self.assertEqual(fp16.precision, "fp16")
        self.assertEqual(fp16.model_precision, bobine.ModelPrecision.Fp16)

        int8 = bobine.ConverterConfig(precision="int8")
        self.assertEqual(int8.model_quantization, bobine.ModelQuantization.Int8)

        # The legacy name still works and still means what it did.
        legacy = bobine.ConverterConfig(model_precision=bobine.ModelPrecision.Fp16)
        self.assertEqual(legacy.precision, "auto")
        self.assertEqual(legacy.model_precision, bobine.ModelPrecision.Fp16)

    def test_setting_both_is_a_conflict(self):
        with self.assertRaises(ValueError) as err:
            bobine.ConverterConfig(
                precision="fp16",
                model_quantization=bobine.ModelQuantization.Int8,
            )
        self.assertIn("conflicts", str(err.exception))


class Describe(unittest.TestCase):
    def test_describe_before_any_session(self):
        """No ONNX pass has opened, so an auto slot reports its policy."""
        conv = bobine.HybridConverter(
            bobine.ConverterConfig(), cache_dir="~/.cache/bobine"
        )
        d = conv.describe()
        self.assertEqual(d["device_requested"], "auto")
        self.assertEqual(d["precision_requested"], "auto")
        self.assertEqual(
            set(d["passes"]),
            {"encoder", "decoder", "layout", "ocr", "table", "potatr"},
        )
        for name, pass_ in d["passes"].items():
            self.assertFalse(pass_["landed"], f"{name} must not be loaded")
            self.assertIn("policy", pass_)
            self.assertIn("precision", pass_)
        # Auto-CUDA slots cannot answer the list yet; the pinned ones can.
        self.assertIsNone(d["passes"]["layout"]["providers"])
        self.assertEqual(d["passes"]["layout"]["policy"], "auto-cuda")
        self.assertEqual(
            d["passes"]["table"]["providers"], ["CPUExecutionProvider"]
        )
        self.assertEqual(d["passes"]["table"]["policy"], "cpu-pinned")

    def test_cpu_device_makes_every_list_concrete(self):
        cfg = bobine.ConverterConfig(device="cpu")
        d = bobine.HybridConverter(cfg, cache_dir="~/.cache/bobine").describe()
        self.assertEqual(d["device_requested"], "cpu")
        for name, pass_ in d["passes"].items():
            self.assertEqual(pass_["providers"], ["CPUExecutionProvider"], name)

    def test_cuda_device_lists_cuda_first_everywhere(self):
        cfg = bobine.ConverterConfig(device="cuda", precision="fp16")
        d = bobine.HybridConverter(cfg, cache_dir="~/.cache/bobine").describe()
        for name, pass_ in d["passes"].items():
            self.assertEqual(
                pass_["providers"],
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                name,
            )
        self.assertEqual(d["passes"]["encoder"]["precision"], "fp16")
        self.assertEqual(d["passes"]["layout"]["precision"], "fp32")


if __name__ == "__main__":
    unittest.main()
