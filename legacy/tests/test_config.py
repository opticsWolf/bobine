"""Tests for ConverterConfig and RoutingMode (ported from OKFgraph)."""

from bobine.config import ConverterConfig, RoutingMode


class TestConfig:
    def test_default_config(self):
        cfg = ConverterConfig()
        assert cfg.routing_mode == RoutingMode.AUTO
        assert cfg.use_onnx is True
        assert cfg.device == "cuda"
        assert cfg.ort_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_cuda_providers(self):
        cfg = ConverterConfig(device="cuda")
        assert "CUDAExecutionProvider" in cfg.ort_providers
        assert "CPUExecutionProvider" in cfg.ort_providers

    def test_gpu_alias_providers(self):
        """device='gpu' is accepted as an alias for 'cuda'."""
        cfg = ConverterConfig(device="gpu")
        assert "CUDAExecutionProvider" in cfg.ort_providers
        assert "CPUExecutionProvider" in cfg.ort_providers

    def test_cpu_providers(self):
        cfg = ConverterConfig(device="cpu")
        assert cfg.ort_providers == ["CPUExecutionProvider"]

    def test_explicit_providers(self):
        cfg = ConverterConfig(ort_providers=["DirectMLExecutionProvider"])
        assert cfg.ort_providers == ["DirectMLExecutionProvider"]

    def test_routing_mode_values(self):
        assert RoutingMode.AUTO.value == "auto"
        assert RoutingMode.SURGICAL.value == "surgical"
        assert RoutingMode.ALWAYS.value == "always"
        assert RoutingMode.NEVER.value == "never"
