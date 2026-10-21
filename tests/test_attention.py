from torch.nn.attention import SDPBackend

from irodori_tts import attention


def test_sdpa_cudnn_can_be_disabled(monkeypatch):
    assert attention._sdpa_priority()[0] is SDPBackend.CUDNN_ATTENTION
    monkeypatch.setenv("IRODORI_SDPA_CUDNN", "0")
    assert SDPBackend.CUDNN_ATTENTION not in attention._sdpa_priority()
