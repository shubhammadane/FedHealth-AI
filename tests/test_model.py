import torch
from src.model import ReadmissionNet


def test_forward_shape():
    model = ReadmissionNet(20)
    x = torch.randn(4, 20)
    y = model(x)
    assert y.shape == (4,)
