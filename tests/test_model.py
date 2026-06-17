import torch

from src.model import SignalClassifier


def test_model_output_shape() -> None:
    # Arrange
    batch_size = 8
    model = SignalClassifier(num_classes=11)
    
    # 2 channels (I & Q), 128 time steps
    dummy_input = torch.randn(batch_size, 2, 128)
    
    # Act
    logits = model(dummy_input)
    
    # Assert
    assert logits.shape == (batch_size, 11)


def test_model_no_nan_outputs() -> None:
    # Arrange
    batch_size = 4
    model = SignalClassifier(num_classes=11)
    dummy_input = torch.randn(batch_size, 2, 128)
    
    # Act
    logits = model(dummy_input)
    
    # Assert
    assert not torch.isnan(logits).any(), "Model logits should not contain NaNs"
    assert not torch.isinf(logits).any(), "Model logits should not contain Infs"

