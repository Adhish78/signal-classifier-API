import torch

from src.model import SignalClassifier


def test_model_output_shape() -> None:
    # Arrange: Setup batch parameters and 1D CNN model.
    # The dummy input replicates a batch of 8 signal recordings,
    # each containing 2 channels (I & Q) tracked over 128 time-steps.
    batch_size = 8
    model = SignalClassifier(num_classes=11)
    dummy_input = torch.randn(batch_size, 2, 128)

    # Act: Run forward pass of the network
    logits = model(dummy_input)

    # Assert: Verify that the network resolves the 128 time steps down to
    # a final logits vector matching the 11 target modulation classes.
    assert logits.shape == (batch_size, 11)


def test_model_no_nan_outputs() -> None:
    # Arrange: Initialize model and input tensor
    batch_size = 4
    model = SignalClassifier(num_classes=11)
    dummy_input = torch.randn(batch_size, 2, 128)

    # Act: Compute logits
    logits = model(dummy_input)

    # Assert: Verify numerical stability.
    # The output logits must not contain NaN or Inf values, which would
    # indicate exploding/vanishing gradients or division-by-zero during training.
    assert not torch.isnan(logits).any(), "Model logits should not contain NaNs"
    assert not torch.isinf(logits).any(), "Model logits should not contain Infs"
