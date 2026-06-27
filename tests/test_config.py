import os

import pytest

from api.config import Settings


def test_settings_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: Clear SC_ prefixed environment variables using monkeypatch
    # to guarantee we are testing the default fallbacks defined in Settings.
    for key in list(os.environ.keys()):
        if key.startswith("SC_"):
            monkeypatch.delenv(key, raising=False)

    # Act: Instantiate Settings
    settings = Settings()

    # Assert: Verify defaults are parsed correctly
    assert settings.app_name == "Signal Classifier API"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.model_path == "models/classifier.onnx"


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: Set environmental variables prefixed with 'SC_'
    monkeypatch.setenv("SC_APP_NAME", "Custom Signal API")
    monkeypatch.setenv("SC_HOST", "127.0.0.1")
    monkeypatch.setenv("SC_PORT", "9090")
    monkeypatch.setenv("SC_MODEL_PATH", "models/test_model.onnx")

    # Act: Instantiate Settings
    settings = Settings()

    # Assert: Verify that Settings dynamically captures environment variables
    # and performs correct type casting (e.g. converting port "9090" string to integer).
    assert settings.app_name == "Custom Signal API"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9090
    assert settings.model_path == "models/test_model.onnx"
