import os
from api.config import Settings


def test_settings_default_values():
    # Clear any SC_ prefixed environment variables to test defaults
    for key in list(os.environ.keys()):
        if key.startswith("SC_"):
            del os.environ[key]

    settings = Settings()
    assert settings.app_name == "Signal Classifier API"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.model_path == "models/classifier.onnx"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("SC_APP_NAME", "Custom Signal API")
    monkeypatch.setenv("SC_HOST", "127.0.0.1")
    monkeypatch.setenv("SC_PORT", "9090")
    monkeypatch.setenv("SC_MODEL_PATH", "models/test_model.onnx")

    settings = Settings()
    assert settings.app_name == "Custom Signal API"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9090
    assert settings.model_path == "models/test_model.onnx"
