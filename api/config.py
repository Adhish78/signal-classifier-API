from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Signal Classifier API"
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/classifier.onnx"

    model_config = SettingsConfigDict(
        env_prefix="SC_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
