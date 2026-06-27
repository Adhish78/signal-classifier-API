from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the Signal Classifier API.

    Uses Pydantic Settings to automatically parse and validate configuration.
    This provides type safety (e.g. verifying port is an integer) and
    environment variable override capabilities.
    """

    app_name: str = "Signal Classifier API"
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/classifier.onnx"

    # Configure Pydantic to read environment variables prefixed with 'SC_'
    # (e.g. SC_PORT=8000) from the local '.env' file, ignoring any extra variables.
    model_config = SettingsConfigDict(
        env_prefix="SC_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
