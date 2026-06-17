from pydantic import BaseModel, Field, field_validator


class ModelMetadataResponse(BaseModel):
    model_version: str = Field(..., description="Active model version")
    framework: str = Field(..., description="Framework used to train/serve the model")
    classes: list[str] = Field(
        ..., description="The list of radio frequency modulation classes"
    )
    input_shape: list[int] = Field(
        ..., description="Shape of the input tensor (channels, time-steps)"
    )
    training_accuracy: float = Field(..., description="Training accuracy of the model")
    date_of_training: str = Field(..., description="Date when the model was trained")


EXPECTED_CHANNELS = 2
EXPECTED_TIME_STEPS = 128


class PredictionRequest(BaseModel):
    iq_data: list[list[float]] = Field(
        ..., description="Raw IQ data sample of shape (2, 128)"
    )

    @field_validator("iq_data")
    @classmethod
    def validate_iq_data_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != EXPECTED_CHANNELS:
            raise ValueError(f"iq_data must have exactly {EXPECTED_CHANNELS} channels")
        for i, channel in enumerate(v):
            if len(channel) != EXPECTED_TIME_STEPS:
                raise ValueError(
                    f"Channel {i} must have exactly {EXPECTED_TIME_STEPS} elements"
                )
            for j, val in enumerate(channel):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise ValueError(
                        f"Value at channel {i}, index {j} must be a numeric value"
                    )
        return v


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="The predicted modulation class")
    probabilities: dict[str, float] = Field(
        ..., description="Probability distribution across all 11 modulation classes"
    )


class PredictionMetrics(BaseModel):
    uptime_seconds: float = Field(..., description="Uptime of the service in seconds")
    total_predictions: int = Field(..., description="Total number of prediction requests processed")
    failed_predictions: int = Field(..., description="Total number of failed prediction requests")
    average_inference_time_ms: float = Field(..., description="Average inference execution time in milliseconds")
    min_inference_time_ms: float = Field(..., description="Minimum inference execution time in milliseconds")
    max_inference_time_ms: float = Field(..., description="Maximum inference execution time in milliseconds")

