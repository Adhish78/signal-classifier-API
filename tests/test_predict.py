import math

from fastapi.testclient import TestClient

from api.main import app
from api.routes.metadata import MODULATION_CLASSES

client = TestClient(app)


def test_predict_endpoint_valid_shape() -> None:
    # Arrange & Act: Send a valid float array shape (2, 128) to the predict endpoint.
    valid_iq_data = [[0.1] * 128, [-0.1] * 128]
    response = client.post("/predict", json={"iq_data": valid_iq_data})

    # Assert: Confirm response code is HTTP 200
    assert response.status_code == 200


def test_predict_endpoint_invalid_outer_shape() -> None:
    # Arrange & Act: Send outer channel list lengths other than 2.
    # 1. Invalid outer length (1 instead of 2)
    invalid_iq_data = [[0.1] * 128]
    response1 = client.post("/predict", json={"iq_data": invalid_iq_data})

    # 2. Invalid outer length (3 instead of 2)
    invalid_iq_data = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
    response2 = client.post("/predict", json={"iq_data": invalid_iq_data})

    # Assert: Verify that both requests fail-fast at the schema gateway with HTTP 422.
    assert response1.status_code == 422
    assert response2.status_code == 422


def test_predict_endpoint_invalid_inner_shape() -> None:
    # Arrange & Act: Send inner sequence lengths other than 128.
    # Channel 1 has 128 elements, but Channel 2 has only 127 elements.
    invalid_iq_data = [[0.1] * 128, [0.2] * 127]
    response = client.post("/predict", json={"iq_data": invalid_iq_data})

    # Assert: Verify request returns HTTP 422.
    assert response.status_code == 422


def test_predict_endpoint_empty_payload() -> None:
    # Act: Send empty request object
    response = client.post("/predict", json={})

    # Assert: Verify request returns HTTP 422
    assert response.status_code == 422


def test_predict_endpoint_non_numeric_values() -> None:
    # Arrange & Act: Send a payload containing a string instead of numeric floats.
    invalid_iq_data = [[0.1] * 127 + ["not-a-float"], [0.2] * 128]
    response = client.post("/predict", json={"iq_data": invalid_iq_data})

    # Assert: Verify request returns HTTP 422
    assert response.status_code == 422


def test_predict_response_structure_and_logic() -> None:
    # Arrange: Setup valid payload
    valid_iq_data = [[0.1] * 128, [-0.1] * 128]

    # Act: Query predict endpoint
    response = client.post("/predict", json={"iq_data": valid_iq_data})
    assert response.status_code == 200

    # Assert: Verify that the response json maps the correct response structure.
    data = response.json()
    assert "predicted_class" in data
    assert "probabilities" in data

    predicted_class = data["predicted_class"]
    probabilities = data["probabilities"]

    assert predicted_class in MODULATION_CLASSES
    assert isinstance(probabilities, dict)
    assert len(probabilities) == len(MODULATION_CLASSES)

    for cls in MODULATION_CLASSES:
        assert cls in probabilities
        assert isinstance(probabilities[cls], float)
        assert 0.0 <= probabilities[cls] <= 1.0

    # The probabilities must sum to exactly 1.0 (verifying Softmax implementation)
    total_prob = sum(probabilities.values())
    assert math.isclose(total_prob, 1.0, rel_tol=1e-5)

    # Verify that the 'predicted_class' corresponds to the maximum probability value.
    max_prob_class = max(probabilities, key=lambda k: probabilities[k])
    assert predicted_class == max_prob_class


def test_predict_endpoint_uses_model_inference() -> None:
    # Arrange & Act: Query predict endpoint.
    valid_iq_data = [[0.1] * 128, [-0.1] * 128]
    response = client.post("/predict", json={"iq_data": valid_iq_data})
    assert response.status_code == 200

    data = response.json()
    probabilities = data["probabilities"]

    # Assert: Verify the model is executed dynamically (mock model outputs
    # uniform probabilities of ~0.0909 per class) instead of returning
    # hardcoded placeholders.
    assert not math.isclose(probabilities["QPSK"], 1.0, abs_tol=1e-5)

    # Verify the probabilities are close to 1/11 (0.0909) from the test model.
    expected_prob = 1.0 / len(MODULATION_CLASSES)
    for cls in MODULATION_CLASSES:
        assert math.isclose(probabilities[cls], expected_prob, abs_tol=1e-4)
