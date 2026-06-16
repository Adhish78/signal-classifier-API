import math

from fastapi.testclient import TestClient

from api.main import app
from api.routes.metadata import MODULATION_CLASSES

client = TestClient(app)


def test_predict_endpoint_valid_shape() -> None:
    # Valid shape (2, 128)
    valid_iq_data = [[0.1] * 128, [-0.1] * 128]
    response = client.post("/predict", json={"iq_data": valid_iq_data})
    assert response.status_code == 200


def test_predict_endpoint_invalid_outer_shape() -> None:
    # Invalid outer length (1 instead of 2)
    invalid_iq_data = [[0.1] * 128]
    response = client.post("/predict", json={"iq_data": invalid_iq_data})
    assert response.status_code == 422

    # Invalid outer length (3 instead of 2)
    invalid_iq_data = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
    response = client.post("/predict", json={"iq_data": invalid_iq_data})
    assert response.status_code == 422


def test_predict_endpoint_invalid_inner_shape() -> None:
    # Invalid inner length of one sequence (127 instead of 128)
    invalid_iq_data = [[0.1] * 128, [0.2] * 127]
    response = client.post("/predict", json={"iq_data": invalid_iq_data})
    assert response.status_code == 422


def test_predict_endpoint_empty_payload() -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_endpoint_non_numeric_values() -> None:
    # Contains a string instead of a float
    invalid_iq_data = [[0.1] * 127 + ["not-a-float"], [0.2] * 128]
    response = client.post("/predict", json={"iq_data": invalid_iq_data})
    assert response.status_code == 422


def test_predict_response_structure_and_logic() -> None:
    valid_iq_data = [[0.1] * 128, [-0.1] * 128]
    response = client.post("/predict", json={"iq_data": valid_iq_data})
    assert response.status_code == 200

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

    total_prob = sum(probabilities.values())
    assert math.isclose(total_prob, 1.0, rel_tol=1e-5)

    max_prob_class = max(probabilities, key=lambda k: probabilities[k])
    assert predicted_class == max_prob_class


def test_predict_endpoint_uses_model_inference() -> None:
    # If the endpoint is integrated with the real test model, the output should
    # be dynamically computed by the model (all classes ~0.0909) instead of
    # the hardcoded dummy values (QPSK=1.0, others=0.0).
    valid_iq_data = [[0.1] * 128, [-0.1] * 128]
    response = client.post("/predict", json={"iq_data": valid_iq_data})
    assert response.status_code == 200

    data = response.json()
    probabilities = data["probabilities"]

    # Assert it is not the hardcoded dummy value
    assert not math.isclose(probabilities["QPSK"], 1.0, abs_tol=1e-5)

    # Verify the probabilities are close to 1/11 (0.0909) from the test model
    expected_prob = 1.0 / len(MODULATION_CLASSES)
    for cls in MODULATION_CLASSES:
        assert math.isclose(probabilities[cls], expected_prob, abs_tol=1e-4)
