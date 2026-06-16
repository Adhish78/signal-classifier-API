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
    assert math.isclose(total_prob, 1.0, rel_tol=1e-9)

    max_prob_class = max(probabilities, key=lambda k: probabilities[k])
    assert predicted_class == max_prob_class
