"""
Unit tests for the FastAPI application.
Run with: pytest tests/test_api.py -v
"""
from app.api import app
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

client = TestClient(app)


class TestAPIEndpoints:
    """Test all API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Boston Housing" in data["message"]
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "model_metrics" in data
        assert "timestamp" in data

    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid input."""
        valid_input = {
            'CRIM': 0.00632, 'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
            'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
            'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
        }

        response = client.post("/predict", json=valid_input)

        if response.status_code == 200:
            data = response.json()
            assert "predicted_price" in data
            assert "predicted_price_dollars" in data
            assert "prediction_time" in data
            assert "model_version" in data
            assert data["predicted_price"] > 0
            assert data["predicted_price_dollars"] > 0
        elif response.status_code == 503:
            assert "not available" in response.json()["detail"].lower()

    def test_predict_endpoint_invalid_input(self):
        """Test prediction with invalid CRIM value."""
        invalid_input = {
            'CRIM': -0.00632,  # Negative value not allowed
            'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
            'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
            'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
        }

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
        error_detail = response.json()
        assert "detail" in error_detail

    def test_predict_endpoint_missing_field(self):
        """Test prediction with missing required field."""
        incomplete_input = {
            'CRIM': 0.00632,
            'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
            'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
            'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90
            # Missing 'LSTAT'
        }

        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422

    def test_batch_predict_valid(self):
        """Test batch prediction with valid inputs."""
        valid_batch = {
            "houses": [
                {
                    'CRIM': 0.00632,
                    'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
                    'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
                    'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
                },
                {
                    'CRIM': 0.632,
                    'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
                    'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
                    'RAD': 2, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
                }
            ]
        }

        response = client.post("/predict/batch", json=valid_batch)

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2
            for prediction in data:
                assert "predicted_price" in prediction
                assert "predicted_price_dollars" in prediction
                assert "prediction_time" in prediction
                assert "model_version" in prediction
        elif response.status_code == 503:
            assert "not available" in response.json()["detail"].lower()

    def test_batch_predict_too_many(self):
        """Test batch prediction with too many houses."""
        large_batch = {
            "houses": [
                {
                    'CRIM': 0.00632,
                    'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
                    'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
                    'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
                }
            ] * 101
        }

        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 422
        error_detail = response.json()
        assert "detail" in error_detail
        detail_str = str(error_detail["detail"]).lower()
        assert "100" in detail_str or "maximum" in detail_str

    def test_batch_predict_empty(self):
        """Test batch prediction with empty list."""
        empty_batch = {"houses": []}
        response = client.post("/predict/batch", json=empty_batch)
        assert response.status_code == 422


class TestInputValidation:
    """Test input validation logic."""

    def test_negative_crim(self):
        """Test that negative CRIM value is rejected."""
        invalid_input = {
            'CRIM': -0.5,
            'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
            'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
            'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
        }

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_invalid_rm(self):
        """Test that invalid RM value is rejected."""
        invalid_input = {
            'CRIM': 0.00632, 'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
            'NOX': 0.538, 'RM': 0.0,  # Invalid number of rooms
            'AGE': 65.2, 'DIS': 4.09, 'RAD': 1,
            'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
        }

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json(self):
        """Test with invalid JSON."""
        response = client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_wrong_data_types(self):
        """Test with wrong data types."""
        invalid_input = {
            'CRIM': "not a number", 'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
            'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
            'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
        }

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])