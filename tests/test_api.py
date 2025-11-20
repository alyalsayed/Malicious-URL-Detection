import pytest
from fastapi.testclient import TestClient
from src.api.serve import app
from src.inference import predict
from src.features import feature_engineering


@pytest.fixture
def client():
    """Return a FastAPI TestClient for the app."""
    return TestClient(app)


@pytest.fixture
def mock_features():
    """Standard mock feature dictionary with all 27 features."""
    return {
        "url_length": 25, "hostname_length": 15, "count_letters": 18,
        "count_digits": 1, "count_@": 0, "count_?": 1, "count_-": 0,
        "count_=": 1, "count_.": 2, "count_#": 0, "count_%": 0,
        "count_+": 0, "count_$": 0, "count_!": 0, "count_*": 0,
        "count_,": 0, "count_slashes": 3, "count_www": 0, "has_ip": 0,
        "abnormal_url": 0, "short_url": 0, "https": 1, "count_dir": 1,
        "count_embed_domain": 0, "fd_length": 7, "tld_length": 3, "suspicious": 0,
    }


@pytest.fixture
def setup_mocks(monkeypatch, mock_features):
    """Setup all necessary mocks for predict endpoint."""
    # Mock feature extraction
    monkeypatch.setattr(feature_engineering, "extract_features", lambda url: mock_features)
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return [0]
        def predict_proba(self, X):
            return [[0.9, 0.05, 0.03, 0.02]]
    monkeypatch.setattr(predict, "model", MockModel())
    
    # Mock label encoder
    class MockLabelEncoder:
        classes_ = ["benign", "defacement", "malware", "phishing"]
    encoder = MockLabelEncoder()
    monkeypatch.setattr(predict, "label_encoder", encoder)
    monkeypatch.setattr(predict, "label_map", {i: label for i, label in enumerate(encoder.classes_)})


def test_predict_endpoint_happy_path(client, setup_mocks):
    """Test valid POST /predict request returns correct structure."""
    resp = client.post("/predict", json={"url": "https://example.com"})
    
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["input_url"] == "https://example.com"
    assert data["predicted_class"] == "benign"
    assert data["class_id"] == 0
    assert isinstance(data["probabilities"], dict)
    assert all(cls in data["probabilities"] for cls in ["benign", "defacement", "malware", "phishing"])
    assert data["probabilities"]["benign"] == 0.9


@pytest.mark.parametrize("class_id,predicted_class,proba", [
    (0, "benign", [0.9, 0.05, 0.03, 0.02]),
    (1, "defacement", [0.1, 0.7, 0.15, 0.05]),
    (2, "malware", [0.05, 0.1, 0.8, 0.05]),
    (3, "phishing", [0.03, 0.02, 0.05, 0.9]),
])
def test_predict_different_classes(client, monkeypatch, mock_features, class_id, predicted_class, proba):
    """Test API returns different prediction classes correctly."""
    monkeypatch.setattr(feature_engineering, "extract_features", lambda url: mock_features)
    
    class MockModel:
        def predict(self, X):
            return [class_id]
        def predict_proba(self, X):
            return [proba]
    monkeypatch.setattr(predict, "model", MockModel())
    
    class MockLabelEncoder:
        classes_ = ["benign", "defacement", "malware", "phishing"]
    encoder = MockLabelEncoder()
    monkeypatch.setattr(predict, "label_encoder", encoder)
    monkeypatch.setattr(predict, "label_map", {i: l for i, l in enumerate(encoder.classes_)})
    
    resp = client.post("/predict", json={"url": "https://test.com"})
    
    assert resp.status_code == 200
    assert resp.json()["predicted_class"] == predicted_class
    assert resp.json()["class_id"] == class_id


def test_predict_missing_url_returns_422(client):
    """Missing 'url' field returns 422 validation error."""
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_invalid_json_returns_422(client):
    """Invalid JSON body returns 422."""
    resp = client.post("/predict", content="{bad json}", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_predict_model_error_returns_500(client, monkeypatch, mock_features):
    """Model prediction error returns 500."""
    monkeypatch.setattr(feature_engineering, "extract_features", lambda url: mock_features)
    
    class BrokenModel:
        def predict(self, X):
            raise RuntimeError("Model failed")
    monkeypatch.setattr(predict, "model", BrokenModel())
    
    class MockLabelEncoder:
        classes_ = ["benign", "defacement", "malware", "phishing"]
    encoder = MockLabelEncoder()
    monkeypatch.setattr(predict, "label_encoder", encoder)
    monkeypatch.setattr(predict, "label_map", {i: l for i, l in enumerate(encoder.classes_)})
    
    resp = client.post("/predict", json={"url": "http://test.com"})
    assert resp.status_code == 500


def test_predict_wrong_method_returns_405(client):
    """GET /predict returns 405."""
    resp = client.get("/predict")
    assert resp.status_code == 405


def test_predict_extra_fields_ignored(client, setup_mocks):
    """Extra fields in request are ignored."""
    resp = client.post("/predict", json={"url": "https://ok.com", "extra": "value"})
    assert resp.status_code == 200
    assert resp.json()["input_url"] == "https://ok.com"


def test_predict_without_proba_support(client, monkeypatch, mock_features):
    """Model without predict_proba returns string message."""
    monkeypatch.setattr(feature_engineering, "extract_features", lambda url: mock_features)
    
    class ModelWithoutProba:
        def predict(self, X):
            return [0]
        def predict_proba(self, X):
            raise AttributeError("No proba")
    monkeypatch.setattr(predict, "model", ModelWithoutProba())
    
    class MockLabelEncoder:
        classes_ = ["benign", "defacement", "malware", "phishing"]
    encoder = MockLabelEncoder()
    monkeypatch.setattr(predict, "label_encoder", encoder)
    monkeypatch.setattr(predict, "label_map", {i: l for i, l in enumerate(encoder.classes_)})
    
    resp = client.post("/predict", json={"url": "https://test.com"})
    
    assert resp.status_code == 200
    assert isinstance(resp.json()["probabilities"], str)
    assert "does not support" in resp.json()["probabilities"]


def test_predict_probabilities_sum_to_one(client, setup_mocks):
    """Probabilities should sum to approximately 1.0."""
    resp = client.post("/predict", json={"url": "https://test.com"})
    
    assert resp.status_code == 200
    probs = resp.json()["probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01


def test_predict_url_with_special_characters(client, setup_mocks):
    """URLs with special characters are handled correctly."""
    special_url = "https://example.com/path?query=test&param=value#fragment"
    resp = client.post("/predict", json={"url": special_url})
    
    assert resp.status_code == 200
    assert resp.json()["input_url"] == special_url


def test_predict_url_with_ip_address(client, monkeypatch):
    """URL with IP address sets has_ip=1."""
    features = {
        "url_length": 20, "hostname_length": 11, "count_letters": 4,
        "count_digits": 10, "count_@": 0, "count_?": 0, "count_-": 0,
        "count_=": 0, "count_.": 3, "count_#": 0, "count_%": 0,
        "count_+": 0, "count_$": 0, "count_!": 0, "count_*": 0,
        "count_,": 0, "count_slashes": 2, "count_www": 0, "has_ip": 1,
        "abnormal_url": 0, "short_url": 0, "https": 0, "count_dir": 1,
        "count_embed_domain": 0, "fd_length": 4, "tld_length": 0, "suspicious": 0,
    }
    monkeypatch.setattr(feature_engineering, "extract_features", lambda url: features)
    
    class MockModel:
        def predict(self, X):
            return [0]
        def predict_proba(self, X):
            return [[0.9, 0.05, 0.03, 0.02]]
    monkeypatch.setattr(predict, "model", MockModel())
    
    class MockLabelEncoder:
        classes_ = ["benign", "defacement", "malware", "phishing"]
    encoder = MockLabelEncoder()
    monkeypatch.setattr(predict, "label_encoder", encoder)
    monkeypatch.setattr(predict, "label_map", {i: l for i, l in enumerate(encoder.classes_)})
    
    resp = client.post("/predict", json={"url": "http://192.168.1.1/path"})
    assert resp.status_code == 200


def test_predict_unknown_class_id(client, monkeypatch, mock_features):
    """Unknown class_id returns 'unknown' label."""
    monkeypatch.setattr(feature_engineering, "extract_features", lambda url: mock_features)
    
    class MockModel:
        def predict(self, X):
            return [999]
        def predict_proba(self, X):
            return [[0.25, 0.25, 0.25, 0.25]]
    monkeypatch.setattr(predict, "model", MockModel())
    
    class MockLabelEncoder:
        classes_ = ["benign", "defacement", "malware", "phishing"]
    encoder = MockLabelEncoder()
    monkeypatch.setattr(predict, "label_encoder", encoder)
    monkeypatch.setattr(predict, "label_map", {i: l for i, l in enumerate(encoder.classes_)})
    
    resp = client.post("/predict", json={"url": "https://test.com"})
    
    assert resp.status_code == 200
    assert resp.json()["predicted_class"] == "unknown"
    assert resp.json()["class_id"] == 999