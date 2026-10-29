import os
os.environ["MONGO_URI"] = "mongomock://"
from fastapi.testclient import TestClient
from app.api import app


def test_endpoints():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json={"text": "you are a disgusting idiot, shut up", "model": "both"}).json()
        assert set(r["predictions"]) == {"logreg", "naive_bayes"}
        r = c.post("/predict", json={"text": "Your videos bring me so much joy, thank you!"}).json()
        assert r["predictions"]["logreg"]["label"] == "Not-Bullying"
        assert len(c.post("/predict/batch", json={"texts": ["a", "b"]}).json()) == 2
        assert c.post("/predict", json={"text": "x", "model": "zz"}).status_code == 422
        assert "logreg" in c.get("/metrics").json()
        assert "label_counts" in c.get("/stats").json()
