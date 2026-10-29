"""API FastAPI : prédiction de harcèlement social, métriques et statistiques."""
import json
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import config, mongo_utils
from . import cleaning  # noqa: F401  (nécessaire au chargement des pipelines picklés)

MODEL_NAMES = ("logreg", "naive_bayes")
state = {"models": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for n in MODEL_NAMES:
        p = config.MODELS_DIR / f"{n}.joblib"
        if p.exists():
            state["models"][n] = joblib.load(p)
    yield


app = FastAPI(title="Cyberbullying Detection API", version="1.0.0", lifespan=lifespan)


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["you are so stupid"])
    model: str = Field("logreg", description="logreg | naive_bayes | both")


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)
    model: str = "logreg"


def _predict(texts: list[str], model: str):
    if model != "both" and model not in MODEL_NAMES:
        raise HTTPException(422, f"model doit être l'un de {list(MODEL_NAMES)} ou 'both'")
    names = MODEL_NAMES if model == "both" else (model,)
    missing = [n for n in names if n not in state["models"]]
    if missing:
        raise HTTPException(503, f"Modèle(s) non chargé(s): {missing}. Lancez run_pipeline.py")
    out = []
    for i, _ in enumerate(texts):
        out.append({})
    for n in names:
        proba = state["models"][n].predict_proba(texts)[:, 1]
        for i, p in enumerate(proba):
            out[i][n] = {"label": "Bullying" if p >= 0.5 else "Not-Bullying", "probability_bullying": round(float(p), 4)}
    return out


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(state["models"])}


@app.post("/predict")
def predict(body: PredictIn):
    return {"text": body.text, "predictions": _predict([body.text], body.model)[0]}


@app.post("/predict/batch")
def predict_batch(body: BatchIn):
    res = _predict(body.texts, body.model)
    return [{"text": t, "predictions": r} for t, r in zip(body.texts, res)]


def _report(name):
    p = config.REPORTS_DIR / name
    if not p.exists():
        raise HTTPException(404, f"{name} introuvable : lancez run_pipeline.py")
    return json.loads(p.read_text())


@app.get("/metrics")
def metrics():
    """Étude comparative des deux modèles (jeu de test)."""
    return _report("model_metrics.json")


@app.get("/stats")
def stats():
    """Analyse descriptive du jeu nettoyé."""
    return _report("descriptive_stats.json")


@app.get("/comments")
def comments(label: str | None = None, category: str | None = None, limit: int = Query(20, le=200)):
    """Lecture des commentaires nettoyés depuis MongoDB (si disponible)."""
    try:
        coll = mongo_utils.get_collection()
    except Exception as e:  # MongoDB non joignable
        raise HTTPException(503, f"MongoDB indisponible: {e.__class__.__name__}")
    q = {k: v for k, v in {"label": label, "category": category}.items() if v}
    return list(coll.find(q, {"_id": 0, "loaded_at": 0}).limit(limit))
