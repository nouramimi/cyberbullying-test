import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, average_precision_score,
                             classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from . import config
from .cleaning import clean_text


def _tfidf():
    return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, strip_accents="unicode")


def build_models() -> dict:
    return {
        "naive_bayes": (
            Pipeline([("tfidf", _tfidf()), ("clf", MultinomialNB())]),
            {"clf__alpha": [0.05, 0.1, 0.3, 1.0]},
        ),
        "logreg": (
            Pipeline([("tfidf", _tfidf()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE))]),
            {"clf__C": [0.5, 1, 5, 10, 30]},
        ),
    }


def mcnemar(y, p1, p2) -> dict:
    c1, c2 = (p1 == y), (p2 == y)
    b = int(np.sum(c1 & ~c2)); c = int(np.sum(~c1 & c2))
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    return {"only_nb_correct": b, "only_logreg_correct": c, "statistic": round(float(stat), 4),
            "p_value": round(float(chi2.sf(stat, 1)), 4)}


def train_and_evaluate(df: pd.DataFrame, save_models=True) -> dict:
    X, y = df["text"], df["label_binary"]
    idx_tr, idx_te = train_test_split(df.index, test_size=config.TEST_SIZE, stratify=y,
                                      random_state=config.RANDOM_STATE)
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    results, preds = {}, {}
    for name, (pipe, grid) in build_models().items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1).fit(Xtr, ytr)
        fit_s = time.time() - t0
        best = gs.best_estimator_
        t0 = time.time(); proba = best.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
        pred_ms = (time.time() - t0) / len(Xte) * 1000
        fpr, tpr, _ = roc_curve(yte, proba)
        results[name] = {
            "best_params": gs.best_params_,
            "cv_f1_mean": round(gs.best_score_, 4),
            "cv_f1_std": round(float(gs.cv_results_["std_test_score"][gs.best_index_]), 4),
            "accuracy": round(accuracy_score(yte, pred), 4),
            "precision": round(precision_score(yte, pred), 4),
            "recall": round(recall_score(yte, pred), 4),
            "f1": round(f1_score(yte, pred), 4),
            "f1_macro": round(f1_score(yte, pred, average="macro"), 4),
            "roc_auc": round(roc_auc_score(yte, proba), 4),
            "avg_precision": round(average_precision_score(yte, proba), 4),
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
            "train_time_s": round(fit_s, 2),
            "predict_ms_per_sample": round(pred_ms, 4),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "report": classification_report(yte, pred, target_names=["Not-Bullying", "Bullying"], output_dict=True),
        }
        preds[name] = pred
        # performance par catégorie de harcèlement (rappel sur les cas Bullying)
        te = df.loc[idx_te].assign(pred=pred)
        bt = te[te["label_binary"] == 1]
        results[name]["recall_by_category"] = (bt["pred"] == 1).groupby(bt["category"]).mean().round(4).to_dict()
        if save_models:
            config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
            joblib.dump(best, config.MODELS_DIR / f"{name}.joblib")
        results[name]["_pipeline"] = best

    results["mcnemar"] = mcnemar(yte.values, preds["naive_bayes"], preds["logreg"])
    results["split"] = {"train": len(Xtr), "test": len(Xte),
                        "test_bullying_share": round(float(yte.mean()), 4)}
    results["_y_test"] = yte
    return results


def top_features(pipe, n=20) -> pd.DataFrame:
    """Termes les plus influents de la régression logistique (coefficients)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)
    return pd.concat([
        pd.DataFrame({"term": names[order[:n]], "coef": coef[order[:n]], "class": "Not-Bullying"}),
        pd.DataFrame({"term": names[order[-n:][::-1]], "coef": coef[order[-n:][::-1]], "class": "Bullying"}),
    ], ignore_index=True)


def save_metrics(results: dict, path=None):
    path = path or (config.REPORTS_DIR / "model_metrics.json")
    clean = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
             for k, v in results.items() if not k.startswith("_")}
    for k in ("naive_bayes", "logreg"):
        clean[k].pop("roc_curve", None)
    config.REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
