#!/usr/bin/env python3
"""Pipeline complet : exploration -> nettoyage -> MongoDB -> analyse -> modèles -> comparaison -> figures.

Exemples :
  python run_pipeline.py                         # MongoDB sur MONGO_URI (défaut localhost:27017)
  python run_pipeline.py --skip-mongo            # sans MongoDB (utilisé au build Docker)
  MONGO_URI=mongomock:// python run_pipeline.py  # test sans serveur
"""
import argparse
import json

import pandas as pd

from app import analysis, config, mongo_utils, modeling, visualize
from app.cleaning import clean, explore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.RAW_CSV))
    ap.add_argument("--mongo-uri", default=config.MONGO_URI)
    ap.add_argument("--skip-mongo", action="store_true")
    a = ap.parse_args()
    config.REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    print("[1/6] Exploration et nettoyage")
    raw = pd.read_csv(a.csv)
    exp = explore(raw)
    df, log = clean(raw, )
    (config.REPORTS_DIR / "exploration.json").write_text(json.dumps({"raw": exp, "cleaning_log": log}, indent=2, default=str))
    df.to_csv(config.CLEAN_CSV, index=False)
    print("      ", log)

    print("[2/6] Chargement MongoDB")
    if a.skip_mongo:
        print("       ignoré (--skip-mongo)")
    else:
        coll = mongo_utils.get_collection(a.mongo_uri)
        n = mongo_utils.load_dataframe(df, coll)
        print(f"       {n} documents insérés dans {config.MONGO_DB}.{config.MONGO_COLLECTION}")
        print("       répartition (agrégation Mongo):", mongo_utils.aggregate_counts(coll, "label"))
        df = mongo_utils.read_dataframe(coll)  # les étapes suivantes lisent depuis MongoDB

    print("[3/6] Analyse descriptive")
    stats = analysis.descriptive(df)
    analysis.save(stats)

    print("[4/6] Entraînement des deux modèles")
    res = modeling.train_and_evaluate(df)
    modeling.save_metrics(res)

    print("[5/6] Comparaison")
    for n in ("naive_bayes", "logreg"):
        r = res[n]
        print(f"       {n:12s} acc={r['accuracy']} P={r['precision']} R={r['recall']} F1={r['f1']} AUC={r['roc_auc']}")
    print("       McNemar:", res["mcnemar"])

    print("[6/6] Figures")
    visualize.descriptive_figures(df)
    visualize.model_figures(res)
    print("Terminé. Rapports : reports/ | Modèles : models/")


if __name__ == "__main__":
    main()
