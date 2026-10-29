#!/usr/bin/env python3
"""
Exploration, nettoyage et chargement MongoDB du dataset
"Approach to Social Media Cyberbullying and Harassment Detection Using Advanced Machine Learning"

Installation :  pip install pandas pymongo
Usage       :  python cyberbullying_etl.py
               python cyberbullying_etl.py --csv "chemin/fichier.csv" --uri mongodb://localhost:27017 \
                      --db cyberbullying_db --collection comments --drop
Options     :  --text-col / --label-col pour forcer les colonnes (sinon auto-détection)
               --drop-unlabeled pour supprimer les lignes sans label
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone

import pandas as pd

DEFAULT_CSV = "data/raw.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")


def read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Encodage du CSV non reconnu")


def detect_columns(df, text_col, label_col):
    if text_col is None and "Text" in df.columns:
        text_col = "Text"
    if label_col is None and "Label" in df.columns:
        label_col = "Label"
    obj = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    if not obj:
        sys.exit("Aucune colonne texte détectée.")
    if text_col is None:
        text_col = max(obj, key=lambda c: df[c].dropna().astype(str).str.len().mean())
    if label_col is None:
        cands = [c for c in obj if c != text_col and 1 < df[c].nunique() <= 30]
        label_col = min(cands, key=lambda c: df[c].nunique()) if cands else None
    return text_col, label_col


def explore(df, title):
    print(f"\n===== {title} =====")
    print("Dimensions :", df.shape)
    print("\nTypes :\n", df.dtypes.to_string())
    nulls = df.isna().sum()
    print("\nValeurs manquantes :\n", pd.DataFrame({"n": nulls, "%": (nulls / len(df) * 100).round(1)}).to_string())
    print("\nLignes dupliquées (toutes colonnes) :", int(df.duplicated().sum()))
    print("\nÉchantillon :\n", df.head(5).to_string())


def clean_text(s):
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip().lower()
    return s


CAT_MAP = {"sexual": "Sexual", "saxual": "Sexual", "troll": "Troll", "political": "Political",
           "vocational": "Vocational", "vocation": "Vocational", "religion": "Religion", "religon": "Religion",
           "religious": "Religion", "threat": "Threat", "threats": "Threat", "ethnicity": "Ethnicity",
           "ethnically": "Ethnicity", "racism": "Ethnicity"}


def norm_label(x):
    """Regroupe les variantes orthographiques en 'Bullying' / 'Not-Bullying'."""
    if pd.isna(x):
        return pd.NA
    k = re.sub(r"[^a-z]", "", str(x).lower())
    if k.startswith("not") or k == "no":
        return "Not-Bullying"
    return "Bullying" if k.startswith("bull") else pd.NA


def clean(df, text_col, label_col, drop_unlabeled):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Texte : suppression des lignes sans texte, nettoyage, dédoublonnage
    df[text_col] = df[text_col].astype("string").str.strip()
    df = df[df[text_col].notna() & (df[text_col] != "")]
    df["text_clean"] = df[text_col].map(clean_text)
    df = df[df["text_clean"].str.len() > 0]
    df = df.drop_duplicates(subset=["text_clean"], keep="first")

    # Labels : normalisation
    if label_col:
        df[label_col] = df[label_col].map(norm_label).astype("string")
        df["is_labeled"] = df[label_col].notna()
        if drop_unlabeled:
            df = df[df["is_labeled"]]

    # Colonne 'Types' : catégories normalisées, vide si Not-Bullying
    if "Types" in df.columns:
        df["Types"] = df["Types"].map(
            lambda x: CAT_MAP.get(re.sub(r"[^a-z]", "", str(x).lower()), pd.NA) if pd.notna(x) else pd.NA
        ).astype("string")
        if label_col:
            df.loc[df[label_col] == "Not-Bullying", "Types"] = pd.NA

    # Autres colonnes : strip des chaînes
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()

    df["text_length"] = df["text_clean"].str.len()
    df["word_count"] = df["text_clean"].str.split().str.len()
    return df.reset_index(drop=True)


def load_mongo(df, uri, db_name, coll_name, drop, label_col):
    from pymongo import MongoClient, ASCENDING

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    coll = client[db_name][coll_name]
    if drop:
        coll.drop()

    now = datetime.now(timezone.utc)
    records = json.loads(df.to_json(orient="records", force_ascii=False))  # NaN -> null
    for r in records:
        r["_loaded_at"] = now

    for i in range(0, len(records), 1000):
        coll.insert_many(records[i:i + 1000])

    coll.create_index([("text_clean", ASCENDING)])
    if label_col:
        coll.create_index([(label_col, ASCENDING)])

    print(f"\nMongoDB : {coll.count_documents({})} documents dans {db_name}.{coll_name}")
    if label_col:
        print("Répartition par label (agrégation MongoDB) :")
        for d in coll.aggregate([{"$group": {"_id": f"${label_col}", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}]):
            print(f"  {d['_id']}: {d['n']}")
    client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default="cyberbullying_cleaned.csv")
    ap.add_argument("--uri", default="mongodb://localhost:27017")
    ap.add_argument("--db", default="cyberbullying_db")
    ap.add_argument("--collection", default="comments")
    ap.add_argument("--text-col")
    ap.add_argument("--label-col")
    ap.add_argument("--drop-unlabeled", action="store_true")
    ap.add_argument("--drop", action="store_true", help="vider la collection avant insertion")
    ap.add_argument("--no-mongo", action="store_true")
    a = ap.parse_args()

    df = read_csv(a.csv)
    explore(df, "AVANT NETTOYAGE")

    text_col, label_col = detect_columns(df, a.text_col, a.label_col)
    print(f"\nColonne texte : {text_col!r} | colonne label : {label_col!r}")
    if label_col:
        print("\nDistribution brute des labels :\n", df[label_col].value_counts(dropna=False).to_string())

    cleaned = clean(df, text_col, label_col, a.drop_unlabeled)
    explore(cleaned, "APRÈS NETTOYAGE")
    print(f"\nLignes : {len(df)} -> {len(cleaned)}")
    if label_col:
        print("\nDistribution des labels nettoyés :\n", cleaned[label_col].value_counts(dropna=False).to_string())
    print("\nLongueur des textes :\n", cleaned["text_length"].describe().round(1).to_string())

    cleaned.to_csv(a.out, index=False, encoding="utf-8")
    print(f"\nCSV nettoyé enregistré : {a.out}")

    if not a.no_mongo:
        load_mongo(cleaned, a.uri, a.db, a.collection, a.drop, label_col)


if __name__ == "__main__":
    main()
