import re

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = str(s)
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = SPACE_RE.sub(" ", s).strip().strip('"').strip().lower()
    return s


def normalize_label(x):
    if pd.isna(x):
        return np.nan
    k = re.sub(r"[^a-z]", "", str(x).lower())
    if k.startswith("not") or k == "no":
        return "Not-Bullying"
    if k.startswith("bull"):
        return "Bullying"
    return np.nan


CATEGORY_MAP = {
    "sexual": "Sexual", "saxual": "Sexual",
    "troll": "Troll",
    "political": "Political",
    "vocational": "Vocational", "vocation": "Vocational",
    "religion": "Religion", "religon": "Religion", "religious": "Religion",
    "threat": "Threat", "threats": "Threat",
    "ethnicity": "Ethnicity", "ethnically": "Ethnicity", "racism": "Ethnicity",
}


def normalize_category(x):
    if pd.isna(x):
        return np.nan
    return CATEGORY_MAP.get(re.sub(r"[^a-z]", "", str(x).lower()), np.nan)


def explore(df: pd.DataFrame) -> dict:
    return {
        "shape": list(df.shape),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": df.isna().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "duplicated_texts": int(df["Text"].duplicated().sum()),
        "n_raw_label_variants": int(df["Label"].nunique()),
        "n_raw_type_variants": int(df["Types"].nunique()),
        "raw_label_counts": df["Label"].value_counts(dropna=False).to_dict(),
        "raw_type_counts": df["Types"].value_counts(dropna=False).to_dict(),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"rows_raw": len(df)}
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]

    d["Text"] = d["Text"].astype("string").str.strip()
    d = d[d["Text"].notna() & (d["Text"] != "")]
    log["after_empty_text"] = len(d)

    d["label"] = d["Label"].map(normalize_label)
    log["dropped_missing_label"] = int(d["label"].isna().sum())
    d = d[d["label"].notna()]

    d["category"] = d["Types"].map(normalize_category)
    d.loc[d["label"] == "Not-Bullying", "category"] = np.nan  # une catégorie n'a de sens que pour Bullying
    d["category"] = d["category"].fillna("None")
    d.loc[(d["label"] == "Bullying") & (d["category"] == "None"), "category"] = "Unspecified"

    d["text_clean"] = d["Text"].map(clean_text)
    d = d[d["text_clean"].str.len() > 0]

    # Textes présents avec des labels contradictoires : retirés (bruit d'annotation)
    nlab = d.groupby("text_clean")["label"].nunique()
    conflicts = nlab[nlab > 1].index
    log["dropped_conflicting_label_rows"] = int(d["text_clean"].isin(conflicts).sum())
    d = d[~d["text_clean"].isin(conflicts)]

    before = len(d)
    d = d.drop_duplicates(subset="text_clean", keep="first")
    log["dropped_duplicates"] = before - len(d)

    d["label_binary"] = (d["label"] == "Bullying").astype(int)
    d["n_chars"] = d["text_clean"].str.len()
    d["n_words"] = d["text_clean"].str.split().str.len()
    d = d.rename(columns={"Text": "text"})[
        ["text", "text_clean", "label", "label_binary", "category", "n_chars", "n_words"]
    ].reset_index(drop=True)
    log["rows_clean"] = len(d)
    return d, log
