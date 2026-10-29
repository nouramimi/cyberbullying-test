import json
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from . import config

EXTRA_STOP = {"just", "like", "im", "dont", "don", "ve", "ll", "ur", "u", "amp"}
STOP = list(ENGLISH_STOP_WORDS | EXTRA_STOP)


def top_terms(texts, n=20, ngram=(1, 1)) -> pd.Series:
    cv = CountVectorizer(stop_words=STOP, ngram_range=ngram, min_df=3, token_pattern=r"(?u)\b[a-z][a-z']+\b")
    X = cv.fit_transform(texts)
    s = pd.Series(X.sum(axis=0).A1, index=cv.get_feature_names_out())
    return s.sort_values(ascending=False).head(n)


def descriptive(df: pd.DataFrame) -> dict:
    out = {}
    out["n_comments"] = len(df)
    out["label_counts"] = df["label"].value_counts().to_dict()
    out["label_pct"] = (df["label"].value_counts(normalize=True) * 100).round(2).to_dict()
    b = df[df["label"] == "Bullying"]
    out["category_counts"] = b["category"].value_counts().to_dict()
    out["category_pct_of_bullying"] = (b["category"].value_counts(normalize=True) * 100).round(2).to_dict()
    out["length_by_label"] = {
        lab: g[["n_chars", "n_words"]].describe().round(2).to_dict() for lab, g in df.groupby("label")}
    out["mean_words_by_category"] = b.groupby("category")["n_words"].mean().round(2).to_dict()
    out["top_words_bullying"] = top_terms(b["text_clean"]).to_dict()
    out["top_words_not_bullying"] = top_terms(df[df["label"] == "Not-Bullying"]["text_clean"]).to_dict()
    out["top_bigrams_bullying"] = top_terms(b["text_clean"], 15, (2, 2)).to_dict()
    out["top_words_by_category"] = {
        c: top_terms(g["text_clean"], 10).to_dict() for c, g in b.groupby("category") if len(g) >= 30
    }
    out["share_with_exclamation"] = df.assign(e=df["text"].str.contains("!")).groupby("label")["e"].mean().round(3).to_dict()
    out["share_with_question"] = df.assign(e=df["text"].str.contains(r"\?")).groupby("label")["e"].mean().round(3).to_dict()
    out["share_uppercase_word"] = df.assign(e=df["text"].str.contains(r"\b[A-Z]{3,}\b")).groupby("label")["e"].mean().round(3).to_dict()
    out["share_with_mention"] = df.assign(e=df["text"].str.contains(r"@\w+")).groupby("label")["e"].mean().round(3).to_dict()
    return out


def save(stats: dict, name="descriptive_stats.json"):
    config.REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    (config.REPORTS_DIR / name).write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
