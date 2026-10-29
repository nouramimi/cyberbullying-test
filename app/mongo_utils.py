from datetime import datetime, timezone

import pandas as pd

from . import config


def get_collection(uri: str | None = None):
    uri = uri or config.MONGO_URI
    if uri.startswith("mongomock"):
        import mongomock
        client = mongomock.MongoClient()
    else:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    return client[config.MONGO_DB][config.MONGO_COLLECTION]


def load_dataframe(df: pd.DataFrame, coll, drop: bool = True) -> int:
    if drop:
        coll.drop()
    now = datetime.now(timezone.utc)
    docs = df.to_dict("records")
    for d in docs:
        d["loaded_at"] = now
    for i in range(0, len(docs), 1000):
        coll.insert_many(docs[i:i + 1000])
    coll.create_index("label")
    coll.create_index("category")
    coll.create_index("text_clean", unique=True)
    return coll.count_documents({})


def read_dataframe(coll) -> pd.DataFrame:
    return pd.DataFrame(list(coll.find({}, {"_id": 0, "loaded_at": 0})))


def aggregate_counts(coll, field: str) -> dict:
    return {d["_id"]: d["n"] for d in coll.aggregate(
        [{"$group": {"_id": f"${field}", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}])}
