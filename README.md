# Détection du harcèlement social : pipeline ML + API FastAPI + Docker

Jeu de données : *Approach to Social Media Cyberbullying and Harassment Detection Using Advanced Machine Learning* (Kaggle, Md. Saiful Islam). Colonnes brutes : `Text`, `Label`, `Types`.

## Structure

```
app/
  config.py        chemins et variables d'environnement
  cleaning.py      exploration + nettoyage
  mongo_utils.py   chargement / lecture / agrégation MongoDB
  analysis.py      analyse descriptive
  modeling.py      Naive Bayes vs Régression logistique, métriques, McNemar
  visualize.py     figures (reports/figures/*.png)
  api.py           API FastAPI
run_pipeline.py    exécute les étapes 1 à 6 (pipeline complet)
cyberbullying_etl.py   script autonome : exploration, nettoyage et chargement MongoDB (étapes 1 et 2)
demo_cleaning.py   rapport de nettoyage lisible dans le terminal (pour présentation)
tests/test_api.py  test de l'API
Dockerfile · docker-compose.yml · requirements.txt
data/raw.csv       jeu de données brut (à placer ici)
ETUDE_COMPARATIVE.md   étude comparative des deux modèles
```

## Exécution locale

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
# placer le CSV dans data/raw.csv

# MongoDB local (une des deux options)
docker run -d --name mongo -p 27017:27017 mongo:7
#   ou un MongoDB déjà installé sur localhost:27017

python run_pipeline.py                       # pipeline complet avec MongoDB
python run_pipeline.py --skip-mongo          # sans MongoDB
MONGO_URI=mongomock:// python run_pipeline.py   # MongoDB simulé, sans serveur

uvicorn app.api:app --reload                 # API sur http://localhost:8000/docs
pytest -q tests                              # test de l'API
```

Le pipeline : (1) explore et nettoie, (2) charge dans MongoDB (`cyberbullying_db.comments`, index sur `label`, `category`, `text_clean` unique), (3) relit depuis MongoDB et réalise l'analyse descriptive, (4) entraîne les deux modèles avec `GridSearchCV`, (5) les compare, (6) produit les figures. Sorties : `data/cleaned.csv`, `reports/*.json`, `reports/figures/*.png`, `models/*.joblib`.

## Script autonome : `cyberbullying_etl.py` (étapes 1 et 2)

Ce script ne fait que l'exploration, le nettoyage et le chargement dans MongoDB, sans modèles ni API. Il n'a besoin que de `pandas` et `pymongo`.

```bash
python cyberbullying_etl.py --no-mongo           # exploration + nettoyage, écrit cyberbullying_cleaned.csv
python cyberbullying_etl.py --drop               # + chargement dans cyberbullying_db.comments (vide la collection avant)
python cyberbullying_etl.py --csv autre.csv --uri mongodb://localhost:27017 --db ma_base --collection ma_collection
```

Options : `--text-col` / `--label-col` (forcer les colonnes), `--drop-unlabeled` (supprimer les lignes sans label), `--out` (fichier CSV nettoyé).

Il regroupe les variantes de labels en `Bullying` / `Not-Bullying`, normalise `Types`, supprime les doublons et ajoute `text_clean`, `text_length` et `word_count`. Il conserve 5 969 lignes, contre 5 778 pour `run_pipeline.py`, qui retire en plus les textes à labels contradictoires. Pour la suite du projet (analyse, modèles, API), utilisez `run_pipeline.py`.

Pour un rapport détaillé du nettoyage (variantes de labels, entonnoir 8 452 → 5 778 lignes, exemples), lancez `python demo_cleaning.py`.

## Docker

**API seule** (modèles entraînés pendant le build, sans MongoDB) :
```bash
docker build -t cyberbullying-api .
docker run -p 8000:8000 cyberbullying-api
```

**Stack complète** (MongoDB + pipeline + API) :
```bash
docker compose up --build -d mongo api
docker compose run --rm pipeline        # charge MongoDB et régénère rapports/figures dans ./reports
```
Avec Docker Compose, `GET /comments` lit les données nettoyées dans MongoDB. Dans l'image seule, cet endpoint renvoie 503 (pas de MongoDB) ; le reste fonctionne.

## Endpoints

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état et modèles chargés |
| POST | `/predict` | `{"text": "...", "model": "logreg" \| "naive_bayes" \| "both"}` |
| POST | `/predict/batch` | `{"texts": ["..."], "model": "logreg"}` |
| GET | `/metrics` | comparaison des deux modèles |
| GET | `/stats` | analyse descriptive |
| GET | `/comments?label=&category=&limit=` | lecture MongoDB |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "you are a disgusting idiot", "model": "both"}'
```

## Déploiement

- **Variables** : `MONGO_URI`, `MONGO_DB`, `MONGO_COLLECTION`.
- **Scalabilité** : l'API est sans état (modèles en mémoire) ; lancer plusieurs réplicas derrière un reverse proxy (`docker compose up --scale api=3` avec un proxy, ou Kubernetes/Cloud Run), ou plusieurs workers : `uvicorn app.api:app --workers 4`.
- **Sécurité** : prévoir authentification, limitation de débit et HTTPS (reverse proxy) avant exposition publique ; MongoDB ne doit pas être exposé sur Internet.
- **Mise à jour des modèles** : relancer `run_pipeline.py` puis redémarrer l'API.
