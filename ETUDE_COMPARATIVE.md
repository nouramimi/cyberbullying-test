# Étude comparative : Naive Bayes vs Régression logistique

## 1. Données et protocole

- **Données brutes** : 8 452 commentaires, 3 colonnes. Les `Label` existent en ~20 variantes orthographiques (« Not- Bullying », « Bullyinfg », « Not-Bulllying »…), `Types` en 25 variantes, 3 627 valeurs nulles dans `Types` (cas Not-Bullying), 2 labels manquants.
- **Nettoyage** : labels et catégories normalisés (2 classes, 7 catégories de harcèlement + « Unspecified »), 2 lignes sans label supprimées, 304 lignes à labels contradictoires pour un même texte supprimées, 2 368 doublons supprimés. **Résultat : 5 778 commentaires** (3 160 Bullying = 54,7 % ; 2 618 Not-Bullying = 45,3 %).
- **Découpage** : 80/20 stratifié (4 622 entraînement / 1 156 test), graine 42. Le dédoublonnage évite qu'un même texte se retrouve dans les deux ensembles.
- **Représentation** : TF-IDF uni+bigrammes (`min_df=2`, `sublinear_tf`). La colonne `Types` n'est **pas** utilisée comme variable (fuite de label).
- **Réglage** : `GridSearchCV`, validation croisée stratifiée à 5 plis, score F1 (classe Bullying).
- **Modèles** : (A) Naive Bayes multinomial, famille générative, alpha ∈ {0,05 ; 0,1 ; 0,3 ; 1} ; (B) régression logistique, famille discriminative, C ∈ {0,5 ; 1 ; 5 ; 10 ; 30}, `class_weight="balanced"`.

## 2. Résultats (jeu de test, n = 1 156)

| Métrique | Naive Bayes (α=0,3) | Régression logistique (C=5) |
|---|---|---|
| Accuracy | 0,911 | **0,914** |
| Précision (Bullying) | 0,909 | **0,915** |
| Rappel (Bullying) | 0,930 | 0,930 |
| F1 (Bullying) | 0,919 | **0,922** |
| F1 macro | 0,910 | **0,913** |
| ROC-AUC | **0,969** | 0,967 |
| Précision moyenne (PR-AUC) | **0,974** | 0,969 |
| F1 en validation croisée | **0,905 ± 0,004** | 0,899 ± 0,003 |
| Matrice de confusion [[TN, FP],[FN, TP]] | [[465, 59],[44, 588]] | [[469, 55],[44, 588]] |

**Test de McNemar** : NB seul correct sur 23 exemples, régression logistique seule correcte sur 27 ; χ² = 0,18, **p = 0,67**. Il n'y a donc **aucune différence statistiquement significative** entre les deux modèles.

**Rappel par type de harcèlement (test)** :

| Catégorie | NB | LogReg |
|---|---|---|
| Ethnicity | 1,00 | 0,96 |
| Political | 0,98 | 0,99 |
| Religion | 1,00 | 1,00 |
| Threat | 0,94 | 0,97 |
| Sexual | 0,92 | 0,91 |
| Vocational | 0,91 | 0,88 |
| Troll | 0,87 | 0,89 |

Les catégories « Troll », « Vocational » et « Sexual » sont les plus difficiles ; les catégories à vocabulaire distinctif (religion, politique, ethnicité) sont presque toujours détectées. Les effectifs de test par catégorie sont petits (Ethnicity, Threat, Unspecified surtout) : ces rappels sont indicatifs.

## 3. Discussion

- **Performance** : les deux modèles sont équivalents (écart de 0,3 point d'accuracy, non significatif). La régression logistique a un léger avantage en précision et en F1 sur le test ; Naive Bayes est meilleur en validation croisée, en ROC-AUC et en précision moyenne. Aucun ne domine nettement.
- **Types d'erreurs** : même nombre de faux négatifs (44). La régression logistique commet 4 faux positifs de moins (55 contre 59).
- **Interprétabilité** : la régression logistique fournit des coefficients directement lisibles (figure `10_top_features_logreg.png`, `reports/top_features_logreg.csv`). Les termes qui poussent vers *Bullying* sont `comedy`, `religion`, `humor`, `religious`, `political`, `fuck`, `sexual`, `sexy` ; vers *Not-Bullying* : `beauty`, `love`, `beautiful`, `inspiration`, `positivity`, `content`.
- **Coût** : les deux entraînements prennent quelques secondes et la prédiction est de l'ordre de 0,02 ms par message ; Naive Bayes reste plus simple et plus rapide à entraîner sans réglage lourd.
- **Recommandation** : régression logistique par défaut dans l'API (probabilités plus exploitables avec `class_weight`, coefficients explicables) ; Naive Bayes comme référence rapide. Le choix est une préférence d'usage, pas une supériorité démontrée.

## 4. Facteurs influençant le harcèlement (analyse descriptive)

- Parmi les commentaires Bullying : Troll 25,6 %, Sexual 21,0 %, Vocational 14,8 %, Political 14,7 %, Religion 12,8 %, Threat 6,3 %, Ethnicity 4,0 %.
- Les messages de harcèlement contiennent plus de points d'interrogation (3,9 % contre 0,9 %) et de mots en majuscules (2,1 % contre 0,3 %) et un peu plus de points d'exclamation (16,2 % contre 11,9 %). La longueur est comparable (9,4 contre 9,7 mots en moyenne) ; les messages Sexual et Threat sont les plus courts (~7 mots).
- Le vocabulaire dominant par catégorie est très marqué (voir `05_top_words_by_category.png`) : « political / government / leaders », « religion / religious », « sexual / sex », « kill », insultes raciales.

## 5. Limites

- **Jeu de données peu naturel** : de nombreux commentaires Not-Bullying ressemblent à des compliments génériques (« Your positivity is like a beacon… ») et beaucoup de commentaires Bullying suivent des gabarits thématiques (« political leaders undermines… »). Les modèles apprennent en partie ces gabarits, ce qui gonfle probablement les scores (~91 %) par rapport à des commentaires réels de réseaux sociaux. Les résultats ne se transposent pas tels quels à des données réelles.
- **Qualité des annotations** : 304 lignes contradictoires et des labels très bruités à l'origine ; certains exemples étiquetés Bullying sont ambigus (ex. propos sexuels non agressifs).
- **Évaluation** : un seul découpage train/test ; les écarts entre modèles sont dans le bruit statistique.
- **Pistes** : validation sur un autre jeu de données, modèles pré-entraînés (BERT/RoBERTa), classification multi-classes des catégories, seuil de décision ajusté selon le coût des faux négatifs.
