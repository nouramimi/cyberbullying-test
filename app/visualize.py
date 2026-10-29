import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .analysis import top_terms
from .modeling import top_features

sns.set_theme(style="whitegrid")
C_BULLY, C_NOT = "#d1495b", "#2e86ab"
NAMES = {"naive_bayes": "Naive Bayes", "logreg": "Régression logistique"}


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=140)
    plt.close(fig)


def descriptive_figures(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df["label"].value_counts()
    ax.bar(vc.index, vc.values, color=[C_BULLY if i == "Bullying" else C_NOT for i in vc.index])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v}\n({v / vc.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Répartition des classes"); ax.set_ylabel("Commentaires"); ax.set_ylim(0, vc.max() * 1.15)
    _save(fig, "01_label_distribution.png")

    b = df[df["label"] == "Bullying"]
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = b["category"].value_counts()
    sns.barplot(x=vc.values, y=vc.index, color=C_BULLY, ax=ax)
    ax.set_title("Types de harcèlement (commentaires Bullying)"); ax.set_xlabel("Commentaires")
    _save(fig, "02_category_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x="label", y="n_words", hue="label", palette=[C_BULLY, C_NOT], legend=False, order=["Bullying", "Not-Bullying"],
                showfliers=False, ax=axes[0]); axes[0].set_title("Nombre de mots par classe")
    sns.boxplot(data=b, x="n_words", y="category", color=C_BULLY, showfliers=False, ax=axes[1])
    axes[1].set_title("Nombre de mots par type de harcèlement")
    _save(fig, "03_length_analysis.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (lab, col) in zip(axes, [("Bullying", C_BULLY), ("Not-Bullying", C_NOT)]):
        t = top_terms(df[df["label"] == lab]["text_clean"], 15)[::-1]
        ax.barh(t.index, t.values, color=col); ax.set_title(f"Mots les plus fréquents - {lab}")
    _save(fig, "04_top_words.png")

    cats = [c for c, g in b.groupby("category") if len(g) >= 30]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
    for ax, c in zip(axes, cats):
        t = top_terms(b[b["category"] == c]["text_clean"], 8)[::-1]
        ax.barh(t.index, t.values, color=C_BULLY); ax.set_title(c)
    for ax in axes[len(cats):]:
        ax.axis("off")
    fig.suptitle("Mots dominants par type de harcèlement")
    _save(fig, "05_top_words_by_category.png")


def model_figures(results: dict):
    names = ["naive_bayes", "logreg"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, n in zip(axes, names):
        cm = np.array(results[n]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Not-Bullying", "Bullying"], yticklabels=["Not-Bullying", "Bullying"])
        ax.set_title(NAMES[n]); ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    _save(fig, "06_confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n, c in zip(names, [C_NOT, C_BULLY]):
        r = results[n]["roc_curve"]
        ax.plot(r["fpr"], r["tpr"], color=c, label=f"{NAMES[n]} (AUC={results[n]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC"); ax.legend(loc="lower right")
    _save(fig, "07_roc_curves.png")

    m = pd.DataFrame({NAMES[n]: {k: results[n][k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                      for n in names})
    fig, ax = plt.subplots(figsize=(8, 4))
    m.plot.bar(ax=ax, color=[C_NOT, C_BULLY], rot=0)
    ax.set_ylim(max(0, m.values.min() - 0.1), 1.02); ax.set_title("Comparaison des métriques (jeu de test)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
    _save(fig, "08_metrics_comparison.png")

    rc = pd.DataFrame({NAMES[n]: results[n]["recall_by_category"] for n in names})
    fig, ax = plt.subplots(figsize=(8, 4))
    rc.plot.bar(ax=ax, color=[C_NOT, C_BULLY], rot=30); ax.set_ylabel("Rappel"); ax.set_ylim(0, 1.05)
    ax.set_title("Rappel par type de harcèlement")
    _save(fig, "09_recall_by_category.png")

    tf = top_features(results["logreg"]["_pipeline"], 15)
    fig, ax = plt.subplots(figsize=(8, 7))
    tf = tf.sort_values("coef")
    ax.barh(tf["term"], tf["coef"], color=[C_BULLY if c == "Bullying" else C_NOT for c in tf["class"]])
    ax.set_title("Facteurs les plus influents (coefficients, régression logistique)")
    _save(fig, "10_top_features_logreg.png")
    tf.to_csv(config.REPORTS_DIR / "top_features_logreg.csv", index=False)
