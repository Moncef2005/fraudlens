"""Analyse exploratoire (EDA) de PaySim, sur l'ensemble d'entrainement.

Usage :
    python -m fraudlens.data.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudlens.core.config import FIGURES_DIR, RANDOM_STATE  # noqa: E402
from fraudlens.data.prepare import (  # noqa: E402
    TEST_PATH,
    TRAIN_END_STEP,
    TRAIN_PATH,
    VAL_END_STEP,
    VAL_PATH,
)

RISK_TYPES = ["TRANSFER", "CASH_OUT"]
COLORS = {0: "#4C78A8", 1: "#E45756"}
LABELS = {0: "Normale", 1: "Fraude"}
COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure enregistree : {path.name}")


def plot_class_balance(train: pd.DataFrame) -> None:
    counts = train["isFraud"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Normale", "Fraude"], counts.values, color=[COLORS[0], COLORS[1]])
    ax.set_yscale("log")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("Transactions (echelle log)")
    ax.set_title("Desequilibre des classes (train)")
    save(fig, "01_class_balance.png")


def plot_drift(all_steps: pd.DataFrame) -> None:
    g = all_steps.groupby("step")["isFraud"].agg(total="count", fraud="sum")
    g = g.reindex(range(1, int(g.index.max()) + 1), fill_value=0)
    g["normal"] = (g["total"] - g["fraud"]).clip(lower=1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(g.index, g["normal"], color=COLORS[0])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Normales / step (log)")
    axes[0].set_title("Derive temporelle : volume normal vs fraudes par step")
    axes[1].plot(g.index, g["fraud"], color=COLORS[1])
    axes[1].set_ylabel("Fraudes / step")
    axes[1].set_xlabel("step (1 step = 1 heure)")
    for ax in axes:
        ax.axvline(TRAIN_END_STEP, ls="--", color="grey")
        ax.axvline(VAL_END_STEP, ls="--", color="grey")
    save(fig, "02_drift_per_step.png")

    print("\nMoyennes par step et par ensemble :")
    bounds = [
        ("Train", 1, TRAIN_END_STEP),
        ("Validation", TRAIN_END_STEP + 1, VAL_END_STEP),
        ("Test", VAL_END_STEP + 1, int(g.index.max())),
    ]
    for name, lo, hi in bounds:
        part = g.loc[lo:hi]
        print(
            f"  {name:<11} normales/step : {(part['total'] - part['fraud']).mean():>9,.0f} | "
            f"fraudes/step : {part['fraud'].mean():>5.1f}"
        )


def plot_types(train: pd.DataFrame) -> None:
    by_type = train.groupby("type", observed=True)["isFraud"].agg(["count", "sum"])
    by_type["rate"] = by_type["sum"] / by_type["count"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(by_type.index.astype(str), by_type["count"], color=COLORS[0])
    axes[0].set_yscale("log")
    axes[0].set_title("Transactions par type (log)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(by_type.index.astype(str), by_type["rate"] * 100, color=COLORS[1])
    axes[1].set_title("Taux de fraude par type (%)")
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, "03_types.png")


def plot_amounts(risk: pd.DataFrame) -> None:
    bins = np.linspace(0, np.log10(risk["amount"].max() + 1) + 0.1, 60)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cls in (0, 1):
        vals = np.log10(risk.loc[risk["isFraud"] == cls, "amount"] + 1)
        ax.hist(vals, bins=bins, density=True, alpha=0.6, color=COLORS[cls], label=LABELS[cls])
    ax.set_xlabel("log10(montant + 1)")
    ax.set_ylabel("Densite")
    ax.set_title("Distribution des montants (TRANSFER et CASH_OUT, train)")
    ax.legend()
    save(fig, "04_amounts.png")

    print("\nMontants par type et par classe (train) :")
    stats = risk.groupby(["type", "isFraud"], observed=True)["amount"].agg(
        ["count", "median", "mean", "max"]
    )
    print(stats.to_string(float_format=lambda x: f"{x:,.0f}"))


def plot_hours(train: pd.DataFrame) -> None:
    hour = train["step"] % 24
    share = pd.crosstab(hour, train["isFraud"], normalize="columns")
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.4
    x = np.arange(24)
    ax.bar(x - width / 2, share.get(0, 0) * 100, width, color=COLORS[0], label=LABELS[0])
    ax.bar(x + width / 2, share.get(1, 0) * 100, width, color=COLORS[1], label=LABELS[1])
    ax.set_xticks(x)
    ax.set_xlabel("Heure de la journee (step modulo 24)")
    ax.set_ylabel("Part des transactions de la classe (%)")
    ax.set_title("Repartition horaire : normales vs fraudes (train)")
    ax.legend()
    save(fig, "05_hours.png")


def plot_balance_behavior(risk: pd.DataFrame) -> None:
    flags = pd.DataFrame(
        {
            "Compte emetteur vide": (risk["newbalanceOrig"] == 0) & (risk["oldbalanceOrg"] > 0),
            "Solde dest. nul avant": risk["oldbalanceDest"] == 0,
            "Solde dest. inchange apres": risk["oldbalanceDest"] == risk["newbalanceDest"],
        }
    )
    rates = flags.groupby(risk["isFraud"]).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(rates.shape[1])
    width = 0.4
    ax.bar(x - width / 2, rates.loc[0], width, color=COLORS[0], label=LABELS[0])
    ax.bar(x + width / 2, rates.loc[1], width, color=COLORS[1], label=LABELS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(rates.columns, rotation=10)
    ax.set_ylabel("% des transactions de la classe")
    ax.set_title("Comportement des soldes (TRANSFER et CASH_OUT, train)")
    ax.legend()
    save(fig, "06_balance_behavior.png")

    print("\nComportement des soldes (% par classe) :")
    print(rates.round(2).to_string())


def plot_correlation(risk: pd.DataFrame) -> None:
    cols = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]
    sample = risk.sample(min(500_000, len(risk)), random_state=RANDOM_STATE)
    corr = sample[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlations (TRANSFER et CASH_OUT, echantillon)")
    save(fig, "07_correlation.png")


def main() -> None:
    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} introuvable. Lance d'abord : python -m fraudlens.data.prepare"
            )

    train = pd.read_parquet(TRAIN_PATH, columns=COLUMNS)
    val_steps = pd.read_parquet(VAL_PATH, columns=["step", "isFraud"])
    test_steps = pd.read_parquet(TEST_PATH, columns=["step", "isFraud"])
    all_steps = pd.concat([train[["step", "isFraud"]], val_steps, test_steps], ignore_index=True)

    risk = train[train["type"].isin(RISK_TYPES)]
    print(f"Train : {len(train):,} lignes | TRANSFER + CASH_OUT : {len(risk):,} lignes")
    print(f"Taux de fraude dans TRANSFER + CASH_OUT (train) : {risk['isFraud'].mean():.4%}\n")

    print("Generation des figures :")
    plot_class_balance(train)
    plot_drift(all_steps)
    plot_types(train)
    plot_amounts(risk)
    plot_hours(train)
    plot_balance_behavior(risk)
    plot_correlation(risk)
    print(f"\nTermine. Figures dans : {FIGURES_DIR}")


if __name__ == "__main__":
    main()
