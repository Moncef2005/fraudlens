"""Chargement, validation et conversion de PaySim en Parquet.

Usage :
    python -m fraudlens.data.load_raw
"""
from pathlib import Path

import pandas as pd

from fraudlens.core.config import PROCESSED_DIR, RAW_DIR

PAYSIM_CSV = RAW_DIR / "paysim" / "PS_20174392719_1491204439457_Log.csv"
PAYSIM_PARQUET = PROCESSED_DIR / "paysim_raw.parquet"

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Vérifie que toutes les colonnes attendues sont présentes."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")


def load_paysim(path: Path = PAYSIM_CSV) -> pd.DataFrame:
    """Charge le CSV PaySim avec des types mémoire-efficaces."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Télécharge PaySim sur Kaggle et place le CSV dans data/raw/paysim/."
        )

    dtypes = {
        "step": "int16",
        "type": "category",
        "amount": "float64",
        "nameOrig": "string",
        "oldbalanceOrg": "float64",
        "newbalanceOrig": "float64",
        "nameDest": "string",
        "oldbalanceDest": "float64",
        "newbalanceDest": "float64",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    df = pd.read_csv(path, dtype=dtypes)
    validate_columns(df)
    return df


def summarize(df: pd.DataFrame) -> None:
    """Affiche un résumé de contrôle qualité."""
    n = len(df)
    n_fraud = int(df["isFraud"].sum())

    print("=" * 60)
    print("RÉSUMÉ DU DATASET PAYSIM")
    print("=" * 60)
    print(f"Lignes                : {n:,}")
    print(f"Colonnes              : {df.shape[1]}")
    print(f"Mémoire               : {df.memory_usage(deep=True).sum() / 1e6:.0f} Mo")
    print(f"Période (step)        : {df['step'].min()} -> {df['step'].max()} (1 step = 1 heure)")
    print(f"Doublons exacts       : {df.duplicated().sum():,}")
    print(f"Valeurs manquantes    : {int(df.isna().sum().sum()):,}")
    print(f"Comptes émetteurs     : {df['nameOrig'].nunique():,}")
    print(f"Comptes destinataires : {df['nameDest'].nunique():,}")
    print()
    print(f"Fraudes               : {n_fraud:,} ({n_fraud / n:.4%})")
    print(f"Transactions saines   : {n - n_fraud:,}")
    print()
    print("Répartition par type de transaction :")
    by_type = df.groupby("type", observed=True)["isFraud"].agg(["count", "sum"])
    by_type.columns = ["transactions", "fraudes"]
    by_type["taux_fraude"] = (by_type["fraudes"] / by_type["transactions"]).map("{:.4%}".format)
    print(by_type.to_string())
    print("=" * 60)


def main() -> None:
    df = load_paysim()
    summarize(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PAYSIM_PARQUET, index=False)
    print(f"\nFichier Parquet enregistré : {PAYSIM_PARQUET}")


if __name__ == "__main__":
    main()
