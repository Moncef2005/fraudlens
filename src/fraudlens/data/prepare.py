"""Nettoyage de PaySim et découpage temporel train / validation / test.

Usage :
    python -m fraudlens.data.prepare
"""
import numpy as np
import pandas as pd

from fraudlens.core.config import PROCESSED_DIR
from fraudlens.data.load_raw import PAYSIM_PARQUET

TRAIN_END_STEP = 520  # train      : step <= 520
VAL_END_STEP = 630    # validation : 520 < step <= 630 ; test : step > 630

TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"

NUMERIC_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def clean_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime doublons et valeurs invalides, trie par temps, ajoute tx_id."""
    out = df.drop_duplicates().copy()

    valid = pd.Series(True, index=out.index)
    for col in NUMERIC_COLS:
        valid &= out[col] >= 0
    out = out[valid]

    out = out.sort_values("step", kind="stable").reset_index(drop=True)
    out.insert(0, "tx_id", np.arange(len(out), dtype="int64"))
    return out


def split_temporal(
    df: pd.DataFrame,
    train_end: int = TRAIN_END_STEP,
    val_end: int = VAL_END_STEP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe selon le temps : le test est toujours dans le futur du train."""
    if train_end >= val_end:
        raise ValueError("train_end doit etre strictement inferieur a val_end.")

    train = df[df["step"] <= train_end]
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)]
    test = df[df["step"] > val_end]

    if train.empty or val.empty or test.empty:
        raise ValueError("Un des trois ensembles est vide : ajuste les bornes de step.")

    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()
    return train, val, test


def describe_split(name: str, part: pd.DataFrame) -> None:
    n_fraud = int(part["isFraud"].sum())
    print(
        f"{name:<11} lignes : {len(part):>9,} | steps {part['step'].min():>3} -> "
        f"{part['step'].max():>3} | fraudes : {n_fraud:>5,} ({n_fraud / len(part):.4%})"
    )


def main() -> None:
    if not PAYSIM_PARQUET.exists():
        raise FileNotFoundError(
            f"{PAYSIM_PARQUET} introuvable. Lance d'abord : python -m fraudlens.data.load_raw"
        )

    raw = pd.read_parquet(PAYSIM_PARQUET)
    df = clean_paysim(raw)
    print(f"Lignes avant nettoyage : {len(raw):,} | apres : {len(df):,}")

    train, val, test = split_temporal(df)

    print()
    describe_split("Train", train)
    describe_split("Validation", val)
    describe_split("Test", test)

    train.to_parquet(TRAIN_PATH, index=False)
    val.to_parquet(VAL_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    print(f"\nFichiers enregistres dans : {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
