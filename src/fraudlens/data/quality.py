"""Contrôle qualité approfondi du dataset PaySim.

Usage :
    python -m fraudlens.data.quality
"""
import json

import pandas as pd

from fraudlens.core.config import REPORTS_DIR
from fraudlens.data.load_raw import PAYSIM_PARQUET

NUMERIC_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TOLERANCE = 0.01


def run_quality_checks(df: pd.DataFrame) -> dict:
    """Calcule les indicateurs de qualité et renvoie un dictionnaire."""
    report: dict = {}
    n = len(df)
    n_fraud = int(df["isFraud"].sum())

    report["rows"] = n
    report["fraud_count"] = n_fraud
    report["fraud_rate"] = n_fraud / n
    report["duplicates"] = int(df.duplicated().sum())
    report["missing_values"] = int(df.isna().sum().sum())
    report["negative_values"] = {c: int((df[c] < 0).sum()) for c in NUMERIC_COLS}

    # Montants nuls
    zero_amount = df["amount"] == 0
    report["zero_amount_rows"] = int(zero_amount.sum())
    report["zero_amount_fraud"] = int((zero_amount & (df["isFraud"] == 1)).sum())

    # Destinataires marchands (nom commençant par "M") : soldes non suivis
    is_merchant = df["nameDest"].str.startswith("M").fillna(False).astype(bool)
    report["merchant_dest_rows"] = int(is_merchant.sum())
    report["merchant_dest_fraud"] = int(df.loc[is_merchant, "isFraud"].sum())

    # Cohérence des soldes (émetteur : ancien - montant = nouveau)
    err_orig = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).abs() > TOLERANCE
    # Cohérence des soldes (destinataire : ancien + montant = nouveau), hors marchands
    err_dest = (
        (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).abs() > TOLERANCE
    ) & ~is_merchant

    report["balance_error_orig_rate"] = float(err_orig.mean())
    report["balance_error_dest_rate"] = float(err_dest.mean())
    report["balance_error_orig_rate_by_class"] = {
        str(k): float(v) for k, v in err_orig.groupby(df["isFraud"]).mean().items()
    }
    report["balance_error_dest_rate_by_class"] = {
        str(k): float(v) for k, v in err_dest.groupby(df["isFraud"]).mean().items()
    }

    # Règle métier existante isFlaggedFraud
    flagged = df["isFlaggedFraud"] == 1
    report["flagged_rows"] = int(flagged.sum())
    report["flagged_and_fraud"] = int((flagged & (df["isFraud"] == 1)).sum())
    report["flag_rule_recall"] = float(report["flagged_and_fraud"] / n_fraud) if n_fraud else 0.0

    # Fraudes par type
    by_type = df.groupby("type", observed=True)["isFraud"].agg(["count", "sum"])
    report["fraud_by_type"] = {
        str(t): {"transactions": int(r["count"]), "fraudes": int(r["sum"])}
        for t, r in by_type.iterrows()
    }

    # Fraudes dans le temps
    fraud_per_step = df[df["isFraud"] == 1].groupby("step").size()
    report["fraud_per_step_min"] = int(fraud_per_step.min())
    report["fraud_per_step_max"] = int(fraud_per_step.max())
    report["steps_without_fraud"] = int(df["step"].nunique() - len(fraud_per_step))

    # Répétition des comptes (utile pour le graphe et les risques de fuite)
    orig_counts = df["nameOrig"].value_counts()
    dest_counts = df["nameDest"].value_counts()
    report["orig_accounts_repeated"] = int((orig_counts > 1).sum())
    report["dest_accounts_repeated"] = int((dest_counts > 1).sum())

    return report


def print_report(r: dict) -> None:
    print("=" * 64)
    print("RAPPORT QUALITE PAYSIM")
    print("=" * 64)
    print(f"Lignes                          : {r['rows']:,}")
    print(f"Fraudes                         : {r['fraud_count']:,} ({r['fraud_rate']:.4%})")
    print(f"Doublons exacts                 : {r['duplicates']:,}")
    print(f"Valeurs manquantes              : {r['missing_values']:,}")
    print(f"Valeurs negatives               : {r['negative_values']}")
    print()
    print(f"Montants nuls                   : {r['zero_amount_rows']:,}")
    print(f"  dont fraudes                  : {r['zero_amount_fraud']:,}")
    print(f"Destinataires marchands (M...)  : {r['merchant_dest_rows']:,}")
    print(f"  dont fraudes                  : {r['merchant_dest_fraud']:,}")
    print()
    print("Incoherences de soldes (taux de lignes) :")
    print(f"  emetteur, global              : {r['balance_error_orig_rate']:.2%}")
    print(f"  emetteur, par classe          : {r['balance_error_orig_rate_by_class']}")
    print(f"  destinataire, global          : {r['balance_error_dest_rate']:.2%}")
    print(f"  destinataire, par classe      : {r['balance_error_dest_rate_by_class']}")
    print()
    print(f"Regle isFlaggedFraud : {r['flagged_rows']} lignes signalees, "
          f"{r['flagged_and_fraud']} vraies fraudes, rappel = {r['flag_rule_recall']:.4%}")
    print()
    print("Fraudes par type :")
    for t, v in r["fraud_by_type"].items():
        print(f"  {t:<9} {v['transactions']:>10,} transactions | {v['fraudes']:>6,} fraudes")
    print()
    print(f"Fraudes par step : min {r['fraud_per_step_min']}, max {r['fraud_per_step_max']}, "
          f"steps sans fraude : {r['steps_without_fraud']}")
    print(f"Comptes emetteurs apparaissant plusieurs fois     : {r['orig_accounts_repeated']:,}")
    print(f"Comptes destinataires apparaissant plusieurs fois : {r['dest_accounts_repeated']:,}")
    print("=" * 64)


def main() -> None:
    if not PAYSIM_PARQUET.exists():
        raise FileNotFoundError(
            f"{PAYSIM_PARQUET} introuvable. Lance d'abord : python -m fraudlens.data.load_raw"
        )
    df = pd.read_parquet(PAYSIM_PARQUET)
    report = run_quality_checks(df)
    print_report(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport JSON enregistre : {out}")


if __name__ == "__main__":
    main()
