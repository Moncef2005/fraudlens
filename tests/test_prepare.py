import pandas as pd
import pytest

from fraudlens.data.prepare import clean_paysim, split_temporal


def make_df(steps, **overrides) -> pd.DataFrame:
    n = len(steps)
    data = {
        "step": steps,
        "type": ["TRANSFER"] * n,
        "amount": [100.0 + i for i in range(n)],
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": [1000.0] * n,
        "newbalanceOrig": [900.0] * n,
        "nameDest": [f"C{1000 + i}" for i in range(n)],
        "oldbalanceDest": [0.0] * n,
        "newbalanceDest": [100.0] * n,
        "isFraud": [0] * n,
        "isFlaggedFraud": [0] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_clean_removes_duplicates_and_negatives():
    df = make_df([1, 1, 2, 3])
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # un doublon exact
    df.loc[1, "amount"] = -5.0  # une valeur invalide

    cleaned = clean_paysim(df)

    assert len(cleaned) == 3
    assert (cleaned["amount"] >= 0).all()
    assert cleaned["tx_id"].is_unique
    assert cleaned["step"].is_monotonic_increasing


def test_split_is_strictly_temporal():
    df = clean_paysim(make_df(list(range(1, 101))))
    train, val, test = split_temporal(df, train_end=60, val_end=80)

    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_split_invalid_bounds():
    df = clean_paysim(make_df(list(range(1, 101))))
    with pytest.raises(ValueError):
        split_temporal(df, train_end=80, val_end=60)
