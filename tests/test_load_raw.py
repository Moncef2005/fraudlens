import pandas as pd
import pytest

from fraudlens.data.load_raw import EXPECTED_COLUMNS, validate_columns


def test_validate_columns_ok():
    df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    validate_columns(df)  # ne doit pas lever d'exception


def test_validate_columns_missing():
    df = pd.DataFrame(columns=EXPECTED_COLUMNS[:-1])
    with pytest.raises(ValueError):
        validate_columns(df)
