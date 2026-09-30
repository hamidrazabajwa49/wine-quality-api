import pandas as pd

from wine_api.data import FEATURES, TARGET, load_data


def test_load_data_renames_and_dedups(tmp_path):
    cols = [f.replace("_", " ") for f in FEATURES] + [TARGET]
    row = [1.0] * len(FEATURES) + [5]
    other = [2.0] * len(FEATURES) + [6]
    path = tmp_path / "wine.csv"
    pd.DataFrame([row, row, other], columns=cols).to_csv(path, sep=";", index=False)

    df = load_data(str(path))

    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == 2  # duplicate row is gone
