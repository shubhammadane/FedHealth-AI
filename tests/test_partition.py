import pandas as pd


def test_client_file_schema(tmp_path):
    df = pd.DataFrame({"f1": [0.1, 0.2], "_target": [0, 1]})
    path = tmp_path / "hospital_01.csv"
    df.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert "_target" in loaded.columns
