import pandas as pd
from src.feature_engineering import prepare_target, select_features


def test_target_mapping():
    y = pd.DataFrame({"readmitted": ["NO", ">30", "<30"]})
    assert prepare_target(y).tolist() == [0, 1, 1]


def test_identifier_removal():
    x = pd.DataFrame({"encounter_id": [1], "patient_nbr": [2], "age": ["[50-60)"]})
    out = select_features(x)
    assert "encounter_id" not in out
    assert "patient_nbr" not in out
