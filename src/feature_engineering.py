import pandas as pd

DROP_ALWAYS = [
    "encounter_id",
    "patient_nbr",
]

# Columns that can be treated as categorical even when encoded as integers.
CATEGORICAL_COLUMNS = [
    "race", "gender", "age", "weight", "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
]

def prepare_target(y: pd.DataFrame) -> pd.Series:
    mapping = {"NO": 0, ">30": 1, "<30": 1}
    result = y["readmitted"].map(mapping)
    if result.isna().any():
        bad = y.loc[result.isna(), "readmitted"].dropna().unique().tolist()
        raise ValueError(f"Unexpected target labels: {bad}")
    return result.astype(int)


def select_features(X: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers and columns documented as unsuitable for prediction."""
    X = X.copy()
    return X.drop(columns=[c for c in DROP_ALWAYS if c in X.columns], errors="ignore")


def get_categorical_columns(X: pd.DataFrame):
    return [c for c in CATEGORICAL_COLUMNS if c in X.columns]


def get_numeric_columns(X: pd.DataFrame):
    return [c for c in X.columns if c not in get_categorical_columns(X)]
