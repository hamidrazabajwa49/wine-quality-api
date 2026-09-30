"""
Data loading and cleaning
"""

import pandas as pd

TARGET = "quality"
FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip().str.replace(" ","_")
    df_cleaned = df.drop_duplicates().dropna(subset=FEATURES + [TARGET])
    
    return df_cleaned
