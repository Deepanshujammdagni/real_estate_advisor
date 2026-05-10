"""
Real Estate Investment Advisor
Step 1: Data Preprocessing
- Handle missing values and duplicates
- Normalize/scale numerical features
- Encode categorical features
- Create new features
- Create binary label "Good Investment"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str = "data/india_housing_prices.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. BASIC CLEANING
# ─────────────────────────────────────────────
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    original_rows = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"🗑️  Removed {original_rows - len(df)} duplicate rows")

    # Drop rows where essential columns are null
    essential_cols = ["Price_in_Lakhs", "Size_in_SqFt", "BHK"]
    df = df.dropna(subset=essential_cols)
    print(f"🗑️  Dropped rows with null essentials → {len(df)} rows remaining")

    # Reset index
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Numeric → median
    num_imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Categorical → most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    print(f"✅ Imputed missing values. Remaining nulls: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    current_year = 2024

    # Age of property (if not already present)
    if "Year_Built" in df.columns and "Age_of_Property" not in df.columns:
        df["Age_of_Property"] = current_year - df["Year_Built"]

    # Price per SqFt (recompute for consistency)
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]

    # School density score (normalised)
    if "Nearby_Schools" in df.columns:
        df["School_Density_Score"] = df["Nearby_Schools"] / (df["Nearby_Schools"].max() + 1e-5)

    # Hospital proximity score
    if "Nearby_Hospitals" in df.columns:
        df["Hospital_Proximity_Score"] = df["Nearby_Hospitals"] / (df["Nearby_Hospitals"].max() + 1e-5)

    # Appreciation rate proxy: inverse of age × transport accessibility
    # (newer + better transport → higher appreciation potential)
    if "Age_of_Property" in df.columns and "Public_Transport_Accessibility" in df.columns:
        transport_map = {"Low": 1, "Medium": 2, "High": 3}
        df["Transport_Score"] = df["Public_Transport_Accessibility"].map(transport_map).fillna(1)
        df["Appreciation_Rate"] = df["Transport_Score"] / (df["Age_of_Property"].clip(lower=1))
    else:
        df["Appreciation_Rate"] = 1 / (df["Age_of_Property"].clip(lower=1))

    # Total amenity score
    if "Amenities" in df.columns:
        # Assume comma-separated list
        df["Amenity_Count"] = df["Amenities"].apply(
            lambda x: len(str(x).split(",")) if pd.notnull(x) else 0
        )

    # Price category (for analysis)
    df["Price_Category"] = pd.qcut(df["Price_in_Lakhs"], q=4, labels=["Low", "Medium", "High", "Premium"])

    print("✅ Feature engineering complete. New features added:")
    new_features = ["Price_per_SqFt", "School_Density_Score", "Hospital_Proximity_Score",
                    "Appreciation_Rate", "Amenity_Count", "Price_Category", "Age_of_Property"]
    print("  →", [f for f in new_features if f in df.columns])
    return df


# ─────────────────────────────────────────────
# 5. CREATE TARGET LABELS
# ─────────────────────────────────────────────
def create_targets(df: pd.DataFrame,
                   appreciation_threshold: float = None) -> pd.DataFrame:
    """
    Classification target: Good_Investment (1/0)
      Rule: appreciation_rate > median threshold
            AND price_per_sqft < 75th percentile (value for money)
            AND transport_score >= 2 (decent connectivity)

    Regression target: Price_After_5_Years (in Lakhs)
      Simple compound growth: Price × (1 + annual_rate)^5
    """
    if appreciation_threshold is None:
        appreciation_threshold = df["Appreciation_Rate"].median()

    price_75th = df["Price_per_SqFt"].quantile(0.75)

    transport_ok = df.get("Transport_Score", pd.Series(2, index=df.index)) >= 2

    df["Good_Investment"] = (
        (df["Appreciation_Rate"] > appreciation_threshold) &
        (df["Price_per_SqFt"] < price_75th) &
        transport_ok
    ).astype(int)

    # Annual appreciation rate for price forecast (3–8%)
    annual_rate = (df["Appreciation_Rate"] / df["Appreciation_Rate"].max()) * 0.05 + 0.03
    df["Price_After_5_Years"] = df["Price_in_Lakhs"] * ((1 + annual_rate) ** 5)

    print(f"✅ Targets created:")
    print(f"   Good_Investment → {df['Good_Investment'].value_counts().to_dict()}")
    print(f"   Price_After_5_Years → mean ₹{df['Price_After_5_Years'].mean():.1f}L")
    return df


# ─────────────────────────────────────────────
# 6. ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cat_cols = [
        "State", "City", "Locality", "Property_Type",
        "Furnished_Status", "Security", "Facing",
        "Owner_Type", "Availability_Status", "Amenities",
        "Public_Transport_Accessibility"
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print(f"✅ Encoded {len(cat_cols)} categorical columns")
    return df, encoders


# ─────────────────────────────────────────────
# 7. SCALE NUMERICAL FEATURES
# ─────────────────────────────────────────────
def scale_features(df: pd.DataFrame,
                   model_features: list) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    existing = [f for f in model_features if f in df.columns]
    df[existing] = scaler.fit_transform(df[existing])
    print(f"✅ Scaled {len(existing)} numerical features")
    return df, scaler


# ─────────────────────────────────────────────
# 8. OUTLIER HANDLING (IQR-based capping)
# ─────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame,
                    cols: list = None) -> pd.DataFrame:
    if cols is None:
        cols = ["Price_in_Lakhs", "Price_per_SqFt", "Size_in_SqFt"]
    cols = [c for c in cols if c in df.columns]

    for col in cols:
        q1 = df[col].quantile(0.01)
        q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q3)

    print(f"✅ Outliers capped (1st–99th percentile) for: {cols}")
    return df


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def preprocess_pipeline(input_path: str = "data/india_housing_prices.csv",
                        output_path: str = "data/processed_housing.csv"):
    print("=" * 60)
    print("  REAL ESTATE INVESTMENT ADVISOR — DATA PREPROCESSING")
    print("=" * 60)

    df = load_data(input_path)
    df = basic_cleaning(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = feature_engineering(df)
    df = create_targets(df)
    df, encoders = encode_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Processed data saved → {output_path}")
    print(f"   Final shape: {df.shape}")
    return df, encoders


if __name__ == "__main__":
    df, encoders = preprocess_pipeline()
    print("\nColumn overview:")
    print(df.dtypes.tail(20))
