"""
Mobile Phone Recommendation Engine
Content-Based Filtering using Cosine Similarity
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"


def load_models():
    """Load all saved joblib artifacts."""
    df_raw          = joblib.load(MODEL_DIR / "phones_df.pkl")
    df_norm         = joblib.load(MODEL_DIR / "phones_df_normalized.pkl")
    feature_cols    = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return df_raw, df_norm, feature_cols


# Load once at import time
try:
    DF_RAW, DF_NORM, FEATURE_COLS = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    _LOAD_ERROR = str(e)


def recommend(
    budget_min: float,
    budget_max: float,
    os_pref: str,
    camera_w: float,
    battery_w: float,
    ram_w: float,
    performance_w: float,
    storage_w: float,
    selfie_w: float,
    brand_pref: str = "Any",
    top_n: int = 5,
) -> list[dict]:
    """
    Returns top_n phone recommendations as a list of dicts.

    Parameters
    ----------
    budget_min, budget_max : price range in USD
    os_pref  : 'iOS' | 'Android' | 'Any'
    *_w      : preference weights 0.0 – 1.0
    brand_pref : brand string or 'Any'
    top_n    : number of results to return
    """
    if not MODELS_LOADED:
        raise RuntimeError(f"Models not loaded. Run the notebook first.\n{_LOAD_ERROR}")

    # --- Build user preference vector ----------------------------------------
    # Order must match FEATURE_COLS:
    # ['internal memory', 'RAM', 'performance', 'main camera',
    #  'selfie camera', 'battery size', 'screen size']
    user_vec = np.array([
        storage_w,
        ram_w,
        performance_w,
        camera_w,
        selfie_w,
        battery_w,
        0.5,            # screen size — neutral
    ], dtype=float).reshape(1, -1)

    # --- Filter by budget (use RAW prices) -----------------------------------
    mask = DF_RAW["price"].between(float(budget_min), float(budget_max))
    if os_pref != "Any":
        mask &= DF_RAW["operating system"] == os_pref
    if brand_pref and brand_pref != "Any":
        brand_mask = DF_RAW["brand"] == brand_pref
    else:
        brand_mask = pd.Series(True, index=DF_RAW.index)

    candidates_idx = DF_RAW.index[mask].tolist()

    if not candidates_idx:
        return []

    # --- Cosine similarity between user vec and candidate feature vectors -----
    cand_features = DF_NORM.loc[candidates_idx, FEATURE_COLS].values
    scores = cosine_similarity(user_vec, cand_features)[0]

    # Brand bonus
    bonus = brand_mask.loc[candidates_idx].values.astype(float) * 0.05
    scores = np.clip(scores + bonus, 0, 1)

    # --- Rank & pick top_n ---------------------------------------------------
    ranked_idx = np.argsort(scores)[::-1][:top_n]
    selected_idx = [candidates_idx[i] for i in ranked_idx]
    selected_scores = scores[ranked_idx]

    results = []
    for idx, score in zip(selected_idx, selected_scores):
        row = DF_RAW.loc[idx]
        results.append({
            "rank":             len(results) + 1,
            "cellphone_id":     int(row["cellphone_id"]),
            "brand":            str(row["brand"]),
            "model":            str(row["model"]),
            "os":               str(row["operating system"]),
            "storage":          int(row["internal memory"]),
            "ram":              int(row["RAM"]),
            "performance":      int(row["performance"]),
            "main_camera":      int(row["main camera"]),
            "selfie_camera":    int(row["selfie camera"]),
            "battery":          int(row["battery size"]),
            "screen_size":      float(row["screen size"]),
            "weight":           float(row["weight"]),
            "price":            int(row["price"]),
            "release_date":     str(row["release date"]),
            "match_score":      round(float(score) * 100, 1),
        })

    return results


def get_brands() -> list[str]:
    """All unique brands from the dataset."""
    if not MODELS_LOADED:
        return []
    return sorted(DF_RAW["brand"].unique().tolist())


def get_price_range() -> tuple[int, int]:
    """Min and max prices in the dataset."""
    if not MODELS_LOADED:
        return (0, 2000)
    return int(DF_RAW["price"].min()), int(DF_RAW["price"].max())


def get_dataset_stats() -> dict:
    """Quick stats for the dashboard."""
    if not MODELS_LOADED:
        return {}
    return {
        "total_phones":   len(DF_RAW),
        "total_brands":   DF_RAW["brand"].nunique(),
        "price_min":      int(DF_RAW["price"].min()),
        "price_max":      int(DF_RAW["price"].max()),
        "avg_price":      int(DF_RAW["price"].mean()),
        "os_counts":      DF_RAW["operating system"].value_counts().to_dict(),
        "brand_counts":   DF_RAW["brand"].value_counts().to_dict(),
    }
