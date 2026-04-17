import pandas as pd
from xgboost import XGBRegressor

def build_model():
    return XGBRegressor(
        n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1,
        random_state=42, verbosity=0,
    )

def get_feature_importance(model, feature_names):
    return pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
