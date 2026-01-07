"""Tabular Model Training (XGBoost, LightGBM, CatBoost)"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

from config import *
from utils import *

def prepare_tabular_features(train_df, val_df=None, test_df=None):
    """Prepare features for tabular models (uses log-transformed prices)"""
    feature_cols = [col for col in train_df.columns
                   if col not in [TARGET_COL, 'price_log', ID_COL, 'image_path', 'image_exists', 'date']]

    X_train = train_df[feature_cols].copy()
    # Use log-transformed price for training
    y_train = train_df['price_log']

    # Identify categorical columns (object dtype or specific known categorical columns)
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Handle zipcode specially (keep top 20, encode rest as -1)
    if 'zipcode' in categorical_cols:
        top_zipcodes = X_train['zipcode'].value_counts().head(20).index
        X_train['zipcode'] = X_train['zipcode'].apply(lambda x: x if x in top_zipcodes else -1)

    # One-hot encode all categorical columns
    if len(categorical_cols) > 0:
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)

    # Scale numerical features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    result = {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'feature_names': X_train.columns.tolist(),
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'top_zipcodes': top_zipcodes if 'zipcode' in categorical_cols else None
    }

    if val_df is not None:
        X_val = val_df[feature_cols].copy()

        # Apply same categorical handling
        if 'zipcode' in categorical_cols and result['top_zipcodes'] is not None:
            X_val['zipcode'] = X_val['zipcode'].apply(lambda x: x if x in result['top_zipcodes'] else -1)

        if len(categorical_cols) > 0:
            X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=False)
            # Align columns with training data
            X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

        X_val_scaled = scaler.transform(X_val)
        result['X_val'] = X_val_scaled
        # Use log-transformed price for validation
        result['y_val'] = val_df['price_log']
        # Also store actual prices for evaluation
        result['y_val_actual'] = val_df[TARGET_COL]

    return result

def train_all_tabular_models(train_df, val_df):
    """Train XGBoost, LightGBM, and CatBoost (on log-transformed prices)"""
    print("\n" + "="*80)
    print("TRAINING TABULAR MODELS")
    print("="*80)
    print("\n📝 Training Strategy:")
    print("  1. Train models to predict log(price)")
    print("  2. At inference: y_pred = exp(y_pred_log)")
    print("  3. Evaluate on actual price (not log)")

    data = prepare_tabular_features(train_df, val_df)

    # XGBoost
    print("\n📊 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS, early_stopping_rounds=50)
    xgb_model.fit(data['X_train'], data['y_train'],
                  eval_set=[(data['X_val'], data['y_val'])],
                  verbose=False)
    xgb_pred_log = xgb_model.predict(data['X_val'])
    xgb_pred = np.exp(xgb_pred_log)  # Convert to actual price
    xgb_metrics = calculate_metrics(data['y_val_actual'], xgb_pred)
    print(f"XGBoost Val RMSE: ${xgb_metrics['rmse']:,.2f}, R²: {xgb_metrics['r2']:.4f}")

    # LightGBM
    print("\n📊 Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    lgb_model.fit(data['X_train'], data['y_train'],
                  eval_set=[(data['X_val'], data['y_val'])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    lgb_pred_log = lgb_model.predict(data['X_val'])
    lgb_pred = np.exp(lgb_pred_log)  # Convert to actual price
    lgb_metrics = calculate_metrics(data['y_val_actual'], lgb_pred)
    print(f"LightGBM Val RMSE: ${lgb_metrics['rmse']:,.2f}, R²: {lgb_metrics['r2']:.4f}")

    # CatBoost
    print("\n📊 Training CatBoost...")
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(data['X_train'], data['y_train'],
                  eval_set=(data['X_val'], data['y_val']))
    cat_pred_log = cat_model.predict(data['X_val'])
    cat_pred = np.exp(cat_pred_log)  # Convert to actual price
    cat_metrics = calculate_metrics(data['y_val_actual'], cat_pred)
    print(f"CatBoost Val RMSE: ${cat_metrics['rmse']:,.2f}, R²: {cat_metrics['r2']:.4f}")

    # Ensemble (average in log space, then convert)
    ensemble_pred_log = (xgb_pred_log + lgb_pred_log + cat_pred_log) / 3
    ensemble_pred = np.exp(ensemble_pred_log)
    ensemble_metrics = calculate_metrics(data['y_val_actual'], ensemble_pred)
    print(f"\n🎯 Ensemble Val RMSE: ${ensemble_metrics['rmse']:,.2f}, R²: {ensemble_metrics['r2']:.4f}")
    
    # Save models
    joblib.dump(xgb_model, MODELS_DIR / 'xgboost_model.pkl')
    joblib.dump(lgb_model, MODELS_DIR / 'lightgbm_model.pkl')
    joblib.dump(cat_model, MODELS_DIR / 'catboost_model.pkl')
    joblib.dump(data['scaler'], MODELS_DIR / 'tabular_scaler.pkl')
    
    models = {'xgboost': xgb_model, 'lightgbm': lgb_model, 'catboost': cat_model}
    predictions = {'xgboost': xgb_pred, 'lightgbm': lgb_pred, 'catboost': cat_pred, 'ensemble': ensemble_pred}
    metrics = {'xgboost': xgb_metrics, 'lightgbm': lgb_metrics, 'catboost': cat_metrics, 'ensemble': ensemble_metrics}
    
    return models, predictions, metrics

def predict_with_tabular_models(models, df, scaler, feature_names, categorical_cols=None, top_zipcodes=None):
    """Generate predictions from all tabular models (converts from log to actual prices)"""
    feature_cols = [col for col in df.columns
                   if col not in [TARGET_COL, 'price_log', ID_COL, 'image_path', 'image_exists', 'date']]
    X = df[feature_cols].copy()

    # Handle categorical columns
    if categorical_cols is not None:
        # Apply zipcode encoding if needed
        if 'zipcode' in categorical_cols and top_zipcodes is not None:
            X['zipcode'] = X['zipcode'].apply(lambda x: x if x in top_zipcodes else -1)

        # One-hot encode
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    # Align with training features
    X = X.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X)

    predictions = {}
    predictions_log = {}

    # Get predictions in log space and convert to actual prices
    for name, model in models.items():
        pred_log = model.predict(X_scaled)
        predictions_log[name] = pred_log
        predictions[name] = np.exp(pred_log)  # Convert to actual price

    # Ensemble in log space, then convert
    ensemble_log = sum(predictions_log.values()) / len(predictions_log)
    predictions['ensemble'] = np.exp(ensemble_log)

    return predictions
