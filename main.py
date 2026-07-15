# ============================================
# FINAL PROJECT - MINI MACHINE LEARNING
# Prediksi Harga Mobil Bekas (Regresi)
# FILE UTAMA (main.py)
# ============================================

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import fungsi dari utils.py
from utils import (
    load_data,
    handle_outlier,
    feature_engineering,
    preprocess_data,
    evaluate_model,
    save_statistics_to_txt,
    create_visualizations
)

warnings.filterwarnings('ignore')

# Buat folder output
os.makedirs('output', exist_ok=True)

# ============================================
# 1. LOAD DATA
# ============================================
df = load_data('data/USA_cars_datasets.csv')

# ============================================
# 2. HANDLING OUTLIER
# ============================================
df = handle_outlier(df)

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
df = feature_engineering(df)

# ============================================
# 4. PREPROCESSING
# ============================================
X_train, X_test, y_train, y_test, X_scaled, df_clean = preprocess_data(df)

# ============================================
# 5. MODELING
# ============================================
print("\n" + "="*60)
print("MODELING")
print("="*60)

# Model 1: Linear Regression
print("\n--- Model 1: Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
results_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")

# Model 2: Random Forest (Default)
print("\n--- Model 2: Random Forest (Default) ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
results_rf = evaluate_model(y_test, y_pred_rf, "Random Forest (Default)")

# ============================================
# 6. HYPERPARAMETER TUNING
# ============================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_grid, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

print(f"✅ Parameter terbaik: {grid_search.best_params_}")
print(f"✅ R² Score terbaik (CV): {grid_search.best_score_:.4f}")

y_pred_rf_tuned = grid_search.predict(X_test)
results_rf_tuned = evaluate_model(y_test, y_pred_rf_tuned, "Random Forest (Tuned)")
joblib.dump(grid_search.best_estimator_, 'output/rf_tuned.pkl')

# ============================================
# 7. PERBANDINGAN PERFORMANCE
# ============================================
print("\n" + "="*60)
print("PERBANDINGAN PERFORMANCE")
print("="*60)

print(f"{'Model':<25} {'R²':<10} {'MAE':<15} {'RMSE':<15}")
print("-"*65)
print(f"{'Linear Regression':<25} {results_lr['R2']:<10.4f} ${results_lr['MAE']:<14,.2f} ${results_lr['RMSE']:<14,.2f}")
print(f"{'Random Forest (Default)':<25} {results_rf['R2']:<10.4f} ${results_rf['MAE']:<14,.2f} ${results_rf['RMSE']:<14,.2f}")
print(f"{'Random Forest (Tuned)':<25} {results_rf_tuned['R2']:<10.4f} ${results_rf_tuned['MAE']:<14,.2f} ${results_rf_tuned['RMSE']:<14,.2f}")

# ============================================
# 8. I/O FILE - SAVE STATISTICS TO TXT
# ============================================
save_statistics_to_txt(X_train, X_test, y_train, y_test, 
                       results_lr, results_rf, results_rf_tuned)

# ============================================
# 9. VISUALISASI
# ============================================
feature_importance = create_visualizations(
    y_test, y_pred_lr, y_pred_rf, y_pred_rf_tuned,
    results_lr, results_rf, results_rf_tuned,
    X_scaled, rf_model, df
)

# ============================================
# 10. KESIMPULAN
# ============================================
print("\n" + "="*60)
print("KESIMPULAN")
print("="*60)

best_model = "Linear Regression" if results_lr['R2'] >= results_rf_tuned['R2'] else "Random Forest (Tuned)"
best_r2 = max(results_lr['R2'], results_rf_tuned['R2'])

print(f"\n✅ Model terbaik: {best_model}")
print(f"✅ R² Score terbaik: {best_r2:.4f}")

print("\n📊 Insight:")
print(f"1. Model terbaik mampu menjelaskan {best_r2*100:.2f}% variasi harga")
print(f"2. Rata-rata error prediksi terbaik: ${min(results_lr['MAE'], results_rf_tuned['MAE']):,.2f}")
print(f"3. Fitur paling penting: {feature_importance.iloc[-1]['feature']}")

print("\n" + "="*60)
print("PROJECT SELESAI!")
print("="*60)