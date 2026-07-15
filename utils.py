# ============================================
# utils.py - Modul Fungsi untuk Final Project
# Prediksi Harga Mobil Bekas
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(filepath):
    """
    Memuat dataset dari file CSV
    """
    df = pd.read_csv(filepath)
    print("="*60)
    print("INFORMASI DATASET")
    print("="*60)
    print(f"Jumlah baris: {df.shape[0]}")
    print(f"Jumlah kolom: {df.shape[1]}")
    print("\n5 Data Pertama:")
    print(df.head())
    return df


def handle_outlier(df):
    """
    Menghapus outlier pada kolom price menggunakan metode IQR
    """
    print("\n" + "="*60)
    print("HANDLING OUTLIER")
    print("="*60)
    
    print(f"Jumlah harga $0: {len(df[df['price'] == 0])}")
    print(f"Jumlah harga di bawah $1000: {len(df[df['price'] < 1000])}")
    
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(500, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"Batas bawah harga: ${lower_bound:,.0f}")
    print(f"Batas atas harga: ${upper_bound:,.0f}")
    
    df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    print(f"Jumlah data sebelum: {len(df)}")
    print(f"Jumlah data setelah: {len(df_clean)}")
    print(f"Data yang dihapus: {len(df) - len(df_clean)}")
    
    return df_clean


def feature_engineering(df):
    """
    Membuat fitur baru: car_age dan mileage_per_year
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    CURRENT_YEAR = 2026
    df['car_age'] = CURRENT_YEAR - df['year']
    df['mileage_per_year'] = df['mileage'] / df['car_age'].clip(lower=1)
    print("Fitur baru dibuat: 'car_age', 'mileage_per_year'")
    
    return df


def preprocess_data(df):
    """
    Preprocessing data: drop kolom, encoding, scaling, split
    """
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)
    
    columns_to_drop = ['vin', 'lot', 'Unnamed: 0', 'country', 'condition']
    df_clean = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Kolom yang dihapus: {columns_to_drop}")
    
    print("\nMissing values:")
    print(df_clean.isnull().sum())
    
    X = df_clean.drop('price', axis=1)
    y = df_clean['price']
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Kolom kategorikal: {categorical_cols}")
    
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Jumlah fitur setelah encoding: {X_encoded.shape[1]}")
    
    numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    joblib.dump(scaler, 'output/scaler.pkl')
    print("✅ Preprocessing selesai")
    
    return X_train, X_test, y_train, y_test, X_scaled, df_clean


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluasi model dengan R², MAE, RMSE
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- {model_name} ---")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : ${mae:,.2f}")
    print(f"RMSE     : ${rmse:,.2f}")
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse}


def save_statistics_to_txt(X_train, X_test, y_train, y_test, results_lr, results_rf, results_rf_tuned, filename='output/hasil_statistik.txt'):
    """
    Menyimpan ringkasan statistik ke file .txt (I/O File)
    """
    print("\n" + "="*60)
    print("MENYIMPAN STATISTIK KE FILE TXT")
    print("="*60)
    
    os.makedirs('output', exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LAPORAN STATISTIK FINAL PROJECT\n")
        f.write("PREDIKSI HARGA MOBIL BEKAS\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. INFORMASI DATASET\n")
        f.write("-"*40 + "\n")
        f.write(f"Jumlah data train: {X_train.shape[0]} baris\n")
        f.write(f"Jumlah data test: {X_test.shape[0]} baris\n")
        f.write(f"Jumlah fitur: {X_train.shape[1]} kolom\n\n")
        
        f.write("2. STATISTIK TARGET (PRICE)\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean target (train): ${y_train.mean():,.2f}\n")
        f.write(f"Median target (train): ${y_train.median():,.2f}\n")
        f.write(f"Std target (train): ${y_train.std():,.2f}\n")
        f.write(f"Min target (train): ${y_train.min():,.2f}\n")
        f.write(f"Max target (train): ${y_train.max():,.2f}\n\n")
        
        f.write("3. PERFORMA MODEL\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Model':<25} {'R²':<10} {'MAE':<15} {'RMSE':<15}\n")
        f.write("-"*65 + "\n")
        f.write(f"{'Linear Regression':<25} {results_lr['R2']:<10.4f} ${results_lr['MAE']:<14,.2f} ${results_lr['RMSE']:<14,.2f}\n")
        f.write(f"{'Random Forest (Default)':<25} {results_rf['R2']:<10.4f} ${results_rf['MAE']:<14,.2f} ${results_rf['RMSE']:<14,.2f}\n")
        f.write(f"{'Random Forest (Tuned)':<25} {results_rf_tuned['R2']:<10.4f} ${results_rf_tuned['MAE']:<14,.2f} ${results_rf_tuned['RMSE']:<14,.2f}\n\n")
        
        f.write("4. MODEL TERBAIK\n")
        f.write("-"*40 + "\n")
        if results_lr['R2'] >= results_rf_tuned['R2']:
            best = "Linear Regression"
            best_r2 = results_lr['R2']
        else:
            best = "Random Forest (Tuned)"
            best_r2 = results_rf_tuned['R2']
        f.write(f"Model terbaik: {best}\n")
        f.write(f"R² Score: {best_r2:.4f}\n")
        f.write(f"MAE: ${min(results_lr['MAE'], results_rf_tuned['MAE']):,.2f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("File ini dibuat otomatis oleh program\n")
        f.write("Final Project - Mini Machine Learning\n")
        f.write("="*60 + "\n")
    
    print(f"✅ Statistik disimpan ke: {filename}")
    return filename


def create_visualizations(y_test, y_pred_lr, y_pred_rf, y_pred_rf_tuned, 
                          results_lr, results_rf, results_rf_tuned,
                          X_scaled, rf_model, df, filename='output/hasil_statistik.txt'):
    """
    Membuat visualisasi: scatter plot, feature importance, heatmap
    """
    print("\n" + "="*60)
    print("MEMBUAT VISUALISASI")
    print("="*60)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # 1. Scatter Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Perbandingan Prediksi vs Aktual', fontsize=14, fontweight='bold')
    
    models = [
        (y_pred_lr, results_lr, 'Linear Regression', 'blue'),
        (y_pred_rf, results_rf, 'Random Forest', 'green'),
        (y_pred_rf_tuned, results_rf_tuned, 'Random Forest (Tuned)', 'purple')
    ]
    
    for ax, (y_pred, results, name, color) in zip(axes, models):
        ax.scatter(y_test, y_pred, alpha=0.5, color=color, s=15)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Harga Aktual ($)')
        ax.set_ylabel('Harga Prediksi ($)')
        ax.set_title(f'{name}\nR² = {results["R2"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/scatter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Scatter comparison disimpan")
    
    # 2. Feature Importance
    fig, ax = plt.subplots(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'feature': X_scaled.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_importance)))
    ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Fitur Paling Penting - Random Forest', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(feature_importance['importance']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/feature_importance_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Feature importance disimpan")
    
    # 3. Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_cols_heatmap = ['price', 'year', 'mileage', 'car_age', 'mileage_per_year']
    numeric_df = df[numeric_cols_heatmap].dropna()
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Korelasi Fitur Numerik', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/heatmap_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Heatmap disimpan")
    
    print("\n✅ Semua visualisasi selesai!")
    
    return feature_importance