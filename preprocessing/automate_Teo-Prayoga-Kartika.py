"""
automate_Teo-Prayoga-Kartika.py
Automated Preprocessing Pipeline for Wine Quality Dataset

Kelas  : Membangun Sistem Machine Learning (SMSML)
Penulis: Teo Prayoga Kartika
Deskripsi:
    Script ini melakukan preprocessing otomatis terhadap dataset Wine Quality.
    Fungsi-fungsi di bawah ini merepresentasikan langkah-langkah yang dilakukan
    pada notebook eksperimen secara terstruktur dan modular.

Penggunaan:
    python automate_Teo-Prayoga-Kartika.py --input <path_raw> --output <path_output>
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath (str): Path ke file CSV dataset mentah.

    Returns:
        pd.DataFrame: DataFrame yang sudah dimuat.
    """
    print(f"[INFO] Loading data dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset berhasil dimuat. Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ──────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Menangani missing values dengan mengisi kolom numerik menggunakan median.

    Args:
        df (pd.DataFrame): DataFrame input.
        target_col (str): Nama kolom target yang dikecualikan dari imputasi.

    Returns:
        pd.DataFrame: DataFrame dengan missing values yang sudah ditangani.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    missing_before = df.isnull().sum().sum()

    if missing_before == 0:
        print(f"[INFO] Tidak ada missing values pada dataset.")
    else:
        for col in feature_cols:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        print(f"[INFO] {missing_before} missing values berhasil diisi dengan median.")

    return df


# ──────────────────────────────────────────────
# 3. HANDLE OUTLIERS
# ──────────────────────────────────────────────

def handle_outliers(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Capping (Winsorization).

    Nilai di bawah Q1 - 1.5*IQR akan di-clip ke lower bound,
    dan nilai di atas Q3 + 1.5*IQR akan di-clip ke upper bound.

    Args:
        df (pd.DataFrame): DataFrame input.
        target_col (str): Nama kolom target yang dikecualikan.

    Returns:
        pd.DataFrame: DataFrame dengan outlier yang sudah ditangani.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    df_clean = df.copy()

    total_outliers = 0
    for col in feature_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)].shape[0]
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        total_outliers += n_outliers

    print(f"[INFO] Outlier handling selesai. Total {total_outliers} nilai di-cap menggunakan IQR.")
    return df_clean


# ──────────────────────────────────────────────
# 4. FEATURE SCALING
# ──────────────────────────────────────────────

def scale_features(X: pd.DataFrame) -> tuple:
    """
    Melakukan standarisasi fitur menggunakan StandardScaler.

    Args:
        X (pd.DataFrame): DataFrame fitur input.

    Returns:
        tuple: (X_scaled DataFrame, scaler object)
    """
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns)
    print(f"[INFO] Feature scaling selesai. Mean setelah scaling: {X_scaled.mean().mean():.6f}")
    return X_scaled, scaler


# ──────────────────────────────────────────────
# 5. TRAIN-TEST SPLIT
# ──────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Membagi dataset menjadi data latih dan data uji.

    Args:
        X (pd.DataFrame): Fitur.
        y (pd.Series): Target.
        test_size (float): Proporsi data uji (default 0.2).
        random_state (int): Seed untuk reproduktibilitas.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"[INFO] Split selesai. Train: {X_train.shape[0]} sampel | Test: {X_test.shape[0]} sampel")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# 6. SAVE OUTPUT
# ──────────────────────────────────────────────

def save_output(X_train, X_test, y_train, y_test,
                X_scaled, y, output_dir: str) -> None:
    """
    Menyimpan hasil preprocessing ke direktori output.

    Args:
        X_train, X_test, y_train, y_test: Data hasil split.
        X_scaled (pd.DataFrame): Seluruh fitur yang sudah di-scale.
        y (pd.Series): Seluruh target.
        output_dir (str): Direktori tujuan penyimpanan.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    df_full = X_scaled.copy()
    df_full['target'] = y.values
    df_full.to_csv(os.path.join(output_dir, 'winequality_preprocessed.csv'), index=False)

    print(f"[INFO] Hasil preprocessing disimpan di: {output_dir}")
    print(f"[INFO] File yang tersimpan: {os.listdir(output_dir)}")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def preprocess(input_path: str, output_dir: str, target_col: str = 'target') -> pd.DataFrame:
    """
    Pipeline preprocessing lengkap: load → clean → scale → split → save.

    Args:
        input_path (str): Path ke file CSV raw.
        output_dir (str): Direktori output untuk data yang sudah diproses.
        target_col (str): Nama kolom target.

    Returns:
        pd.DataFrame: Dataset yang sudah diproses dan siap dilatih.
    """
    print("=" * 60)
    print("  AUTOMATED PREPROCESSING PIPELINE")
    print("  Wine Quality Dataset - Teo Prayoga Kartika")
    print("=" * 60)

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Handle missing values
    df = handle_missing_values(df, target_col=target_col)

    # Step 3: Handle outliers
    df = handle_outliers(df, target_col=target_col)

    # Step 4: Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Step 5: Scale features
    X_scaled, scaler = scale_features(X)

    # Step 6: Split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Step 7: Save
    save_output(X_train, X_test, y_train, y_test, X_scaled, y, output_dir)

    print("=" * 60)
    print("  PREPROCESSING SELESAI!")
    print("=" * 60)

    # Return full preprocessed dataset
    result = X_scaled.copy()
    result[target_col] = y.values
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated Preprocessing Wine Quality')
    parser.add_argument('--input', type=str,
                        default='../winequality_raw/winequality_raw.csv',
                        help='Path ke file CSV raw dataset')
    parser.add_argument('--output', type=str,
                        default='winequality_preprocessing',
                        help='Direktori output untuk data yang sudah diproses')
    parser.add_argument('--target', type=str, default='target',
                        help='Nama kolom target')
    args = parser.parse_args()

    preprocess(
        input_path=args.input,
        output_dir=args.output,
        target_col=args.target
    )
