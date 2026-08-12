"""
N2KShield Preprocessing
=======================
Feature extraction for NMEA 2000 navigation signals.

Pipeline:
1. Load the navigation signal table
2. Resample signals to 1Hz grid
3. Compute statistical features (mean, std, min, max)
4. Apply correlation-based signal ordering
5. Normalize features to [0, 1]
6. Create sliding windows
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from config import (
    SIGNALS, AGGREGATION_STATS, NUM_FEATURES,
    DEFAULT_WINDOW_SIZE, TRAIN_STRIDE, EVAL_STRIDE,
    RESAMPLE_FREQ, TRAIN_RATIO, VAL_RATIO,
    DATA_DIR, RESULTS_DIR
)


def load_frames(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate raw NMEA 2000 frame files.
    
    Args:
        data_dir: Directory containing frame files
    
    Returns:
        DataFrame with decoded frames
    """
    print(f"Loading frames from {data_dir}...")
    
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.csv'):
            fpath = os.path.join(data_dir, fname)
            df = pd.read_csv(fpath)
            frames.append(df)
    
    if not frames:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} frames")
    return df


def resample_to_1hz(df: pd.DataFrame, signals: List[str]) -> pd.DataFrame:
    """
    Resample signals to 1Hz grid using forward-fill.
    
    Args:
        df: DataFrame with timestamp and signal columns
        signals: List of signal names to resample
    
    Returns:
        DataFrame with 1-second timestamps
    """
    print("Resampling to 1Hz...")
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample with forward-fill
    df_resampled = df[signals].resample(RESAMPLE_FREQ).last()
    df_resampled = df_resampled.ffill()
    
    print(f"Resampled to {len(df_resampled):,} seconds")
    return df_resampled.reset_index()


def compute_features(df: pd.DataFrame, signals: List[str], 
                     window_size: int) -> np.ndarray:
    """
    Compute statistical features for each window.
    
    For each signal, computes: mean, std, min, max
    Total features = 15 signals × 4 stats = 60
    
    Args:
        df: Resampled DataFrame
        signals: List of signal names
        window_size: Window size in seconds
    
    Returns:
        Array of shape (num_windows, window_size, num_features)
    """
    print(f"Computing features (window={window_size}s)...")
    
    data = df[signals].values
    num_samples = len(data)
    num_signals = len(signals)
    
    features_list = []
    
    for i in range(num_samples):
        # Get window data (or pad at start)
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i+1]
        
        # Pad if needed
        if len(window_data) < window_size:
            pad_size = window_size - len(window_data)
            window_data = np.vstack([
                np.tile(window_data[0], (pad_size, 1)),
                window_data
            ])
        
        # Compute rolling statistics for each timestep
        timestep_features = []
        for t in range(window_size):
            # Use data up to timestep t
            window_slice = window_data[:t+1]
            
            row = []
            for s in range(num_signals):
                signal_data = window_slice[:, s]
                row.extend([
                    np.mean(signal_data),
                    np.std(signal_data),
                    np.min(signal_data),
                    np.max(signal_data),
                ])
            timestep_features.append(row)
        
        features_list.append(timestep_features)
    
    features = np.array(features_list)
    print(f"Feature shape: {features.shape}")
    return features


def compute_signal_ordering(df: pd.DataFrame, signals: List[str]) -> List[int]:
    """
    Compute optimal signal ordering using hierarchical clustering.
    
    Signals are ordered so that correlated signals are adjacent,
    improving convolutional filter effectiveness.
    
    Args:
        df: DataFrame with signal data
        signals: List of signal names
    
    Returns:
        List of indices for reordering
    """
    print("Computing correlation-based signal ordering...")
    
    # Compute correlation matrix
    corr_matrix = df[signals].corr().abs()
    
    # Convert to distance matrix
    dist_matrix = 1 - corr_matrix
    dist_condensed = squareform(dist_matrix.values, checks=False)
    
    # Hierarchical clustering
    linkage_matrix = linkage(dist_condensed, method='average')
    
    # Get optimal leaf ordering
    order = leaves_list(linkage_matrix)
    
    print(f"Signal order: {[signals[i] for i in order]}")
    return order.tolist()


def normalize_features(X: np.ndarray, 
                       scaler: Optional[MinMaxScaler] = None
                       ) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize features to [0, 1] range.
    
    Args:
        X: Feature array of shape (n, time_steps, features)
        scaler: Optional fitted scaler (use for val/test)
    
    Returns:
        Normalized array and fitted scaler
    """
    print("Normalizing features...")
    
    n, t, f = X.shape
    X_flat = X.reshape(-1, f)
    
    if scaler is None:
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X_flat)
    else:
        X_normalized = scaler.transform(X_flat)
    
    return X_normalized.reshape(n, t, f), scaler


def create_windows(X: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    Create sliding windows from feature array.
    
    Args:
        X: Feature array of shape (n, time_steps, features)
        stride: Step size between windows
    
    Returns:
        Windows array
    """
    indices = range(0, len(X), stride)
    windows = X[list(indices)]
    print(f"Created {len(windows)} windows (stride={stride})")
    return windows


def split_data(X: np.ndarray, 
               train_ratio: float = TRAIN_RATIO,
               val_ratio: float = VAL_RATIO
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data chronologically into train/val/test.
    
    Args:
        X: Feature array
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        (X_train, X_val, X_test) tuple
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test


def preprocess(data_dir: str = DATA_DIR,
               output_dir: str = RESULTS_DIR,
               window_size: int = DEFAULT_WINDOW_SIZE,
               signals: List[str] = SIGNALS) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline.
    
    Args:
        data_dir: Directory with raw frame files
        output_dir: Directory to save processed data
        window_size: Window size in seconds
        signals: List of signals to use
    
    Returns:
        Dictionary with train/val/test arrays
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and resample
    df = load_frames(data_dir)
    df = resample_to_1hz(df, signals)
    
    # Compute signal ordering
    signal_order = compute_signal_ordering(df, signals)
    ordered_signals = [signals[i] for i in signal_order]
    
    # Compute features
    X = compute_features(df, ordered_signals, window_size)
    
    # Split chronologically
    X_train, X_val, X_test = split_data(X)
    
    # Normalize using training statistics
    X_train, scaler = normalize_features(X_train)
    X_val, _ = normalize_features(X_val, scaler)
    X_test, _ = normalize_features(X_test, scaler)
    
    # Create windows with appropriate stride
    X_train = create_windows(X_train, TRAIN_STRIDE)
    X_val = create_windows(X_val, EVAL_STRIDE)
    X_test = create_windows(X_test, EVAL_STRIDE)
    
    # Add channel dimension for 2D CNN
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    # Save
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    
    # Save scaler parameters
    scaler_params = {
        'min': scaler.data_min_.tolist(),
        'max': scaler.data_max_.tolist(),
    }
    with open(os.path.join(output_dir, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f)
    
    # Save signal ordering
    with open(os.path.join(output_dir, 'signal_order.json'), 'w') as f:
        json.dump({'order': signal_order, 'signals': ordered_signals}, f)
    
    print(f"\nSaved to {output_dir}:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess NMEA 2000 data')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW_SIZE,
                        help='Window size in seconds')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Input data directory')
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Output directory')
    
    args = parser.parse_args()
    
    preprocess(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window,
    )
