"""
N2KShield Evaluation
====================
Evaluate detection performance and compare with baselines.
"""

import os
import argparse
import numpy as np
from typing import Dict, Tuple
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

from config import (
    ATTACK_TYPES, NUM_ATTACK_SAMPLES, THRESHOLD_PERCENTILE,
    RESULTS_DIR, MODELS_DIR
)
from attacks import generate_attacks


def load_model_and_data(models_dir: str = MODELS_DIR,
                        results_dir: str = RESULTS_DIR):
    """Load trained model, threshold, and test data."""
    print("Loading model and data...")
    
    model = tf.keras.models.load_model(
        os.path.join(models_dir, 'autoencoder_best.keras')
    )
    threshold = np.load(os.path.join(models_dir, 'threshold.npy'))
    X_test = np.load(os.path.join(results_dir, 'X_test.npy'))
    
    print(f"  Model parameters: {model.count_params():,}")
    print(f"  Threshold: {threshold:.8f}")
    print(f"  X_test shape: {X_test.shape}")
    
    return model, threshold, X_test


def compute_mse(model, X: np.ndarray) -> np.ndarray:
    """
    Compute reconstruction MSE for each window.
    
    Args:
        model: Trained autoencoder
        X: Input windows
    
    Returns:
        Array of MSE values
    """
    reconstructions = model.predict(X, verbose=0)
    mse = np.mean(np.square(X - reconstructions), axis=(1, 2, 3))
    return mse


def evaluate_detection(model, X_normal: np.ndarray, X_attack: np.ndarray,
                       threshold: float) -> Dict:
    """
    Evaluate detection performance.
    
    Args:
        model: Trained autoencoder
        X_normal: Normal test windows
        X_attack: Attack windows
        threshold: Detection threshold
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Compute MSE
    mse_normal = compute_mse(model, X_normal)
    mse_attack = compute_mse(model, X_attack)
    
    # Create labels
    y_true = np.concatenate([
        np.zeros(len(mse_normal)),  # Normal = 0
        np.ones(len(mse_attack))    # Attack = 1
    ])
    y_scores = np.concatenate([mse_normal, mse_attack])
    
    # Binary predictions
    y_pred = (y_scores > threshold).astype(int)
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    auroc = roc_auc_score(y_true, y_scores)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'auroc': auroc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'mse_normal': mse_normal,
        'mse_attack': mse_attack,
    }


def evaluate_per_attack(model, X_normal: np.ndarray, threshold: float,
                        attack_types: list = ATTACK_TYPES,
                        num_per_type: int = NUM_ATTACK_SAMPLES) -> Dict:
    """
    Evaluate detection performance for each attack type.
    
    Args:
        model: Trained autoencoder
        X_normal: Normal test windows
        threshold: Detection threshold
        attack_types: List of attack types
        num_per_type: Samples per attack type
    
    Returns:
        Dictionary with per-attack metrics
    """
    results = {}
    
    for attack_type in attack_types:
        # Generate attacks of this type
        X_attack, _, _ = generate_attacks(
            X_normal,
            attack_types=[attack_type],
            num_per_type=num_per_type,
            intensity='medium'
        )
        
        # Compute MSE
        mse_attack = compute_mse(model, X_attack)
        
        # Detection
        detected = np.sum(mse_attack > threshold)
        recall = detected / len(mse_attack)
        
        # AUROC against normal
        mse_normal = compute_mse(model, X_normal[:len(X_attack)])
        y_true = np.concatenate([np.zeros(len(mse_normal)), np.ones(len(mse_attack))])
        y_scores = np.concatenate([mse_normal, mse_attack])
        auroc = roc_auc_score(y_true, y_scores)
        
        results[attack_type] = {
            'samples': len(X_attack),
            'tp': detected,
            'recall': recall,
            'auroc': auroc,
        }
        
        print(f"  {attack_type}: Recall={recall:.1%}, AUROC={auroc:.3f}")
    
    return results


def run_baseline_comparison(X_train: np.ndarray, X_test_normal: np.ndarray,
                            X_attack: np.ndarray) -> Dict:
    """
    Compare with baseline anomaly detection methods.
    
    Baselines:
    - One-Class SVM
    - Isolation Forest
    - Local Outlier Factor
    
    Args:
        X_train: Training data (benign)
        X_test_normal: Normal test data
        X_attack: Attack data
    
    Returns:
        Dictionary with baseline results
    """
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    
    print("\nRunning baseline comparison...")
    
    # Flatten windows for traditional ML
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_normal_flat = X_test_normal.reshape(len(X_test_normal), -1)
    X_attack_flat = X_attack.reshape(len(X_attack), -1)
    
    # Labels
    y_true = np.concatenate([
        np.zeros(len(X_normal_flat)),
        np.ones(len(X_attack_flat))
    ])
    
    baselines = {
        'One-Class SVM': OneClassSVM(kernel='rbf', nu=0.1),
        'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
        'LOF': LocalOutlierFactor(n_neighbors=20, novelty=True),
    }
    
    results = {}
    
    for name, clf in baselines.items():
        print(f"  Training {name}...")
        clf.fit(X_train_flat)
        
        # Predict (-1 = anomaly, 1 = normal)
        y_pred_normal = clf.predict(X_normal_flat)
        y_pred_attack = clf.predict(X_attack_flat)
        
        # Convert: -1 -> 1 (anomaly), 1 -> 0 (normal)
        y_pred = np.concatenate([
            (y_pred_normal == -1).astype(int),
            (y_pred_attack == -1).astype(int)
        ])
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results[name] = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        }
        
        print(f"    Precision: {results[name]['precision']:.3f}")
        print(f"    Recall: {results[name]['recall']:.3f}")
        print(f"    FPR: {results[name]['fpr']:.1%}")
    
    return results


def evaluate(models_dir: str = MODELS_DIR,
             results_dir: str = RESULTS_DIR,
             threshold_percentile: int = THRESHOLD_PERCENTILE):
    """
    Full evaluation pipeline.
    
    Args:
        models_dir: Directory with trained model
        results_dir: Directory with preprocessed data
        threshold_percentile: Percentile for threshold
    """
    # Load
    model, threshold, X_test = load_model_and_data(models_dir, results_dir)
    
    # Generate attacks
    print("\nGenerating attacks...")
    X_attack, _, attack_labels = generate_attacks(
        X_test,
        attack_types=ATTACK_TYPES,
        num_per_type=NUM_ATTACK_SAMPLES,
        intensity='medium'
    )
    
    # Overall evaluation
    print("\n=== Overall Detection Performance ===")
    results = evaluate_detection(model, X_test, X_attack, threshold)
    
    print(f"  AUROC: {results['auroc']:.3f}")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1: {results['f1']:.3f}")
    print(f"  FPR: {results['fpr']:.1%}")
    
    # Per-attack evaluation
    print("\n=== Per-Attack Detection Performance ===")
    per_attack = evaluate_per_attack(model, X_test, threshold)
    
    # Save results
    np.save(os.path.join(results_dir, 'evaluation_results.npy'), {
        'overall': results,
        'per_attack': per_attack,
    })
    
    print(f"\nResults saved to {results_dir}")
    
    return results, per_attack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate N2KShield')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR,
                        help='Directory with trained model')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Directory with preprocessed data')
    parser.add_argument('--threshold-percentile', type=int, 
                        default=THRESHOLD_PERCENTILE,
                        help='Threshold percentile')
    
    args = parser.parse_args()
    
    evaluate(
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        threshold_percentile=args.threshold_percentile,
    )
