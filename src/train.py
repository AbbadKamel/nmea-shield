"""
N2KShield Training
==================
Train the convolutional autoencoder on benign NMEA 2000 data.
"""

import os
import argparse
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (
    DEFAULT_WINDOW_SIZE, NUM_FEATURES, TRAINING_CONFIG,
    THRESHOLD_PERCENTILE, RESULTS_DIR, MODELS_DIR
)
from models import get_model


def load_data(results_dir: str = RESULTS_DIR):
    """Load preprocessed training and validation data."""
    print("Loading preprocessed data...")
    
    X_train = np.load(os.path.join(results_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(results_dir, 'X_val.npy'))
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    
    return X_train, X_val


def compute_threshold(model, X_val, percentile: int = THRESHOLD_PERCENTILE):
    """
    Compute anomaly detection threshold from validation set.
    
    Uses the specified percentile of MSE on benign validation data.
    
    Args:
        model: Trained autoencoder
        X_val: Validation data (benign only)
        percentile: Percentile for threshold (default: 99)
    
    Returns:
        Threshold value
    """
    print(f"Computing threshold (P{percentile})...")
    
    reconstructions = model.predict(X_val, verbose=0)
    mse = np.mean(np.square(X_val - reconstructions), axis=(1, 2, 3))
    
    threshold = np.percentile(mse, percentile)
    
    print(f"  Threshold: {threshold:.8f}")
    print(f"  MSE range: [{mse.min():.8f}, {mse.max():.8f}]")
    print(f"  MSE mean: {mse.mean():.8f}")
    
    return threshold


def train(results_dir: str = RESULTS_DIR,
          models_dir: str = MODELS_DIR,
          window_size: int = DEFAULT_WINDOW_SIZE,
          epochs: int = TRAINING_CONFIG['epochs'],
          batch_size: int = TRAINING_CONFIG['batch_size']):
    """
    Train the autoencoder model.
    
    Args:
        results_dir: Directory with preprocessed data
        models_dir: Directory to save model
        window_size: Window size used in preprocessing
        epochs: Maximum training epochs
        batch_size: Training batch size
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    X_train, X_val = load_data(results_dir)
    
    # Verify dimensions
    time_steps = X_train.shape[1]
    num_features = X_train.shape[2]
    
    print(f"\nModel configuration:")
    print(f"  Time steps: {time_steps}")
    print(f"  Features: {num_features}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Create model
    model = get_model(time_steps, num_features)
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=TRAINING_CONFIG['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(models_dir, 'autoencoder_best.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train (autoencoder: input = output)
    print("\nTraining...")
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Compute and save threshold
    threshold = compute_threshold(model, X_val)
    np.save(os.path.join(models_dir, 'threshold.npy'), threshold)
    
    # Save final model
    model.save(os.path.join(models_dir, 'autoencoder_final.keras'))
    
    # Save training history
    np.save(os.path.join(models_dir, 'history.npy'), history.history)
    
    print(f"\nTraining complete!")
    print(f"  Model saved to: {models_dir}")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Threshold (P{THRESHOLD_PERCENTILE}): {threshold:.8f}")
    
    return model, threshold, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train N2KShield autoencoder')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
                        help='Training batch size')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Directory with preprocessed data')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR,
                        help='Directory to save model')
    
    args = parser.parse_args()
    
    train(
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
