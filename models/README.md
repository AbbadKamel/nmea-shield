# Models Directory

Trained model weights are saved here after running `src/train.py`.

## Files

- `autoencoder_best.keras` - Best model (lowest validation loss)
- `autoencoder_final.keras` - Final model after training
- `threshold.npy` - Detection threshold (99th percentile of validation MSE)
- `history.npy` - Training history (loss curves)
