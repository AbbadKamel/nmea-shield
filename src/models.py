"""
N2KShield Model Architecture
============================
2D Convolutional Autoencoder for NMEA 2000 anomaly detection.

Architecture:
- Encoder: 3 blocks of Conv2D → LeakyReLU → MaxPooling2D
- Decoder: 3 blocks of Conv2D → LeakyReLU → UpSampling2D
- Output: Conv2D (linear activation) → Cropping2D

Total Parameters: 37,825 (~148 KB)
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    LeakyReLU,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
    Cropping2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from config import MODEL_CONFIG, TRAINING_CONFIG, NUM_FEATURES


def create_autoencoder(time_steps: int, num_features: int = NUM_FEATURES) -> Sequential:
    """
    Create a 2D Convolutional Autoencoder.
    
    Args:
        time_steps: Window size in seconds (e.g., 75)
        num_features: Number of features (default: 60 = 15 signals × 4 stats)
    
    Returns:
        Compiled Keras Sequential model
    """
    in_shape = (time_steps, num_features, 1)
    encoder_cfg = MODEL_CONFIG['encoder']
    decoder_cfg = MODEL_CONFIG['decoder']
    alpha = MODEL_CONFIG['leaky_relu_alpha']
    
    model = Sequential()
    
    # ==================== ENCODER ====================
    # Zero padding to handle edge cases
    model.add(ZeroPadding2D((2, 2), input_shape=in_shape))
    
    # Encoder Block 1
    model.add(Conv2D(
        encoder_cfg[0]['filters'],
        encoder_cfg[0]['kernel_size'],
        strides=(1, 1),
        padding='same'
    ))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D((2, 2), padding='same'))
    
    # Encoder Block 2
    model.add(Conv2D(
        encoder_cfg[1]['filters'],
        encoder_cfg[1]['kernel_size'],
        strides=(1, 1),
        padding='same'
    ))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D((2, 2), padding='same'))
    
    # Encoder Block 3
    model.add(Conv2D(
        encoder_cfg[2]['filters'],
        encoder_cfg[2]['kernel_size'],
        strides=(1, 1),
        padding='same'
    ))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D((2, 2), padding='same'))
    
    # ==================== DECODER ====================
    # Decoder Block 1
    model.add(Conv2D(
        decoder_cfg[0]['filters'],
        decoder_cfg[0]['kernel_size'],
        strides=(1, 1),
        padding='same'
    ))
    model.add(LeakyReLU(alpha=alpha))
    model.add(UpSampling2D((2, 2)))
    
    # Decoder Block 2
    model.add(Conv2D(
        decoder_cfg[1]['filters'],
        decoder_cfg[1]['kernel_size'],
        strides=(1, 1),
        padding='same'
    ))
    model.add(LeakyReLU(alpha=alpha))
    model.add(UpSampling2D((2, 2)))
    
    # Decoder Block 3
    model.add(Conv2D(
        decoder_cfg[2]['filters'],
        decoder_cfg[2]['kernel_size'],
        strides=(1, 1),
        padding='same'
    ))
    model.add(LeakyReLU(alpha=alpha))
    model.add(UpSampling2D((2, 2)))
    
    # ==================== OUTPUT ====================
    # Calculate cropping needed to match input shape
    temp_shape = model.output_shape
    diff_h = temp_shape[1] - in_shape[0]
    top = int(diff_h / 2)
    bottom = diff_h - top
    diff_w = temp_shape[2] - in_shape[1]
    left = int(diff_w / 2)
    right = diff_w - left
    
    # Output convolution with linear activation
    model.add(Conv2D(
        1,
        (3, 3),
        strides=(1, 1),
        padding='same',
        activation=MODEL_CONFIG['output_activation']
    ))
    
    # Crop to match input dimensions exactly
    model.add(Cropping2D(((top, bottom), (left, right))))
    
    return model


def compile_model(model: Sequential) -> Sequential:
    """
    Compile the autoencoder with Adam optimizer and MSE loss.
    
    Args:
        model: Keras Sequential model
    
    Returns:
        Compiled model
    """
    optimizer = Adam(
        learning_rate=TRAINING_CONFIG['learning_rate'],
        beta_1=TRAINING_CONFIG['beta_1'],
        beta_2=TRAINING_CONFIG['beta_2'],
    )
    
    model.compile(
        optimizer=optimizer,
        loss=TRAINING_CONFIG['loss'],
    )
    
    return model


def get_model(time_steps: int, num_features: int = NUM_FEATURES) -> Sequential:
    """
    Create and compile the autoencoder model.
    
    Args:
        time_steps: Window size in seconds
        num_features: Number of features
    
    Returns:
        Compiled Keras model ready for training
    """
    model = create_autoencoder(time_steps, num_features)
    model = compile_model(model)
    return model


if __name__ == '__main__':
    # Demo: Create model and print summary
    from config import DEFAULT_WINDOW_SIZE
    
    model = get_model(DEFAULT_WINDOW_SIZE)
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Memory footprint: ~{model.count_params() * 4 / 1024:.1f} KB (FP32)")
