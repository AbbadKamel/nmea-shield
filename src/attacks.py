"""
N2KShield Attack Generation
===========================
Generate synthetic attacks for evaluation.

Attack Types:
- Spike: Sudden value jump
- Noise: Random interference
- Drift: Gradual change
- Scaling: Multiplicative change
- Constant: Fixed value forcing
"""

import numpy as np
from typing import Tuple, List

from config import ATTACK_TYPES, NUM_ATTACK_SAMPLES, ATTACK_INTENSITIES


def generate_spike_attack(window: np.ndarray, 
                          intensity: float = 0.6) -> np.ndarray:
    """
    Generate spike attack: sudden value jump at random timestep.
    
    Args:
        window: Input window of shape (time_steps, features, 1)
        intensity: Attack intensity (0-1)
    
    Returns:
        Attacked window
    """
    attacked = window.copy()
    t, f, _ = attacked.shape
    
    # Random timestep and feature
    attack_t = np.random.randint(t)
    attack_f = np.random.randint(f)
    
    # Spike value
    spike_value = np.clip(attacked[attack_t, attack_f, 0] + intensity, 0, 1)
    attacked[attack_t, attack_f, 0] = spike_value
    
    return attacked


def generate_noise_attack(window: np.ndarray,
                          intensity: float = 0.3) -> np.ndarray:
    """
    Generate noise attack: add random noise to signal.
    
    Args:
        window: Input window
        intensity: Noise standard deviation
    
    Returns:
        Attacked window
    """
    attacked = window.copy()
    t, f, _ = attacked.shape
    
    # Random feature
    attack_f = np.random.randint(f)
    
    # Add Gaussian noise
    noise = np.random.normal(0, intensity, t)
    attacked[:, attack_f, 0] = np.clip(attacked[:, attack_f, 0] + noise, 0, 1)
    
    return attacked


def generate_drift_attack(window: np.ndarray,
                          intensity: float = 0.5) -> np.ndarray:
    """
    Generate drift attack: gradual increase/decrease over time.
    
    Args:
        window: Input window
        intensity: Maximum drift amount
    
    Returns:
        Attacked window
    """
    attacked = window.copy()
    t, f, _ = attacked.shape
    
    # Random feature
    attack_f = np.random.randint(f)
    
    # Random start point
    start_t = np.random.randint(0, t // 2)
    
    # Gradual drift
    drift_steps = t - start_t
    drift = np.linspace(0, intensity, drift_steps)
    
    # Random direction
    if np.random.rand() > 0.5:
        drift = -drift
    
    attacked[start_t:, attack_f, 0] = np.clip(
        attacked[start_t:, attack_f, 0] + drift, 0, 1
    )
    
    return attacked


def generate_scaling_attack(window: np.ndarray,
                            intensity: float = 0.5) -> np.ndarray:
    """
    Generate scaling attack: multiply signal by a factor.
    
    Args:
        window: Input window
        intensity: Scaling factor deviation from 1.0
    
    Returns:
        Attacked window
    """
    attacked = window.copy()
    t, f, _ = attacked.shape
    
    # Random feature
    attack_f = np.random.randint(f)
    
    # Scaling factor
    scale = 1.0 + intensity * (1 if np.random.rand() > 0.5 else -1)
    scale = max(0.1, scale)  # Avoid negative/zero scaling
    
    attacked[:, attack_f, 0] = np.clip(attacked[:, attack_f, 0] * scale, 0, 1)
    
    return attacked


def generate_constant_attack(window: np.ndarray,
                             intensity: float = 0.5) -> np.ndarray:
    """
    Generate constant attack: force signal to fixed value.
    
    Args:
        window: Input window
        intensity: Constant value offset
    
    Returns:
        Attacked window
    """
    attacked = window.copy()
    t, f, _ = attacked.shape
    
    # Random feature
    attack_f = np.random.randint(f)
    
    # Random start and duration
    start_t = np.random.randint(0, t // 2)
    duration = np.random.randint(t // 4, t - start_t)
    
    # Constant value (biased toward extreme)
    const_value = 0.9 if np.random.rand() > 0.5 else 0.1
    
    attacked[start_t:start_t + duration, attack_f, 0] = const_value
    
    return attacked


ATTACK_FUNCTIONS = {
    'spike': generate_spike_attack,
    'noise': generate_noise_attack,
    'drift': generate_drift_attack,
    'scaling': generate_scaling_attack,
    'constant': generate_constant_attack,
}


def generate_attacks(X_normal: np.ndarray,
                     attack_types: List[str] = ATTACK_TYPES,
                     num_per_type: int = NUM_ATTACK_SAMPLES,
                     intensity: str = 'medium'
                     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic attacks on normal windows.
    
    Args:
        X_normal: Normal windows of shape (n, time_steps, features, 1)
        attack_types: List of attack types to generate
        num_per_type: Number of samples per attack type
        intensity: Attack intensity level ('low', 'medium', 'high')
    
    Returns:
        (X_attack, y_attack, attack_labels) tuple
    """
    intensity_value = ATTACK_INTENSITIES.get(intensity, 0.6)
    
    X_attack = []
    attack_labels = []
    
    for attack_type in attack_types:
        attack_func = ATTACK_FUNCTIONS[attack_type]
        
        for _ in range(num_per_type):
            # Select random normal window
            idx = np.random.randint(len(X_normal))
            window = X_normal[idx]
            
            # Generate attack
            attacked = attack_func(window, intensity_value)
            X_attack.append(attacked)
            attack_labels.append(attack_type)
    
    X_attack = np.array(X_attack)
    y_attack = np.ones(len(X_attack))  # All attacks labeled as 1
    
    print(f"Generated {len(X_attack)} attacks:")
    for at in attack_types:
        count = attack_labels.count(at)
        print(f"  {at}: {count}")
    
    return X_attack, y_attack, attack_labels


if __name__ == '__main__':
    # Demo: Generate attacks on random data
    print("Demo: Generating attacks on synthetic data")
    
    # Create dummy normal windows
    X_normal = np.random.rand(100, 75, 60, 1).astype(np.float32)
    
    # Generate attacks
    X_attack, y_attack, labels = generate_attacks(
        X_normal, 
        attack_types=['spike', 'noise', 'drift'],
        num_per_type=10,
        intensity='medium'
    )
    
    print(f"\nAttack shape: {X_attack.shape}")
