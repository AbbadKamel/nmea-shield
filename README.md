# N2KShield

**A Lightweight CNN Autoencoder IDS for NMEA 2000 Maritime Networks**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Non-Commercial](https://img.shields.io/badge/Use-Non--Commercial%20Only-red.svg)](#license)

> ## ⚠️ NON-COMMERCIAL USE ONLY
>
> This work is the **property of NEAC Industry** (Caen, France) and is released for
> **academic and research purposes only**. **Commercial use is not permitted** without the
> prior written authorisation of NEAC Industry. See [License](#license).
>
> The NMEA 2000 dataset was collected aboard an operational vessel provided by NEAC
> Industry, with dataset acquisition supported by **Marc-Antoine Gambin**
> (marc-antoine.gambin@neac-industry.com) and **Lionnel Mesnil**
> (lionnel.mesnil@neac-industry.fr).

---

## Overview

NMEA 2000 (N2K) inherits CAN's lack of authentication, so an attacker with bus access can
inject **semantically plausible** values — a subtly shifted heading, speed, depth or
position — that stay protocol-compliant while corrupting the vessel's situational
awareness. Frequency- and timing-based monitors miss these.

**N2KShield** detects them without attack labels and without protocol changes: navigation
signals are aggregated into fixed-length windows, a convolutional autoencoder learns normal
behaviour from **benign traffic only**, and windows whose reconstruction error exceeds a
percentile threshold are flagged.

## Method

- **Features** — 1 Hz-aligned signals, windowed at `w` seconds. Per signal: `mean`, `std`,
  **`min`, `max`** → `4m` features, reshaped to `(w, 4m, 1)`. The boundary values matter:
  mean/std alone smooth away transients like a rudder spike.
- **Model** — 2D CAE. Encoder: 3 conv blocks (5×5, 5×5, 3×3 / 32, 16, 16 filters),
  LeakyReLU α = 0.2, MaxPooling2D; decoder mirrors with UpSampling2D. Adam
  (lr 2×10⁻⁴, β₁ 0.5, β₂ 0.99), batch 128, early stopping.
- **Threshold** — τ = 99th percentile of reconstruction loss on **benign validation
  windows**. Alarm if `MSE(X, X̂) > τ`.
- Train/val are benign only; attacks are injected into the held-out test split.

## Results

`w = 75 s` (best of 50 / 75 / 100 s), on nearly 3 million frames from an 85-minute voyage.

| Attack | Samples | TP | Recall | AUROC |
|---|---:|---:|---:|---:|
| Spike | 50 | 50 | 100.0% | 1.000 |
| Noise | 50 | 50 | 100.0% | 1.000 |
| Drift | 50 | 46 | 92.0% | 0.982 |
| Scaling | 50 | 19 | 38.0% | 0.868 |
| Constant | 50 | 6 | 12.0% | 0.795 |
| **Overall** | **250** | **171** | **68.4%** | **0.929** |

| Method | AUROC | Prec. | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| **N2KShield** | **0.929** | **1.000** | 0.684 | **0.812** | **0.0%** |
| One-Class SVM | 0.732 | 0.285 | 0.750 | 0.405 | 100.0% |
| Isolation Forest | 0.576 | 0.315 | 0.280 | 0.297 | 24.3% |
| LOF | 0.641 | 0.298 | 0.650 | 0.407 | 84.9% |

N2KShield is the only method with **0% false positives** at **100% precision**. Scaling and
constant-value attacks remain hard — they preserve plausible ranges and correlations.

## Data

> ### 🔒 To obtain the entire dataset, contact **NEAC Industry** (Caen, France).
>
> The full capture is NEAC Industry proprietary data and is **not** distributed here.
> Access requires their prior written authorisation and is limited to non-commercial use.

`data/sample_raw_10min.txt.gz` — **10 minutes of raw N2K bus traffic** (355,860 frames, hex),
provided as a sample.

Full capture, for reference:

| Raw frames | Duration | Signals monitored |
|---:|---:|---:|
| 2,984,250 | ~85 min | 15 |

## Code

```
src/
├── config.py       signals, window sizes, hyperparameters
├── preprocess.py   1 Hz alignment → windowing → mean/std/min/max features
├── models.py       the convolutional autoencoder
├── train.py        benign-only training
├── attacks.py      spike / noise / drift / scaling / constant injectors
└── evaluate.py     percentile threshold, AUROC / precision / recall / FPR
```

`train.py` fits on benign windows only; `attacks.py` builds the test set.

```bash
pip install -r requirements.txt
python -m src.preprocess --input <signals_csv> --window 75
python -m src.train
python -m src.evaluate
```

## Citation

Citation will be provided here.

## License

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) —
see [LICENSE](LICENSE). Share and adapt for **non-commercial** purposes only, with
attribution, derivatives under the same terms.

**This work is the property of NEAC Industry (Caen, France).** Commercial use is not
permitted without their prior written authorisation.

## Acknowledgments

Carried out at and supported by **NEAC Industry** (Caen, France), which provided access to
the operational vessel. Dataset acquisition supported by **Marc-Antoine Gambin** and
**Lionnel Mesnil**.
