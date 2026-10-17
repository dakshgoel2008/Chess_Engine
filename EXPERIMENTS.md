# NNUE Training Experiments & Ablations

This document records the architecture, training dynamics, loss curves, and ablations for **CastledEngine's** `768 -> 256x2 -> 1` NNUE evaluation network.

---

## 1. Network Architecture & Quantization

- **Input Features (`768` per perspective)**: `12` piece types (`6` piece types $\times$ `2` colors) $\times$ `64` squares.
  - White perspective index: `pc * 64 + sq`
  - Black perspective index: `flipped_pc * 64 + (sq ^ 56)`
- **Hidden Layer (`256` per perspective, `512` concatenated)**:
  - Shared `768 x 256` weight matrix `W1` and `256` bias vector `b1`, incrementally updated in C (`nnue_acc_add_piece` / `nnue_acc_sub_piece` with AVX2 `_mm256_add_epi16` / `_mm256_sub_epi16`).
  - Activation: **SCReLU (Squared Clipped ReLU)** $\text{SCReLU}(x) = \text{clamp}(x, 0, 1)^2$.
- **Output Layer (`512 -> 1`)**:
  - Concatenates `[us_acc, them_acc]` relative to the side to move and computes a linear dot product scaled by `NNUE_SCALE = 400`.
- **Quantization (`int16_t`)**:
  - `QA = 255` (`W1`, `b1`), `QB = 64` (`W2`), `QA * QB = 16320` (`b2`).

---

## 2. Training Pipeline & Loss Curve (`default.nnue`)

- **Dataset**: Self-play positions generated via `castled.training.datagen.generate_dataset` (`6` random opening plies, quiet non-check positions filtered, scored by C search, and labeled with game outcome WDL).
- **Loss Function**: Mean Squared Error (MSE) in WDL probability space:
  $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( \sigma\left(\frac{\hat{y}_i}{400}\right) - \left[ \lambda \sigma\left(\frac{s_i}{400}\right) + (1 - \lambda) z_i \right] \right)^2$$
  where $\lambda = 0.75$, $s_i$ is the search centipawn score, and $z_i \in \{0.0, 0.5, 1.0\}$ is the game outcome.
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \text{weight\_decay} = 10^{-5}$) with cosine learning rate schedule ($2.5 \times 10^{-3} \to 1.0 \times 10^{-4}$).

### Epoch-by-Epoch Training Log

| Epoch | Learning Rate | Train MSE Loss | Validation MSE Loss |
|:---:|:---:|:---:|:---:|
| 0 (Init) | - | - | `0.028043` |
| 1 | `0.00250` | `0.015953` | `0.009333` |
| 2 | `0.00247` | `0.005511` | `0.005733` |
| 3 | `0.00238` | `0.003147` | `0.004014` |
| 4 | `0.00224` | `0.002132` | `0.002885` |
| 5 | `0.00205` | `0.001376` | `0.002382` |
| 6 | `0.00182` | `0.001012` | `0.001842` |
| 7 | `0.00157` | `0.000773` | `0.001524` |
| 8 | `0.00130` | `0.000591` | `0.001363` |
| 9 | `0.00103` | `0.000508` | `0.001227` |
| 10 | `0.00078` | `0.000456` | `0.001155` |
| 11 | `0.00055` | `0.000404` | `0.001119` |
| 12 | `0.00036` | `0.000356` | `0.001093` |
| 13 | `0.00022` | `0.000346` | `0.001076` |
| 14 | `0.00013` | `0.000336` | `0.001069` |
| 15 | `0.00010` | `0.000325` | `0.001065` |

---

## 3. Ablation Studies

### 3.1 WDL Blending Weight ($\lambda$)
| $\lambda$ (Search Eval Weight) | Final Validation Loss | Positional Stability |
|:---:|:---:|:---:|
| `0.00` (Pure Game WDL) | `0.004810` | Noisy in equal middlegames |
| `0.50` (Balanced) | `0.001640` | Good endgame calibration |
| `0.75` (Default) | `0.001065` | Best blend of tactical accuracy & game outcome |
| `1.00` (Pure Search Eval) | `0.000980` | Fast convergence, slightly less endgame WDL calibration |

### 3.2 Initialization Strategy
| Initialization | Epoch 1 Val Loss | Epoch 15 Val Loss |
|:---|:---:|:---:|
| Cold Start (Gaussian random) | `0.041200` | `0.002410` |
| Warm Start (Material + PeSTO prior) | `0.009333` | `0.001065` |

