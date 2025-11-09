# EEG Signal Classification (Extended Version)

This MATLAB project simulates multi-band EEG-like signals and classifies them using both SVM and MLP models.  
Inspired by Dr. **Jihye Bae’s** research on neural signal processing and reinforcement learning for brain–machine interfaces.

---

## 📘 Overview
This project explores signal processing and machine learning techniques for EEG-like data.  
We simulate α (10 Hz), β (20 Hz), and γ (40 Hz) EEG bands, apply 8–45 Hz bandpass filtering, extract time–frequency features, and train classifiers to distinguish between signal types.

---

## 🧩 Features Extracted
| Category | Features |
|-----------|-----------|
| Time-domain | Mean, Variance |
| Frequency-domain | Power (8–45 Hz), Dominant Frequency, Power Ratio (α/β) |

---

## ⚙️ Models
1. **SVM (Linear)** – baseline model  
2. **MLP (1 hidden layer, 10 neurons)** – nonlinear model trained with backpropagation

---

## 🧠 Results
| Model | Accuracy |
|--------|-----------|
| SVM | 94–97 % |
| MLP | 97–99 % |

---

## 📊 Figures
| Description | File |
|--------------|------|
| Simulated EEG signals | `Figure_1.png` |
| Power spectral density | `Figure_2.png` |
| 3D feature distribution | `Figure_3.png` |
| Confusion matrix (MLP) | `Figure_4.png` |

---

## 📂 File Structure
