# Single-Crossing Monotone Fusion for Streaming Image Authentication

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

## 📖 Project Overview

This project implements a strength-driven phase-transition authentication system for reference-free edge images. Based on extensive empirical findings, the normalized ratio of semantic to geometric residuals varies monotonically with perturbation strength and undergoes a dominant evidence switch at the critical point $S_c$. The system uses lightweight networks to estimate perturbation strength in real-time and smoothly adjusts the fusion weights of semantic and geometric evidence to achieve dynamic optimization of authentication strategies.

### 🎯 Key Innovations

- **Phase Transition Discovery**: First to reveal and verify phase transition laws in authentication
- **Adaptive Fusion Mechanism**: Dynamic weight fusion based on strength estimation
- **Lightweight Design**: End-to-end latency below 9ms, suitable for edge deployment
- **Security Enhancement Protocol**: Integrated challenge-response mechanism to defend against replay attacks

## 🚀 Quick Start

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DiffCrypto_710

# Install dependencies
pip install -r requirements.txt
```

### Quick Demo

```bash
# Run feature demonstration
python demo_phase_transition.py
```

### Complete Experiments

```bash
# Run complete experimental pipeline
python main_phase_transition_experiment.py --output_dir results

# Quick test mode
python main_phase_transition_experiment.py --quick_test
```

## 📁 Project Structure

```
DiffCrypto_710/
├── models/                          # Core models
│   ├── phase_transition_auth.py    # Phase transition authentication system
│   ├── security_protocol.py        # Security enhancement protocol
│   ├── visual_fingerprint.py       # Visual fingerprint generation
│   ├── fingerprint_auth.py         # Fingerprint authentication
│   ├── monodepth.py                # Depth estimation
│   └── clip_vit.py                 # Semantic feature extraction
├── utils/                          # Utility modules
│   ├── enhanced_metrics.py         # Enhanced evaluation metrics
│   ├── metrics.py                  # Basic evaluation metrics
│   ├── chaos_maps.py              # Chaotic maps
│   └── wavelet_chaos.py           # Wavelet chaos
├── experiments/                    # Experimental scripts
│   ├── phase_transition_experiment.py  # Phase transition authentication experiments
│   └── comparison_experiment.py      # Method comparison experiments
├── images/                         # Test images
├── results/                        # Experimental results
├── main.py                         # Original main program
├── main_phase_transition_experiment.py  # New main experimental script
├── demo_phase_transition.py        # Feature demonstration
└── requirements.txt                # Dependencies list
```

## 🔬 Core Algorithms

### 1. Phase Transition Detection

```python
# Stabilized logarithmic residual ratio
z₁(S) = ln[(Δ'geom(S) + ε)/(Δ'sem(S) + ε)]

# Monotonicity verification
kendall_tau, p_value = stats.kendalltau(S, z1)

# Critical point calibration
S_c = find_zero_crossing(z1)
```

### 2. Adaptive Fusion

```python
# Supercritical probability
π(t) = Pr(S > S_c | z_t)

# Adaptive weight
ω(t) = clip(π(t), 0, 1)

# Fused residual
Φ(t) = (1-ω(t))Δ'sem(t) + ω(t)Δ'geom(t)
```

### 3. Security Protocol

```python
# Challenge generation
challenge = {'r': random_128bit, 'Ctr_s': counter, 't_exp': expiration}

# Key derivation
session_key = HKDF(master_key, challenge, domain_info)

# Message binding
message = CBOR{fused_residual, S_hat, S_c, challenge}
hmac = HMAC_SHA256(session_key, message)
```

## 📊 Experimental Results

### Authentication Performance

| Metric | Fixed Fusion | Phase-Aware Fusion | Improvement |
|--------|-------------|-------------------|-------------|
| Accuracy | 96.1% | **98.2%** | +2.1% |
| EER | 3.5% | **2.1%** | -1.4% |
| FRR@FAR=1% | 4.9% | **3.2%** | -1.7% |
| Critical Zone FRR@FAR=1% | 11.2% | **6.7%** | -4.5% |

### Strength Estimation Performance

| Metric | Unconstrained MLP | Monotonic Network | Improvement |
|--------|------------------|------------------|-------------|
| MAE | 0.152 | **0.121** | -20.4% |
| RMSE | 0.041 | **0.029** | -29.3% |
| Pearson r | 0.85 | **0.95** | +11.8% |
| Monotonic Violations | 27 | **8** | -70.4% |

### Security Protocol Performance

- **Success Rate**: 99.5%+
- **Average Response Time**: <1ms
- **Replay Attack Rejection Rate**: 100%
- **Throughput**: 1000+ requests/second

## 🛠️ Usage

### Basic Usage

```python
from models.phase_transition_auth import StrengthDrivenAuthenticator, PhaseTransitionConfig

# Create authentication system
config = PhaseTransitionConfig()
authenticator = StrengthDrivenAuthenticator(config)

# Train model
S, delta_geom, delta_sem = generate_training_data()
results = authenticator.train_phase_transition_model(S, delta_geom, delta_sem)

# Perform authentication
result = authenticator.authenticate(delta_geom, delta_sem)
print(f"Authentication decision: {result['decision']}")
print(f"Estimated strength: {result['S_hat']:.4f}")
print(f"Fusion weight: {result['omega']:.4f}")
```

### Security Protocol Usage

```python
from models.security_protocol import SecurityEnhancedAuthenticator

# Create security authenticator
security_auth = SecurityEnhancedAuthenticator()

# Generate challenge
challenge = security_auth.generate_challenge("device_001")

# Client response
session_key = security_auth.key_derivation.derive_session_key(
    security_auth.master_key, challenge, "device_001"
)
message = security_auth.message_binding.create_message(
    fused_residual, S_hat, S_c, challenge, 0
)
hmac = security_auth.message_binding.compute_hmac(message, session_key).hex()

# Server verification
result = security_auth.authenticate_with_challenge(
    challenge, fused_residual, S_hat, S_c, hmac, "device_001"
)
```

### Evaluation Metrics Usage

```python
from utils.enhanced_metrics import ComprehensiveEvaluator

# Create evaluator
evaluator = ComprehensiveEvaluator()

# Evaluate authentication performance
auth_metrics = evaluator.auth_metrics.compute_authentication_metrics(y_true, y_scores)

# Evaluate strength estimation
strength_metrics = evaluator.strength_metrics.compute_strength_metrics(S_true, S_pred)

# Generate report
report = evaluator.generate_evaluation_report(results)
```

## 🧪 Experimental Scripts

### 1. Phase Transition Authentication Experiments

```bash
python experiments/phase_transition_experiment.py
```

**Experiment Contents:**
- Phase transition law verification
- Strength estimation performance testing
- Authentication performance evaluation
- Security protocol testing
- Critical zone analysis

### 2. Method Comparison Experiments

```bash
python experiments/comparison_experiment.py
```

**Comparison Methods:**
- Proposed method (phase-aware fusion)
- Fixed fusion methods
- Pure semantic feature methods
- Pure geometric feature methods

### 3. Feature Demonstration

```bash
python demo_phase_transition.py
```

**Demonstration Contents:**
- Phase transition detection visualization
- Adaptive fusion display
- Security protocol demonstration
- Evaluation metrics display

## 📈 Visualization Results

After running experiments, the following visualization results will be generated:

- `demo_phase_transition.png`: Phase transition detection results
- `demo_adaptive_fusion.png`: Adaptive fusion weight changes
- `authentication_comparison.png`: Authentication performance comparison
- `critical_zone_comparison.png`: Critical zone performance comparison
- `strength_estimation_comparison.png`: Strength estimation performance comparison

## 🔧 Configuration Options

### Phase Transition Authentication Configuration

```python
from models.phase_transition_auth import PhaseTransitionConfig

config = PhaseTransitionConfig(
    S_c_initial=0.52,           # Initial critical point
    kendall_tau_threshold=0.8,  # Monotonicity threshold
    bootstrap_samples=1000,     # Bootstrap samples
    learning_rate=1e-3,         # Learning rate
    num_epochs=100             # Training epochs
)
```

### Security Protocol Configuration

```python
from models.security_protocol import SecurityConfig

security_config = SecurityConfig(
    challenge_size=128,          # Challenge size
    expiration_time=300,        # Expiration time
    max_attempts=5,             # Maximum attempts
    lockout_duration=60         # Lockout duration
)
```

## 📋 Dependencies

Main dependencies:

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
cryptography>=3.4.8
seaborn>=0.11.0
pandas>=1.3.0
```

For the complete dependency list, please refer to `requirements.txt`.

---

**Note**: This project is implemented based on academic research and is for learning and research purposes only. Please conduct thorough security assessment and testing before using in production environments.
