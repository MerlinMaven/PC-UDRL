# Safe Upside-Down Reinforcement Learning (PC-UDRL)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/Docs-ReadTheDocs-green.svg)](docs_build/index.html)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PC-UDRL** (Pessimistic Command UDRL) is a novel Offline Reinforcement Learning framework that shifts the safety paradigm from *action-space regularization* (like CQL, IQL) to **command-space projection**.

> **Core Idea**: Instead of asking an agent to "be careful" (policy constraint), we simply never ask it to do something unsafe (command constraint).

---

## 🚀 Why PC-UDRL?

Traditional Offline RL methods (CQL, TD3+BC) often suffer from:
*   **Conservatism**: Excessive regularization leads to poor performance.
*   **Opaqueness**: Hard to interpret why an agent refuses to act.
*   **Instability**: Value function estimation on OOD (Out-of-Distribution) data is prone to collapse.

**Our Solution**:
1.  **UDRL Backbone**: Transforming RL into Supervised Learning (Conditioned on Return & Horizon).
2.  **Pessimistic Oracle**: A generative model (Quantile, CVAE, or Diffusion) that learns the "Feasible command manifold".
3.  **Safety By Projection**: If a user asks for an unrealistic return (e.g., +200 safety score), the Oracle projects it down to the highest *safe* value (e.g., +150).

---

## 🏗️ Architecture

The framework operates in **2 distinct phases**:

1.  **Pessimist Training**: Learning :math:`Q_\tau(s)` or :math:`p(r, h | s)` from the offline dataset.
2.  **Agent Training**: Learning :math:`\pi(a | s, r, h)` via supervised learning.

During inference, they work together:

```python
# The Safety Loop
feasible_return = pessimist.predict_quantile(state, q=0.1)
safe_command = min(target_return, feasible_return)  # The Safety Shield
action = agent.act(state, horizon, safe_command)
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/PC-UDRL.git
cd PC-UDRL

# Install dependencies (Virtual Environment recommended)
pip install -r requirements.txt
```

**Requirements**: `torch`, `d3rlpy`, `gymnasium[box2d]`, `numpy`.

---

## ⚡ Quickstart

### Phase 1: GridWorld Proof of Concept
Validate the "Obedient Suicide" phenomenon on a simple discrete environment.

```bash
# Generate random dataset
python main.py --phase 1 --mode generate

# Train Agent & Pessimist
python main.py --phase 1 --mode train

# Visualize results
python main.py --phase 1 --mode eval
```

### Phase 2: LunarLander Continuous
Train on full physics environment with mixed synthetic data.

```bash
# 1. Train Baselines (CQL, IQL, TD3+BC)
python scripts/train_baselines.py --algo cql --epochs 30
python scripts/train_baselines.py --algo iql --epochs 30

# 2. Train PC-UDRL
python main.py --phase 2 --mode train

# 3. Compare Results
python scripts/plot_results.py --phase 2
```

---

## 📚 Documentation

Full technical documentation is available in the `documentation/` folder (compiled with Sphinx).

*   **[Theory](documentation/theory.rst)**: Aleatoric vs Epistemic Uncertainty, Quantile Regression.
*   **[Methodology](documentation/methodology.rst)**: Architecture details, Pseudo-code.
*   **[Experiments](documentation/experiments.rst)**: Hyperparameters, Dataset composition.

To build the docs locally:
```bash
sphinx-build -b html documentation docs_build
```

---

## 🗓️ Roadmap

- [x] **Phase 1**: Discrete GridWorld Validation (Quantile).
- [x] **Phase 2**: Continuous LunarLander (Quantile vs Baselines).
- [ ] **Phase 3**: Advanced Generative Models (CVAE, Diffusion) for Multi-modal safety.
- [ ] **Phase 4**: D4RL Benchmarks (MuJoCo).

---

## 🤝 Contributing

Contributions are welcome! Please check the `documentation/contributing.rst` guide (coming soon).

---

**Author**: [Serraji Wiam]
**Year**: 2025
