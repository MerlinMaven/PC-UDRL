
FAQ & Deep Dive
===============

Theoretical Questions
---------------------

**Q: Why not just use `r_min` in the reward function?**

A: Reward shaping modifies the learning objective, often leading to "reward hacking" behaviors. By keeping the reward function pure and constraining the **command** (the goal), we separate the *what* (the task) from the *how* (the safety constraints). This makes the system more modular and interpretable.

**Q: How does this differ from Model-Based Offline RL (MOPO, MOREL)?**

A: Model-Based methods penalize the value function based on model uncertainty. PC-UDRL is model-free in the dynamics sense; it models the **return distribution**. This is computationally lighter (no dynamics model rollout required at inference) and avoids compounding model errors for long horizons.

**Q: Can this handle multi-modal distributions?**

A: The current **Quantile Regressor** assumes a parametric distribution (or set of quantiles) which captures aleatoric uncertainty well. For truly multi-modal command manifolds (e.g., distinct strategies to reach the same goal), we plan to introduce **CVAEs** or **Diffusion Models** in Phase 3.

Implementation Details
----------------------

**Q: Why use `d3rlpy` for baselines?**

A: `d3rlpy` is a highly optimized, industry-standard library for Offline RL. Using it ensures our baselines are "strong baselines," preventing the common pitfall of comparing a tuned novel method against a poorly implemented standard algorithm.

**Q: What is the "Horizon" input in UDRL?**

A: The horizon :math:`h` represents the "time-to-live" or steps remaining to achieve the goal. It is crucial for Markovian property in UDRL. In our implementation, :math:`h` is decremented at each step.

**Q: How is the dataset generated for Phase 2?**

A: We train a PPO expert online on `LunarLanderContinuous-v2`. We then record its trajectories (Expert), add noise to its actions (Medium), and mix in purely random trajectories (Random). This "Mixed" dataset is standard in D4RL benchmarks to evaluate an algorithm's ability to stitch together suboptimal trajectory segments.
