
Safety & Robustness
===================

Structural Safety by Design
---------------------------

In traditional Offline RL (e.g., CQL, IQL), safety is often a byproduct of regularization: the algorithm is penalized for deviating from the behavioral policy. **PC-UDRL** takes a different approach: **Safety by Command Projection**.

Instead of hoping the policy learns to be safe, we explicitly **constrain the command** given to the policy.

The Pessimism Mechanism
~~~~~~~~~~~~~~~~~~~~~~~

The core safety component is the **Pessimist** (or Oracle), denoted as :math:`\mathcal{P}`.
For a given state :math:`s_t`, the pessimist estimates the distribution of feasible future returns:

.. math::
    Q_\tau(s_t) = \text{Quantile}_\tau [ R_{future} | s_t ]

If a user requests a target return :math:`r_{cmd} = +100`, but the pessimist determines that the 10th percentile of feasible returns from :math:`s_t` is only :math:`-50`, the command is **clamped**:

.. math::
    r_{safe} = \min(r_{cmd}, Q_\tau(s_t))

This acts as a "safety shield," preventing the agent from pursuing unrealistic goals that would likely lead to out-of-distribution (OOD) states and catastrophic failure.

The "Obedient Suicide" Phenomenon
---------------------------------

During Phase 1 (GridWorld), we observed a counter-intuitive behavior:
The agent, when capped to a low return (e.g., -10), would sometimes **intentionally** move towards a trap.

*   **Explanation**: The UDRL agent is trained to *satisfy* the command, not maximize reward. If the only robustly known path leads to a -10 outcome (the trap), and the command is clamped to -10, the agent interprets hitting the trap as "mission accomplished."
*   **Implication**: This validates that the agent is controllable. In a safety context, it is often better to accept a minor known loss (controlled landing, emergency stop) than to gamble on a high reward that carries the risk of total system loss.

Comparison with Action-Space Constraints
----------------------------------------

+---------------------+-------------------------------+----------------------------------+
| Feature             | Action-Space (CQL, TD3+BC)    | Command-Space (PC-UDRL)          |
+=====================+===============================+==================================+
| **Locus of Control**| Policy Output (Action)        | Policy Input (Command)           |
+---------------------+-------------------------------+----------------------------------+
| **Mechanism**       | Regularization term in Loss   | Pre-processing of Input          |
+---------------------+-------------------------------+----------------------------------+
| **Flexibility**     | Hard to tune post-training    | Tunable at inference (Quantile)  |
+---------------------+-------------------------------+----------------------------------+
| **Interpretability**| Low (Black box policy)        | High (Explicit command limits)   |
+---------------------+-------------------------------+----------------------------------+

References
----------

*   **Safety in RL**: Amodei, D., et al. (2016). "Concrete Problems in AI Safety." *arXiv preprint arXiv:1606.06565*.
*   **Offline RL Surveys**: Levine, S., et al. (2020). "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems." *arXiv preprint arXiv:2005.01643*.
