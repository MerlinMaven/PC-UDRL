
Safety & Robustness
===================

.. admonition:: Core Philosophy
    :class: important

    In critical systems (robotics, finance, healthcare), **Safety** (avoiding catastrophic failure) often outweighs **Performance** (maximizing total reward). PC-UDRL is designed with this "Safety-First" mindset.

Structural Safety by Design
---------------------------

In traditional Offline RL (e.g., CQL, IQL), safety is often a byproduct of regularization: the algorithm is penalized for deviating from the behavioral policy. **PC-UDRL** takes a deterministic approach: **Safety by Command Projection**.

Instead of hoping the policy learns to be safe implicitly, we explicitly **constrain the command** given to the policy.

The Pessimism Mechanism as a Shield
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core safety component is the **Pessimist** (or Oracle), denoted as :math:`\mathcal{P}`. It acts as a dynamic firewall between the user and the agent.

For a given state :math:`s_t`, the pessimist estimates the distribution of feasible future returns:

.. math::
    Q_\tau(s_t) = \text{Quantile}_\tau [ R_{future} | s_t ]

If a user requests a target return :math:`r_{cmd} = +100` (Optimization Goal), but the pessimist determines that the 10th percentile of feasible returns is only :math:`-50` (Safety Boundary), the command is **clamped**:

.. math::
    r_{safe} = \min(r_{cmd}, Q_\tau(s_t))

.. image:: ../assets/gap_analysis.png
   :width: 600
   :alt: Safety Shield Mechanism (Gap Analysis)
   :align: center

This diagram illustrates the "Shielding" effect: the Agent never sees the impossible command. It only sees a command that lies within the **Safe Feasible Manifold**.

Robustness Metrics: CVaR
------------------------

To quantify robustness, we do not look at the *Average Return*, but at the **Conditional Value at Risk (CVaR)**.

.. math::
    \text{CVaR}_\alpha = \mathbb{E}[R \mid R \le \text{VaR}_\alpha]

* **Standard RL** optimizes for :math:`\mathbb{E}[R]` (Average case).
* **PC-UDRL** optimizes for :math:`\text{CVaR}_{0.1}` (Worst-case scenario).

By clamping commands with a low quantile (:math:`\tau=0.1`), we effectively maximize the CVaR, guaranteeing that even in the worst 10% of scenarios, the outcome is controlled.

The "Fail-Safe" Phenomenon
--------------------------

During Phase 1 (GridWorld), we observed a specific behavior where the agent, when capped to a low return (e.g., -10), would intentionally move towards a trap (-10 penalty) rather than wandering aimlessly (-100 penalty).

This is not a bug, but a **Fail-Safe Mechanism**:

1.  **Certainty vs. Ambiguity**: The agent chooses a *known* negative outcome over an *unknown* catastrophic one.
2.  **Controllability**: It proves the agent is strictly obedient. In an autonomous vehicle context, this equates to choosing a "controlled stop in a ditch" (minor damage) over "continuing at high speed on ice" (risk of total fatality).

.. note::
    It is better to accept a minor known loss (controlled landing, emergency stop) than to gamble on a high reward that carries the risk of Out-of-Distribution (OOD) failure.

Real-Time Observability: The Pessimism Gap
------------------------------------------

Typical RL agents are "black boxes": we don't know they are failing until they crash.
PC-UDRL introduces a novel supervisory signal: the **Pessimism Gap**.

.. math::
    \Delta_t = r_{cmd} - r_{safe}

*   **Green Zone** (:math:`\Delta \approx 0`): The user's request is realistic.
*   **Red Zone** (:math:`\Delta \gg 0`): The user is asking for the impossible (or the environment has degraded).

**Application:** This signal can be utilized as a **Runtime Anomaly Detector**, triggering alarms or handing control back to a human operator *before* the agent takes any action.

Comparison with State-of-the-Art
--------------------------------

+---------------------+-------------------------------+----------------------------------+
| Feature             | Action-Space (CQL, TD3+BC)    | Command-Space (PC-UDRL)          |
+=====================+===============================+==================================+
| **Locus of Control**| Policy Output (Action)        | Policy Input (Command)           |
+---------------------+-------------------------------+----------------------------------+
| **Mechanism**       | Regularization term in Loss   | Pre-processing of Input          |
+---------------------+-------------------------------+----------------------------------+
| **Flexibility**     | Hard to tune post-training    | **Tunable at inference**         |
+---------------------+-------------------------------+----------------------------------+
| **Interpretability**| Low (Black box policy)        | High (Explicit command limits)   |
+---------------------+-------------------------------+----------------------------------+

**Key Advantage:** The flexibility of PC-UDRL allows us to adjust the "Caution Level" (the quantile :math:`\tau`) in real-time without retraining the agent, which is impossible with CQL.

References
----------

*   **Safety in RL**: Amodei, D., et al. (2016). "Concrete Problems in AI Safety." `[arXiv:1606.06565] <https://arxiv.org/abs/1606.06565>`_
*   **Offline RL Surveys**: Levine, S., et al. (2020). "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems." `[arXiv:2005.01643] <https://arxiv.org/abs/2005.01643>`_
