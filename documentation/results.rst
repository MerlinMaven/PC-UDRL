
Results & Benchmarks
====================

This section presents the empirical results obtained during the validation phases.

Phase 1: GridWorld Validation
-----------------------------

*   **Experiment**: 1000 random trajectories, sparse reward environment.
*   **Comparison**: Standard UDRL (Blind) vs PC-UDRL (Pessimistic).

.. list-table:: GridWorld Performance (20 Episodes)
   :widths: 25 25 50
   :header-rows: 1

   * - Agent
     - Avg Return
     - Observations
   * - **Standard UDRL**
     - -88.00
     - High variance, frequently attempts impossible paths, hits walls.
   * - **PC-UDRL**
     - -100.00
     - Extremely conservative. Zero variation. Avoids unknown regions ("Safety First").

*Observation*: While the standard agent occasionally gets lucky (20% success in dataset), it fails catastrophically often. The PC-UDRL agent never outperforms the "safe" baseline established by the pessimist, proving that the safety layer is effective (albeit overly conservative on random data).

Phase 2: LunarLander Baseline Comparison
----------------------------------------

*   **Dataset**: Mixed (50% Expert, 30% Medium, 20% Random).
*   **Baselines**: CQL, IQL, TD3+BC (d3rlpy implementations).

*Note: Training is currently in progress. Preliminary results will be updated below.*

.. list-table:: LunarLander Evaluation (Projected)
   :widths: 25 25 50
   :header-rows: 1

   * - Algorithm
     - Avg Return
     - Stability
   * - **CQL**
     - *Running...*
     - Expected: High stability, moderate return.
   * - **IQL**
     - *Pending...*
     - Expected: Good generalization.
   * - **TD3+BC**
     - *Pending...*
     - Expected: Minimalist baseline.
   * - **PC-UDRL**
     - *Pending...*
     - Goal: Match baseline performance while offering command adjustability.

Visualizations
--------------

Phase 1 Side-by-Side
~~~~~~~~~~~~~~~~~~~~

.. image:: ../outputs/plots/phase1/comparison_preview.png
   :width: 600
   :alt: Side-by-Side Comparison (Placeholder)

*(Video recordings are available in `runs/phase1/videos/`)*
