Usage Guide
===========

This guide details how to run experiments across the different phases of the project.

Phase 1: GridWorld Validation (POC)
-----------------------------------

The first phase validates the *Safe Command Following* concept in a simple discrete environment.

1.  **Generate Random Dataset**
    
    .. code-block:: bash

        python scripts/generate_dataset.py --phase 1

    This creates `data/dataset.h5` with 1000 random trajectories.

2.  **Train PC-UDRL Agent**

    .. code-block:: bash

        python scripts/train_udrl.py --phase 1

    This trains the UDRL agent and the Quantile Pessimist. Artifacts are saved in `runs/phase1/`.

3.  **Live Visualization**

    .. code-block:: bash

        # Visual Comparison (Standard vs Pessimist)
        python scripts/evaluate.py --phase 1 --live_compare

    This opens a window showing two agents side-by-side: one blindly following the target (+10), the other capped by the pessimist (-10).

Phase 2: LunarLander Continuous
-------------------------------

The second phase applies PC-UDRL to a continuous control problem with a mixed-quality dataset.

1.  **Generate Mixed Dataset**

    .. code-block:: bash

        # Trains a PPO expert and collects Expert/Medium/Random data
        python scripts/generate_lunar_dataset.py

2.  **Train Baselines (d3rlpy)**

    Train state-of-the-art offline RL algorithms for comparison.

    .. code-block:: bash

        python scripts/train_baselines.py --algo cql --epochs 30
        python scripts/train_baselines.py --algo iql --epochs 30
        python scripts/train_baselines.py --algo td3plusbc --epochs 30

3.  **Train PC-UDRL**

    .. code-block:: bash

        python scripts/train_udrl.py --phase 2

4.  **Comparative Evaluation**

    .. code-block:: bash

        python scripts/evaluate.py --phase 2 --compare

    This will evaluate PC-UDRL against the trained baselines and print a summary table.

Configuration
-------------

All hyperparameters are centralized in `config.py`. You can modify:

*   `env_id`: Environment name (e.g., `LunarLanderContinuous-v3`)
*   `max_steps`: Episode duration
*   `pessimist_quantile`: Safety level (default 0.9)
*   `hidden_dim`: Network size
