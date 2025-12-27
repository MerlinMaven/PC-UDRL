
.. _methodology:

Méthodologie Algorithmique
==========================

Cette section détaille l'implémentation technique de **PC-UDRL**. Elle complète le cadre théorique en se concentrant sur le "Comment" (Architecture, Hyperparamètres, Code) plutôt que le "Pourquoi".

1. Architecture Système
-----------------------

Le système repose sur deux réseaux de neurones distincts qui interagissent uniquement lors de l'inférence.

.. code-block:: text

    ┌───────────────────────────────────┐      ┌───────────────────────────────┐
    │           AGENT (Pilote)          │      │     PESSIMISTE (Garde-Fou)    │
    │  π(a | s, r_target, h_target)     │      │      P(valid_command | s)     │
    │           (Late Fusion)           │      │      (Quantile / CVAE / Diff) │
    └────────────────┬──────────────────┘      └──────────────┬────────────────┘
                     │                                        │
                     │ Action                                 │ Safety Projection
                     ▼                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                          ENVIRONNEMENT (Gym)                           │
    └────────────────────────────────────────────────────────────────────────┘

.. image:: ../assets/system_architecture.png
   :width: 800
   :alt: PC-UDRL System Architecture
   :align: center
   :class: no-scaled-link

2. Pipeline d'Entraînement
--------------------------

L'entraînement est **séquentiel et décorrélé**, garantissant une grande stabilité.

Phase A : Entraînement de l'Agent UDRL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L'agent apprend par **Apprentissage Supervisé** à imiter les actions qui ont mené à des retours donnés.

*   **Entrée** : État :math:`s` (8 dim), Commande :math:`(r, h)` (2 dim).
*   **Architecture** : MLP "Late Fusion".
    *   Les inputs sont traités séparément avant d'être concaténés.
    *   *Raison* : Évite que la commande (souvent de grande magnitude) n'écrase les features subtiles de l'état au début de l'entraînement.

    .. note:: **Early Fusion vs Late Fusion**
        
        Une "Early Fusion" (concaténation immédiate) risque de laisser le réseau ignorer l'état au profit de la commande qui a une plus grande variance. La **Late Fusion** force le réseau à extraire des caractéristiques de l'état avant de considérer l'objectif.

.. code-block:: python

    class UDRLAgent(nn.Module):
        def __init__(self, state_dim, action_dim):
            self.state_net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
            self.cmd_net = nn.Sequential(nn.Linear(2, 256), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, action_dim), nn.Tanh()
            )

        def forward(self, state, command):
            s_emb = self.state_net(state)
            c_emb = self.cmd_net(command)
            return self.head(torch.cat([s_emb, c_emb], dim=1))

Phase B : Entraînement du Pessimiste
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selon le niveau de sophistication (voir Théorie), l'architecture change.

**Option 1 : Régresseur Quantile (MLP)**
*   **Objectif** : Prédire les quantiles :math:`\tau` de la distribution des retours.
*   **Output** : Vecteur de taille :math:`N_{quantiles}`.
*   **Loss** : Pinball Loss.

**Option 2 : CVAE (Conditional VAE)**
*   **Objectif** : Apprendre la distribution jointe :math:`p(r, h | s)`.
*   **Architecture** :
    *   *Encoder* : :math:`(s, r, h) \rightarrow (\mu, \sigma) \rightarrow z`.
    *   *Decoder* : :math:`(s, z) \rightarrow (\hat{r}, \hat{h})`.
*   **Loss** : ELBO (Reconstruction + KL Divergence).

**Option 3 : Diffusion Model (U-Net 1D)**
*   **Objectif** : Apprendre le gradient de score :math:`\nabla_x \log p(x|s)`.
*   **Architecture** :
    *   *Conditioning* : L'état :math:`s` est injecté à chaque couche résiduelle (FiLM ou Concat).
    *   *Time Embedding* : Sinusoidal (Transformer-style).

3. Algorithme d'Inférence (La Boucle de Contrôle)
-------------------------------------------------

L'innovation principale réside dans la modification dynamique de la commande utilisateur **avant** qu'elle n'atteigne l'agent.

.. code-block:: python

    def run_inference(env, agent, pessimist, target_return, target_horizon):
        state, _ = env.reset()
        
        # Commande initiale (souvent irréaliste, ex: +1000)
        cmd = np.array([target_return, target_horizon])

        while not done:
            # 1. PROJECTION DE SÉCURITÉ
            # Le pessimiste "corrige" la commande en fonction de l'état actuel
            if use_diffusion:
                safe_cmd = pessimist.guided_projection(cmd, state, steps=10)
            else: # Quantile
                max_return = pessimist.predict_quantile(state, q=0.9)
                # On clamp (limite) principalement le retour attendu
                # L'horizon est une contrainte physique subie, pas un choix
                safe_cmd = cmd.copy()
                safe_cmd[0] = min(cmd[0], max_return)

            # 2. PRISE DE DÉCISION
            # L'agent obéit à la commande sécurisée
            action = agent(state, safe_cmd)

            # 3. INTERACTION
            next_state, reward, done, _, _ = env.step(action)

            # 4. MISE À JOUR DE LA COMMANDE (Hindsight)
            # On soustrait ce qu'on vient de gagner pour obtenir le "restant à faire"
            cmd[0] -= reward  # Return remaining
            cmd[1] -= 1       # Horizon remaining
            state = next_state

.. note::
    
    C'est cette boucle de **Replanning Constant** qui rend l'agent robuste. Même si la projection initiale est imparfaite, elle est réévaluée à chaque pas de temps (:math:`50 Hz`).

.. note::  **Stratégie "Greedy Maximization"**

    En fixant :math:`r_{target}` à une valeur impossible (ex: +1000), nous forçons l'agent à chercher la **meilleure performance possible**.
    
    *   **Sans Pessimiste** : L'agent hallucine et crashe (car +1000 est OOD).
    *   **Avec Pessimiste** : L'agent reçoit la commande "faisable" la plus haute (ex: +230).
    
    Ainsi, PC-UDRL transforme un agent "Goal-Conditioned" en un optimiseur glouton robuste.

4. Détails d'Implémentation & Hyperparamètres
---------------------------------------------

Pour assurer la reproductibilité des résultats présentés dans le rapport scientifique.

.. list-table:: Hyperparamètres Globaux
   :widths: 40 60
   :header-rows: 1

   * - Paramètre
     - Valeur
   * - **Optimiseur**
     - AdamW
   * - **Learning Rate**
     - :math:`3 \times 10^{-4}`
   * - **Batch Size**
     - 256
   * - **Dataset**
     - LunarLander-Medium (D4RL format)
   * - **Normalisation**
     - États: Standard (Mean=0, Std=1)
     - Retours: Min-Max Scaling [0, 1] (Crucial pour la Diffusion)

.. list-table:: Spécifique : Régression Quantile (Phase 2)
   :widths: 40 60
   :header-rows: 1

   * - Paramètre
     - Valeur
   * - **Nb Quantiles**
     - 51 (:math:`\tau \in \{0.02, ..., 0.98\}`)
   * - **Hidden Layers**
     - [256, 256] (MLP Standard)
   * - **Loss**
     - Pinball Loss (Huberisée pour stabilité)

.. list-table:: Spécifique : CVAE (Phase 3)
   :widths: 40 60
   :header-rows: 1

   * - Paramètre
     - Valeur
   * - **Latent Dim**
     - 4
   * - **Hidden Layers**
     - [128, 64] (Encoder & Decoder)
   * - **Beta (KL Weight)**
     - 1.0 (Standard VAE) ou >1 pour Disentanglement

.. list-table:: Spécifique : Modèle de Diffusion (Phase 4)
   :widths: 40 60
   :header-rows: 1

   * - Paramètre
     - Valeur
   * - **Diffusion Steps (Training)**
     - 1000
   * - **Diffusion Steps (Inférence)**
     - 10 à 50 (Trade-off Latence/Précision)
   * - **Noise Schedule**
     - Linear (:math:`\beta_{start}=1e^{-4}, \beta_{end}=0.02`)
   * - **Guidance**
     - Strictement conditionné par l'état (Pas de CFG)

5. Gestion des Données
----------------------

Le chargement des données utilise une stratégie **Lazy Loading** optimisée pour la RAM.

*   **Format** : HDF5 compressé (LZF).
*   **Hindsight Relabeling** : Calculé à la volée. Au lieu de stocker :math:`(r, h)` pour chaque transition, on stocke les trajectoires complètes et on calcule le retour restant :math:`\sum_{t'=t}^T r_{t'}` lors du sampling du batch. Cela permet d'augmenter artificiellement la taille du dataset en échantillonnant différents horizons pour un même état.

.. image:: ../assets/hindsight_relabeling.png
   :width: 600
   :alt: Hindsight Relabeling Mechanism
   :align: center
   :class: no-scaled-link


