
Méthodologie Algorithmique
==========================

Cette section détaille l'implémentation de l'architecture **Pessimistic Command UDRL**.

1. Pipeline d'Entraînement Standard
-----------------------------------

L'entraînement se déroule en deux étapes indépendantes, ce qui garantit la stabilité du processus (pas de boucle de rétroaction instable comme en Actor-Critic).

**Étape A : Entraînement de l'Agent UDRL**

L'agent est un réseau de neurones :math:`\pi_\phi(a | s, h, r)` entrainé par apprentissage supervisé.

.. code-block:: python

	# Pseudo-code PyTorch
	loss = nn.MSELoss(pred_action, target_action)  # Cas continu
	loss = nn.CrossEntropy(pred_logits, target_action_idx) # Cas discret

**Hyperparamètres Clés (Phase 2)** :
*   Optimiseur : AdamW (:math:`lr=3e^{-4}`, :math:`\beta=(0.9, 0.999)`)
*   Batch Size : 256
*   Fonction d'activation : ReLU (couches cachées), Tanh (sortie continue)

**Étape B : Entraînement du Pessimiste (Oracle)**

Le pessimiste est un régresseur quantile :math:`f_\psi(s) \rightarrow \hat{r}_\tau`.
Il ne prend *pas* la commande en entrée, uniquement l'état. Il prédit "ce qui est possible depuis l'état :math:`s`".

2. Algorithme de Projection à l'Inférence
-----------------------------------------

Lors de l'évaluation, l'agent ne reçoit jamais la commande brute de l'utilisateur. Elle passe par un filtre de sécurité.

.. math::
    \text{SafePolicy}(s_t, r_{target}, h_{target}) = \pi_\phi \left( s_t, \min(r_{target}, Q_\tau(s_t)), h_{target} \right)

**Détail du calcul du Pessimism Gap** :
Le *Pessimism Gap* quantifie l'intervention du système de sécurité.

.. math::
    \Delta_{gap} = \max(0, r_{target} - Q_\tau(s_t))

*   Si :math:`\Delta_{gap} > 0`, la commande était irréaliste/dangereuse.
*   Si :math:`\Delta_{gap} = 0`, la commande est jugée sûre.

3. Gestion des Données (Dataset)
--------------------------------

La qualité du dataset conditionne la performance. Nous utilisons le format standard HDF5.

**Structure des données** :
*   `observations`: :math:`(N, S_{dim})`
*   `actions`: :math:`(N, A_{dim})`
*   `rewards`: :math:`(N, 1)`
*   `terminals`: :math:`(N, 1)`

L'horizon :math:`h` et le retour cumulé :math:`r` sont calculés à la volée lors du chargement (Lazy Loading) pour économiser la mémoire et permettre du *Hindsight Relabeling* dynamique si nécessaire.

4. Architecture Réseau
----------------------

.. code-block:: text

    State Input (8) ───► [FC 256] ──► [ReLU] ──┐
                                               ▼
    Command Input (2) ─► [FC 256] ──► [ReLU] ──► [Concat] ──► [FC 256] ──► [Action Output]

71: Cette architecture *Late Fusion* permet au réseau de traiter les caractéristiques de l'état indépendamment de la commande avant de les combiner pour la décision.

5. Algorithme Complet (Pseudo-Code)
-----------------------------------

.. code-block:: python

    def Run_Episode(agent, pessimist, env, target_return=200):
        state = env.reset()
        horizon = env.max_steps
        
        for t in range(env.max_steps):
            # 1. Pessimistic Projection
            # Le pessimiste prédit le retour réalisable (quantile 0.1)
            feasible_return = pessimist.predict_quantile(state, q=0.1)
            
            # 2. Safety Check (Min Operator)
            # On ne demande jamais plus que ce qui est réalisable
            safe_command = min(target_return, feasible_return)
            
            # 3. Action Selection
            # L'agent exécute la commande safe
            action = agent.act(state, horizon, safe_command)
            
            # 4. Environment Step
            next_state, reward, done, _ = env.step(action)
            
            # 5. Update State
            state = next_state
            horizon -= 1
            target_return -= reward # La commande restante diminue
            
            if done: break

6. Au-delà des Quantiles : Modèles Génératifs (Phase 3)
-------------------------------------------------------

Pourquoi passer de la régression quantile aux modèles génératifs ?

**Limites de la Régression Quantile**

*   **Hypothèse Unimodale** : QR suppose implicitement que la distribution des retours est "simple" (un seul pic principal).
*   **Manque de Précision** : Elle ne donne qu'une borne (un chiffre), pas la forme de la distribution.

**A. Conditional VAE (Variational Autoencoder)**
Le CVAE apprend à modéliser la densité complète :math:`p(r, h | s)`.

*   **Idée** : Projeter la commande sur le *manifold* latent des commandes possibles.
*   **Avantage** : Permet de trouver la commande "la plus proche" dans l'espace latent continu, assurant une transition douce.

**B. Diffusion Models (Denoising Diffusion)**
La méthode de l'état de l'art pour la génération.

*   **Idée** : Apprendre le gradient de la log-densité (Score Matching) des commandes :math:`\nabla_{(r,h)} \log p(r, h | s)`.
*   **Processus** : On part d'une commande bruitée (random) et on la "débruite" itérativement conditionnellement à l'état :math:`s`.
*   **Résultat** : Capacité à modéliser des distributions multimodales complexes (e.g., soit on gagne petit à coup sûr, soit on gagne gros avec risque, mais rien entre les deux).

.. seealso::
    Pour les résultats expérimentaux de ces méthodes, voir la section "Phase 3" dans :ref:`results`.
