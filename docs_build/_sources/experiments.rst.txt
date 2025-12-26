
Expérimentations et Protocoles
==============================

Cette section décrit rigoureusement les protocoles expérimentaux pour assurer la reproductibilité des résultats.

Environnements de Test
----------------------

1. GridWorld (Environnement Discret)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Environnement de navigation simple pour la preuve de concept.

*   **Espace d'état** : Coordonnées (x, y) normalisées :math:`[0, 1]^2`.
*   **Espace d'action** : Discret {Haut, Bas, Gauche, Droite}.
*   **Dynamique** : Déterministe sur les transitions.
*   **Récompenses** :

    *   Pas de temps : -1
    *   Mur : -0.5
    *   Piège : -10 (Terminal)
    *   Objectif : +10 (Terminal)

2. LunarLanderContinuous-v2 (Environnement Continu)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problème de contrôle optimal via le moteur physique Box2D.

*   **Espace d'état** : :math:`\mathbb{R}^8` (Position, Vitesse, Angle, Vitesse Angulaire, Contact Sol).
*   **Espace d'action** : :math:`[-1, 1]^2` (Moteur Principal, Moteurs Latéraux).
*   **Critère de succès** : Atterrissage doux entre les drapeaux (Reward > 200).

Protocole d'Entraînement
------------------------

Hyperparamètres
~~~~~~~~~~~~~~~

La table ci-dessous recense les hyperparamètres fixés pour toutes les expériences de la Phase 2.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Paramètre
     - Valeur
   * - **Optimizer**
     - Adam
   * - **Learning Rate**
     - :math:`3 \times 10^{-4}`
   * - **Batch Size**
     - 256
   * - **Network Architecture**
     - MLP (2 hidden layers, 256 units)
   * - **Activation**
     - ReLU
   * - **Horizon Max (Training)**
     - 200 (GridWorld), 1000 (LunarLander)
   * - **Pessimist Quantile** (:math:`\tau`)
     - 0.9 (GridWorld), 0.7 (LunarLander)
   * - **Training Epochs**
     - 30 (Checkpointing tous les 10)

Datasets
~~~~~~~~

Pour assurer la robustesse *Offline*, nous générons des datasets synthétiques :

*   **Random (GridWorld)** : Politique uniforme. Couverture faible de l'objectif (20%).
*   **Mixed (LunarLander)** : Mélange conçu pour le benchmark D4RL :
    *   50% Trajectoires Expertes (PPO).
    *   30% Trajectoires Moyennes (Expert bruité, :math:`\epsilon=0.3`).
    *   20% Trajectoires Aléatoires.

Métriques d'Évaluation
----------------------

Nous évaluons les agents selon trois axes :

1.  **Performance Pure** : Retour moyen normalisé sur 20 épisodes.
2.  **Sécurité** : Nombre d'états terminaux "catastrophiques" visités.
3.  **Respect de la Commande** : Erreur quadratique entre le retour demandé et obtenu (pour les commandes réalisables).
