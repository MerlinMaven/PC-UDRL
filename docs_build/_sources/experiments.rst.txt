
Protocole Expérimental
======================

Cette section détaille le banc d'essai utilisé pour valider empiriquement les hypothèses théoriques de PC-UDRL. Nous suivons un protocole rigoureux inspiré des standards **D4RL** (Datasets for Deep Data-Driven RL).

.. note:: **Robustesse Statistique**

    Pour éviter les conclusions hâtives basées sur des "lucky seeds", tous les résultats sont moyennés sur **5 graines aléatoires indépendantes** : `{0, 42, 123, 2024, 999}`.
    Les courbes de résultats affichent l'écart-type (:math:`\pm 1 \sigma`) sous forme de zone ombrée.

1. Environnements de Validation
-------------------------------

Nous testons notre approche sur deux niveaux de complexité : un environnement discret pour la preuve de concept, et un environnement continu dynamique pour la performance.

.. topic:: A. GridWorld (Preuve de Concept)

    Un labyrinthe stochastique conçu pour tester la capacité de l'agent à naviguer en sécurité malgré un dataset de mauvaise qualité.

    .. list-table::
       :widths: 60 40
       :class: borderless

       * - *   **Espace d'état** : Continu :math:`[0, 1]^2` (Coordonnées x, y).
           *   **Espace d'action** : Discret 4-directions {Haut, Bas, Gauche, Droite}.
           *   **Fonction de Récompense** :

               *   **Pas de temps** : :math:`-1` — *Incite à la rapidité*
               
               *   **Mur** : :math:`-0.5` — *Pénalité de collision*
               
               *   **Piège** : :math:`-10` — *Échec Terminal*
               
               *   **Objectif** : :math:`+10` — *Succès Terminal*

         - .. image:: ../assets/gridworld_init.png
              :width: 100%
              :alt: GridWorld Environment
              :align: center

.. topic:: B. LunarLander Continuous (Benchmark Physique)

    Problème de contrôle optimal via le moteur physique Box2D. L'objectif est d'atterrir en douceur entre deux drapeaux.

    .. list-table::
       :widths: 60 40
       :class: borderless

       * - *   **Espace d'état** : :math:`\mathbb{R}^8` (Pos, Vit, Angle, etc).
           *   **Espace d'action** : :math:`[-1, 1]^2` (Moteurs).
           *   **Fonction de Récompense** :

               *   **Distance/Vitesse** : :math:`\sim +140` — *Shaping reward optimal*
               
               *   **Moteur Principal** : :math:`-0.3` / frame — *Coût énergétique*
               
               *   **Atterrissage** : :math:`+100` — *Succès (Entre drapeaux)*
               
               *   **Crash / Sortie** : :math:`-100` — *Échec*

         - .. image:: ../assets/lunarlander_init.png
              :width: 100%
              :alt: LunarLander Environment
              :align: center

2. Données d'Entraînement (Offline)
-----------------------------------

La qualité du dataset est le facteur déterminant en Offline RL.

.. list-table:: Composition des Datasets
   :widths: 20 20 60
   :header-rows: 1

   * - Environnement
     - Volume
     - Composition Qualitative
   * - **GridWorld**
     - :math:`1 \times 10^3`
     - **100% Random**. (Exploration purement aléatoire). L'agent doit apprendre à atteindre un but qu'il n'a vu que par accident (1% des cas).
   * - **LunarLander**
     - :math:`1 \times 10^5`
     - **Mixed (D4RL Style)** :
       
       * 50% **Expert** (Politique PPO convergée)
       * 30% **Medium-Replay** (Replay buffer d'un agent en cours d'apprentissage)
       * 20% **Random** (Bruit pur pour la robustesse)

3. Configuration de l'Entraînement
----------------------------------

Les détails architecturaux sont dans la :ref:`methodology`. Voici les spécificités de la boucle d'expérience.

.. list-table:: Hyperparamètres de Run
   :header-rows: 1
   :widths: 40 60

   * - Paramètre
     - Valeur
   * - **Horizon Max (Training)**
     - 200 (GridWorld), 1000 (LunarLander)
   * - **Training Epochs**
     - 30 (Checkpointing tous les 10)

.. admonition:: Insight Expert : Le Réglage du Pessimisme
    :class: tip

    Le choix du quantile :math:`\tau` n'est pas arbitraire, il reflète notre **confiance en l'environnement** :

    *   **GridWorld** (:math:`\tau=0.9`) : Environnement déterministe. On peut être "optimiste" (quantile haut) car :math:`s \to s'` est prévisible.
    *   **LunarLander** (:math:`\tau=0.7`) : Environnement bruité et dynamique. Il faut être "prudent" (quantile plus bas) pour se prémunir contre les erreurs de modélisation.

4. Métriques d'Évaluation (La "Trinité")
-----------------------------------------

Nous évaluons la performance système selon trois axes complémentaires :

*   **Performance (D4RL Score)** : L'efficacité pure.
    
    .. math::
        Score = 100 \times \frac{R_{agent} - R_{random}}{R_{expert} - R_{random}}

*   **Robustesse (CVaR)** : La sécurité dans le pire des cas.
    
    .. math::
      	ext{CVaR}_{0.1} = \mathbb{E}[R \mid R \le \text{VaR}_{0.1}]

*   **Monitoring (Pessimism Gap)** : La capacité de détection d'anomalies.
    
    .. math::
        \Delta = r_{target} - r_{safe}
