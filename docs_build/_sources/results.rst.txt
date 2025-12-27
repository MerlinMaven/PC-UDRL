
Results & Benchmarks
====================

This section presents the empirical results obtained during the validation phases.

Phase 1: GridWorld Validation (Proof of Concept)
------------------------------------------------

**Objectif** : Valider que l'agent UDRL apprend à naviguer dans un environnement simple avec des récompenses éparses, malgré un dataset purement aléatoire.

.. image:: ../assets/eval_mean_return.png
   :width: 800
   :alt: Performance GridWorld (Random Data)
   :align: center

*   **Experiment**: 1000 random trajectories.
*   **Résultat** : L'agent apprend à atteindre le but (Reward +1), mais reste incertain.
*   **Observation** : Le graphique ci-dessus montre une convergence lente et bruyante. La couverture d'états (:math:`State Coverage`) est faible, ce qui limite la généralisation.

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

Analyse Critique des Données (Dataset Analysis)
-----------------------------------------------

La différence de performance entre la Phase 1 et la Phase 2 s'explique par la qualité des données.

.. topic:: Leçon : Garbage In, Garbage Out

    Un algorithme d'Offline RL ne peut pas inventer des compétences absentes des données.

    **1. Dataset Aléatoire (GridWorld)**
    *   **Composition** : 99% d'échecs, 1% de succès fortuits.
    *   **Problème** : L'agent apprend principalement "ce qui ne marche pas". Le manifold des possibles est dominé par l'échec.
    *   **Conséquence** : Le pessimiste apprend que *tout est dangereux*, paralysant l'agent.

    **2. Dataset Expert/Medium (LunarLander)**
    *   **Composition** : Trajectoires d'un agent partiellement entraîné (Medium) ou optimal (Expert).
    *   **Avantage** : Le manifold contient des stratégies de succès. Le pessimiste peut distinguer "difficile mais faisable" de "impossible".

Phase 2-4: LunarLander Continuous Benchmark
-------------------------------------------

Nous passons à un environnement complexe (continu, physique) avec un dataset de qualité mixte.

.. image:: ../assets/compare_p2_p3.png
   :width: 800
   :alt: Comportement des Agents (Phase 2 vs Phase 3)
   :align: center

**Comparaison des Approches**

Nous avons testé trois paradigmes de modélisation du risque (Pessimisme) :

1.  **Phase 2 (Quantile)** : Statistiques scalaires robustes.
2.  **Phase 3 (CVAE)** : Modèle génératif probabiliste (Voronoi-like).
3.  **Phase 4 (Diffusion)** : Modèle basé sur le score (Gradient-based).

**Résultats Quantitatifs (30 Époques)**

.. list-table:: Comparaison Finale
   :widths: 20 20 20 40
   :header-rows: 1

   * - Méthode
     - Avg Return
     - Pessimism Gap
     - Comportement Observé
   * - **Quantile (P2)**
     - ~ -200
     - Élevé (Static)
     - **Paralysie**. L'agent est trop prudent, il atterrit prématurément ou refuse d'avancer.
   * - **CVAE (P3)**
     - ~ -150
     - ~ 0.0 (Null)
     - **Crash**. Le modèle hallucine que +195 est possible. L'agent tente l'impossible et s'écrase.
   * - **Diffusion (P4)**
     - **~ -110**
     - **Adaptatif**
     - **Sécurité Dynamique**. Le modèle réduit la commande juste assez pour éviter le crash sans bloquer l'agent. Meilleur compromis.

.. image:: ../assets/p4_returns.png
   :width: 800
   :alt: Courbes de Performance Comparées Phase 4
   :align: center

**Analyse Visuelle (Video)**

La vidéo ci-dessous illustre la supériorité de l'approche Diffusion (Phase 4), qui offre un atterrissage contrôlé contrairement au freinage brusque du Quantile.

*(Voir la vidéo `compare_all_phases.mp4` incluse dans les assets)*
