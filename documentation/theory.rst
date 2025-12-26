
Cadre Théorique
===============

.. admonition:: Abstract
	:class: tip

	Ce chapitre développe le formalisme mathématique du projet **PC-UDRL**. Il unifie le paradigme Upside-Down Reinforcement Learning (UDRL) avec une approche de sécurité probabiliste fondée sur la distinction entre incertitude aléatorique et épistémique.

1. Safe Upside-Down RL
----------------------

Le concept central de ce projet repose sur l'inversion de la relation de contrôle en Reinforcement Learning (RL). Au lieu d'optimiser une fonction de valeur scalaire :math:`V^\pi(s)`, nous apprenons une distribution conditionnelle sur les trajectoires futures.

.. math::
	\pi_{\theta}(a_t | s_t, r^*_{t:T}, h^*_{t:T})

Où :math:`\theta` sont les paramètres du réseau de neurones, et le triplet :math:`(s_t, r^*, h^*)` constitue l'état étendu du système.

2. Modélisation de l'Incertitude
--------------------------------

La sécurité en Offline RL dépend de la capacité à quantifier ce qui est réalisable. Nous distinguons deux types d'incertitude dans la distribution des retours :math:`p(R | s)` :

**A. Incertitude Aléatorique (Intrinsèque)**
C'est le bruit inhérent à l'environnement stochastique ou à la politique comportementale (comportement multimodal dans le dataset).
*Exemple : Un expert peut choisir d'aller à gauche ou à droite avec une probabilité égale, menant à deux modes de retour distincts.*

**B. Incertitude Épistémique (Modèle)**
C'est l'incertitude due au manque de données dans certaines régions de l'espace d'états (OOD).
*Exemple : L'agent n'a jamais vu cet état, donc sa prédiction de retour est non fiable.*

3. Quantile Regression pour le Pessimisme
-----------------------------------------

Nous utilisons la **Quantile Regression** (QR) pour modéliser l'incertitude aléatorique de la distribution des retours réalisables.

L'objectif est d'apprendre les quantiles :math:`\tau` de la variable aléatoire :math:`R` (le retour cumulé futur) conditionnellement à l'état :math:`s`.

Soit :math:`q_\tau(s)` la valeur du :math:`\tau`-ième quantile. On minimise la *Quantile Huber Loss* :

.. math::
	\mathcal{L}_{QR} = \mathbb{E}_{(s, r) \sim \mathcal{D}} \left[ \rho_\tau(r - q_\tau(s)) \right]

où :math:`\rho_\tau(u) = u(\tau - \mathbb{I}_{u < 0})`.

**Application à la Sécurité :**
Le pessimisme est induit en choisissant un quantile bas (e.g., :math:`\tau=0.1`).
Cela signifie : *"Dans 90% des cas, le retour obtenu à partir de cet état sera supérieur à cette valeur."*

Si :math:`q_{0.1}(s) = -10`, alors demander un retour de :math:`+100` est irréaliste et dangereux. La commande est alors projetée :

.. math::
	r_{safe} = \min(r_{cmd}, q_{0.1}(s))

Cette méthode est robuste aux outliers et capture la distribution *support* des données d'entraînement.

4. Comparaison Théorique
------------------------

.. list-table::
	:header-rows: 1

	* - Approche
	  - Mécanisme de Sécurité
	  - Type d'Incertitude Géré
	* - **CQL (Conservative Q-Learning)**
	  - Pénalisation de la Value Function (min Q)
	  - Épistémique (via regularisation OOD)
	* - **MOPO (Model-Based)**
	  - Pénalisation basée sur l'erreur du modèle dynamique
	  - Épistémique (incertitude du modèle)
	* - **PC-UDRL (Ours)**
	  - Projection de la Commande (Input Constraint)
	  - Aléatorique (Distribution des retours)

5. Le Rôle Critique de la Donnée ("Garbage In, Garbage Out")
------------------------------------------------------------

En Offline RL, l'agent ne peut pas explorer pour corriger ses erreurs. Sa performance est **strictement bornée** par la qualité et la couverture du dataset :math:`\mathcal{D}`.

.. math::
    V^{\pi_{learned}}(s) \le \max_{\pi_\beta} V^{\pi_\beta}(s) + \epsilon_{generalization}

**Le Principe GIGO (Garbage In, Garbage Out) :**

*   **Dataset Aléatoire ("Garbage")** : Si :math:`\mathcal{D}` ne contient que du bruit (comme notre GridWorld Phase 1), l'agent ne peut apprendre aucune stratégie intelligente. Au mieux, il apprend à être *prudent* (notre phénomène "Obedient Suicide"), au pire il hallucine des valeurs élevées (Overestimation catastrophique).
*   **Dataset Expert** : L'agent apprend par imitation implicite des bonnes trajectoires.
*   **Dataset Mixte (Phase 2)** : C'est le cas le plus réaliste et intéressant. Le dataset contient des trajectoires optimales, sous-optimales et aléatoires. Le rôle du système est de *filtrer* le bruit.

**Pourquoi cela valide notre approche PC-UDRL ?**

Contrairement à l'Imitation Learning pur (Behavior Cloning) qui risquerait de copier aveuglément les erreurs du dataset mixte, UDRL conditionné par un retour élevé :math:`r^*` agit comme un filtre sélectif :
*"Donne-moi uniquement les actions qui ont historiquement mené aux 10% de meilleurs résultats observés."*

Si le dataset contient 90% d'erreurs et 10% de succès, UDRL permet théoriquement d'extraire la politique optimale, là où une méthode moyenne échouerait.

.. seealso::
	Pour la mise en œuvre de la régression quantile, voir :ref:`methodology`.
