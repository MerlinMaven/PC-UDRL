
Cadre Théorique
===============

.. contents:: Sommaire
    :local:
    :depth: 2

1. Le Défi du RL Offline : "The Optimism Trap"
----------------------------------------------

Le Reinforcement Learning (RL) classique repose sur l'interaction : l'agent essaie, échoue, et apprend. Mais dans des domaines critiques (médecine, robotique industrielle, conduite), l'échec n'est pas une option.

Le **RL Offline** impose d'apprendre une politique optimale :math:`\pi^*` uniquement à partir d'un historique de données fixes :math:`\mathcal{D}`, sans aucune interaction supplémentaire.

Le problème majeur est le **Biais d'Optimisme** face à l'inconnu (OOD - Out of Distribution).
Pour une action :math:`a` jamais observée dans un état :math:`s`, les algorithmes classiques (Q-Learning) ont tendance à surestimer sa valeur :math:`Q(s,a)`.

.. danger::

   **Le Piège :** L'agent "hallucine" qu'une action dangereuse (ex: foncer dans un mur) va rapporter une récompense infinie simplement parce qu'il n'a jamais vu la conséquence négative.

2. Upside-Down RL : Fondement de Notre Travail
----------------------------------------------

Contrairement au RL classique qui maximise une récompense, l'**Upside-Down RL (UDRL)** traite le RL comme un problème d'apprentissage supervisé.

**Formulation Précise :**

À l'instant :math:`t`, l'agent observe l'état :math:`s_t` et reçoit une commande :math:`c = (r^*, h^*)` spécifiant le retour cible et l'horizon désiré. Il prédit une action :

.. math::
    a_t \sim \pi_\theta(a_t | s_t, r^*, h^*)

**Entraînement (Supervised Learning) :**

Sur un dataset :math:`\mathcal{D} = \{\tau_i\}`, pour chaque transition :math:`(s_t, a_t, r_t, ..., s_T)` :

1.  Calculer la commande atteinte (Ground Truth) :
    :math:`r_t^{achieved} = \sum_{k=t}^T r_k` et :math:`h_t^{remaining} = T - t`.
2.  Minimiser la perte comportementale (Negative Log-Likelihood) :

.. math::
    \mathcal{L}(\theta) = \mathbb{E}_{(s,a,r^{achieved},h^{remaining}) \sim \mathcal{D}} \left[ || \pi_\theta(s, r^{achieved}, h^{remaining}) - a ||^2 \right]

.. topic::

    **Le Dilemme de l'Obéissance :** Si l'utilisateur demande : *"Atteins un score de +1000"* (alors que le maximum possible est +200), l'agent UDRL va quand même essayer d'obéir. N'ayant jamais vu ce cas, il va extrapoler de manière erratique et **crasher**. C'est le problème de l'**Agent Obéissant**.

3. État de l'Art : Les Limites Existantes
-----------------------------------------

La littérature actuelle tente de résoudre le problème OOD par des contraintes punitives.

**A. Baselines Classiques**

.. list-table:: Performances État de l'Art (LunarLander Medium)
   :header-rows: 1

   * - Méthode
     - Mécanisme
     - Limite
   * - **CQL** (Kumar et al.)
     - **Value Regularization**. Punit les Q-valeurs OOD.
     - **Trop Conservateur**. L'agent "gèle" sur place.
   * - **TD3+BC** (Fujimoto et al.)
     - **Policy Constraint**. Force :math:`\pi \approx \pi_\beta`.
     - **Plafond de Verre**. Ne peut pas dépasser l'expert moyen.
   * - **IQL** (Kostrikov et al.)
     - **Expectile Regression**. Ne query jamais d'actions OOD.
     - Complexe et sensible aux hyperparamètres.

**B. La Lacune Identifiée**

Toutes ces méthodes brident **l'agilité** de l'agent pour assurer sa **sécurité**.
Notre thèse : *Il ne faut pas brider l'agent, il faut filtrer ses ordres.*

4. PC-UDRL : Notre Approche
---------------------------

**Pessimistic Command UDRL** propose une architecture découplée :

1.  **L'Agent (Le Pilote)** : Un modèle UDRL pur, agile et obéissant.
2.  **Le Pessimiste (Le Garde-Fou)** : Un modèle distinct qui apprend la *frontière de faisabilité* du monde.

**L'Innovation : Pessimisme de Commande**
Au lieu de modifier les poids de l'agent (comme CQL), nous modifions son **entrée** (input projection).

.. math::
    r_{safe} = \text{Project}(r_{target} | s)

Si l'utilisateur demande l'impossible, le Pessimiste projette cette demande sur le manifold des retours réalisables.

5. Les Trois Niveaux de Sophistication
--------------------------------------

Nous avons exploré trois paradigmes pour modéliser cette "Frontière de Faisabilité".

Niveau 1 : Régression Quantile (Approche Scalaire)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principe :**
On apprend séparément les bornes de faisabilité pour le retour :math:`r` et l'horizon :math:`h`.

**Formulation Mathématique :**
On apprend séparément les quantiles marginaux :math:`Q_\phi^r(\tau)` et :math:`Q_\phi^h(\tau)`.
La fonction de perte est la somme des **Pinball Losses** :

.. math::
    \mathcal{L}_{QR}(\phi) = \sum_{\tau} \mathbb{E}_{(r,h) \sim \mathcal{D}} [\rho_\tau(r - Q_\phi^r(\tau)) + \rho_\tau(h - Q_\phi^h(\tau))]

Où :math:`\rho_\tau(u) = |u| \times (\tau - \mathbb{I}_{u < 0})` est la perte asymétrique.

**Inférence (Clamping Indépendant) :**

.. math::
    r_{safe} = \text{clip}(r_{target}, Q_\phi^r(\tau_{min}), Q_\phi^r(\tau_{max}))

*   **Verdict** : **Trop Conservateur**. Cette méthode ignore les corrélations entre :math:`r` et :math:`h`. Elle définit le domaine de sécurité comme un rectangle, or c'est souvent une forme plus complexe.

Niveau 2 : Conditional VAE (Approche Latente)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principe :**
On utilise un CVAE pour apprendre la densité jointe :math:`p(r, h | s)` via un espace latent :math:`z`.

**Formulation Mathématique (ELBO) :**

.. math::
    \mathcal{L}_{CVAE} = \mathbb{E}_{(r,h,s) \sim \mathcal{D}} [ \underbrace{||(r,h) - \mu_\psi(z,s)||^2}_{\text{Reconstruction}} + \beta \underbrace{D_{KL}(q_\phi(z|r,h,s) || p(z))}_{\text{Régularisation}} ]

Où :math:`\beta \geq 1` contrôle la structure de l'espace latent (:math:`\beta`-VAE).

**Inférence (Optimisation Latente) :**
On cherche le point latent :math:`z_{safe}` qui minimise la distance à la commande cible, tout en restant dans la zone probable (prior gaussien) :

.. math::
    z_{safe} = \arg\min_z ||\mu_\psi(z,s) - (r_{targets}, h_{targets})||^2 \quad \text{s.c.} \quad ||z||^2 \le R^2

*   **Verdict** : **Trop Optimiste**. Les VAEs souffrent du "Prior Hole Problem". Ils assignent une probabilité non-nulle aux zones vides entre les clusters de données, menant à des commandes "chimériques".

Niveau 3 : Score-Based Diffusion (Approche Manifold)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principe :**
On modélise le **gradient** de la log-densité :math:`\nabla_x \log p(x|s)`. C'est l'approche la plus fidèle à la géométrie des données.

**Formulation Mathématique (DDPM) :**
On entraîne un réseau :math:`\epsilon_\theta(x_t, t, s)` à débruiter une commande bruitée :math:`x_t` à l'étape :math:`t`.

.. math::
    \mathcal{L}_{Diff} = \mathbb{E}_{x_0, \epsilon, t} \left[ || \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t, s) ||^2 \right]

Où :math:`\bar{\alpha}_t` suit un schedule cosine ou linéaire défini.

**Inférence (Guided Projection) :**
Pour projeter une commande cible :math:`x_{target}` :

1.  **Initialisation** : On part d'une version bruitée de la cible : :math:`x_T \sim \mathcal{N}(x_{target}, \sigma^2 I)`.
2.  **Processus Inverse** : On itère pour :math:`t = T \dots 1` :
    :math:`x_{t-1} = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t, s))`
3.  **Résultat** : La commande finale :math:`x_0` a "migré" sur le manifold des données valides.

*   **Verdict** : **Optimal**. Offre un "freinage adaptatif" et respecte les corrélations complexes entre retour et horizon.

6. Bibliographie & Liens Utiles
-------------------------------

Voici les publications fondamentales citées dans ce document :

*   `Reinforcement Learning Upside Down: Don't Predict Rewards -- Just Map Them to Actions <https://arxiv.org/abs/1912.02877>`_
    *J. Schmidhuber (2019)* - Le papier fondateur de l'UDRL.

*   `Conservative Q-Learning for Offline Reinforcement Learning <https://arxiv.org/abs/2006.04779>`_
    *Kumar et al. (NeurIPS 2020)* - La méthode CQL (Value Regularization).

*   `Offline Reinforcement Learning with Implicit Q-Learning <https://arxiv.org/abs/2110.06169>`_
    *Kostrikov et al. (ICLR 2022)* - La méthode IQL (Expectile Regression).

*   `A Minimalist Approach to Offline Reinforcement Learning <https://arxiv.org/abs/2106.06860>`_
    *Fujimoto & Gu (NeurIPS 2021)* - La méthode TD3+BC (Policy Constraint).

*   `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_
    *Ho et al. (NeurIPS 2020)* - Les fondements des modèles de diffusion.

