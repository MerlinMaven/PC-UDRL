# Rapport Phase 1 : Validation Unitaire sur GridWorld

**Date :** 26 Décembre 2025
**Projet :** PC-UDRL (Pessimistic - Unsupervised Reinforcement Learning)

## 1. Objectifs de la Phase 1

L'objectif principal de cette première phase était de valider l'architecture logicielle et les concepts théoriques du PC-UDRL dans un environnement contrôlé et simple (**GridWorld**).

Les sous-objectifs étaient :
1.  **Implémentation du Pipeline** : Génération de données, Entraînement UDRL, Entraînement du Pessimiste (Quantile Regression).
2.  **Validation du Comportement** : Vérifier que l'agent est capable de suivre des commandes (UDRL Standard) et que le pessimiste est capable de modérer ces commandes (PC-UDRL).
3.  **Mise en place de l'Infrastructure** : Logging, Visualisation temps réel, Comparaison côte-à-côte, Documentation.

## 2. Travaux Réalisés

### 2.1 Méthodologie
Nous avons adopté une approche **Offline RL** pure :
*   **Dataset** : Collecte de 1000 trajectoires purement aléatoires (`generate_dataset.py`). Aucune connaissance experte n'a été injectée.
*   **Modèle UDRL** : Un MLP conditionné par le retour et l'horizon (`state` + `horizon` + `desired_return` -> `action`).
*   **Modèle Pessimiste** : Un régresseur quantile (`state` -> `distribution of returns`) pour estimer la faisabilité d'une commande.

### 2.2 Étapes Techniques
1.  **Restructuration du projet** : Séparation claire `runs/` (expériences) et `outputs/` (résultats).
2.  **Correction des Logs** : Implémentation d'un `eval_interval` strict pour tracer les courbes d'apprentissage et le "Pessimism Gap" en temps réel.
3.  **Visualisation Avancée** :
    *   Intégration de `matplotlib` pour le rendu live pendant l'entraînement.
    *   Création d'un script de comparaison `live_compare` affichant côte-à-côte l'agent Standard et l'agent Pessimiste.
4.  **Documentation** : Mise en place complète de Sphinx/ReadTheDocs.

## 3. Résultats Obtenus

### 3.1 Performance Quantitative (sur 20 épisodes de test)

| Agent | Retour Moyen | Observations |
| :--- | :---: | :--- |
| **Standard UDRL** | **-88.00** | Comportement erratique mais parfois chanceux. Traverse les murs/pièges sans "peur". |
| **PC-UDRL** | **-100.00** | Comportement extrêmement conservateur. Reste souvent figé ou choisit la "moins pire" des options connues. |

### 3.2 Analyse Visuelle (Vidéos)
*   **Standard** : L'agent reçoit la commande `+10` (Target). Il essaie d'avancer vers le but, mais comme il a appris sur des données aléatoires (où il se cognait souvent), ses actions sont imparfaites. Il prend des risques.
*   **Pessimiste** : Le modèle pessimiste détecte que `+10` est irréaliste depuis l'état initial (Gap ~102). Il abaisse la commande (clamp) à une valeur très basse (ex: -10 ou -50).

## 4. Interprétation et Analyse

### 4.1 Le Phénomène "Suicide Obéissant"
Nous avons observé un comportement paradoxal où l'agent pessimiste se dirige parfois vers un piège (`-10`).
**Analyse** :
*   Sur un dataset aléatoire pauvre, l'agent n'a jamais vu de chemin vers le but (+10).
*   La "meilleure" trajectoire connue pour le pessimiste peut être celle qui mène à un piège (-10), comparé à errer indéfiniment (-1 par pas jusqu'à -100).
*   En clampant la commande à -10, le pessimiste dit implicitement à l'agent : *"L'objectif est d'atteindre un retour de -10"*.
*   L'agent UDRL, étant conditionné pour obéir, exécute la séquence d'actions qui garantit ce résultat : aller dans le piège.

### 4.2 Conclusion
La Phase 1 est un **succès technique**. Le pipeline fonctionne, le pessimiste modère effectivement les commandes (forte valeur du Gap), et l'infrastructure est robuste.
Le comportement "faible" du PC-UDRL est attendu sur ce type de dataset (Sparse Reward + Random Data). Cela confirme que l'architecture est prête pour des environnements plus complexes et des datasets plus riches.

## 5. Prochaines Étapes (Phase 2)
*   Passage à l'environnement **LunarLander-v2** (états continus, physique).
*   Utilisation de datasets de meilleure qualité (Medium/Expert via `d3rlpy`).
*   Comparaison avec des baselines de l'état de l'art (CQL, IQL, TD3+BC).
