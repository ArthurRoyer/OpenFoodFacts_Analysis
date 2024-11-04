# Projet de Prédiction du Nutri-Score

## Description

Ce projet consiste à développer une application web de prédiction du Nutri-Score pour des produits alimentaires, en utilisant **Flask** comme framework web et **Scikit-learn** pour le Machine Learning. Le Nutri-Score est un indicateur nutritionnel allant de "A" (bon pour la santé) à "E" (à limiter). Le but de cette application est d’offrir une estimation du Nutri-Score à partir des informations nutritionnelles d'un produit.

## Fonctionnalités

1. **Prédiction du Nutri-Score** : Les utilisateurs peuvent entrer des informations nutritionnelles d’un produit et obtenir une estimation du Nutri-Score.
2. **Catégorisation automatique** : Sélection de la catégorie d'aliment pour affiner les prédictions.
3. **Évaluation de précision** : Précision du modèle basée sur les scores d’erreur comme le Mean Absolute Error (MAE), le Mean Squared Error (MSE), et le R2 Score.

## Installation et Exécution

### Installation

1. Clonez ce repository :

    ```bash
    git clone https://github.com/ArthurRoyer/OpenFoofFacts_Analysis.git
    cd OpenFoofFacts_Analysis
    ```

2. Installez les dépendances nécessaires :

    ```bash
    pip install -r requirements.txt
    ```

3. Générez le dataset `cleaned_data.csv` à l'aide de data_cleaner.ipynb
   
4.  Générez le modèle à l'aide de modele_generator.ipynb

5. Exécutez l'application Flask :

    ```bash
    python run.py
    ```

6. Rendez-vous sur `http://127.0.0.1:5000` dans votre navigateur pour accéder à l’application.

## Structure du Projet

- `app/` : Contient les fichiers de configuration Flask et le code de l'application.
  - `routes.py` : Définit les routes de l'application (prediction et results).
- `static/` : Contient les fichiers statiques (CSS, images).
- `templates/` : Contient les templates HTML pour l’interface utilisateur.
  - `predict.html` : Formulaire pour entrer les données d’un produit.
  - `results.html` : Affichage des résultats de la prédiction.
 
    
## Structure du data cleaner
Le module data_cleaner sert à télécharger, décompresser et enlever les valeurs non représentatives dans le fichier csv d'open food fact.
Les étapes de cleans sont : 
    - Sélectionner les colonnes qui paraissaient pertinentes à première vue.
    - Supprimer les colonnes vides à 90%
    - Réunir et sélectionner les lignes avec des pays valides
    - Suppression des outliers et données abbérantes
    - Suppresssion des doublons (plusieurs itération du même produits par marque)
    - Suppression des lignes avec une valeur manquante

    
## Modèle de Machine Learning

Le modèle utilisé pour la prédiction est un **Gradient Boosting Regressor**, optimisé pour minimiser l'erreur d'estimation du Nutri-Score. Voici les étapes de préparation des données et d’entraînement du modèle :

1. **Préparation des données** :
   - Les données d’origine sont chargées depuis un fichier CSV (`cleaned_data.csv`).
   - La colonne cible pour la prédiction est `nutriscore_score`.
   - Transformation des variables catégorielles avec des variables indicatrices (one-hot encoding de la colonne pnns_group_2) pour améliorer la précision du modèle.

2. **Normalisation** : Les caractéristiques sont normalisées via un `MinMaxScaler` pour garantir une échelle uniforme entre les différentes caractéristiques.

3. **Modèle** : Nous avons utilisé un **Gradient Boosting Regressor** avec les hyperparamètres suivants. Ces hypers paramètres ont été sélectionné grâce à halvenGridSearch:
### Hyperparamètres du modèle GradientBoostingRegressor

- **learning_rate (float)** : Définit le pas d'apprentissage, c'est-à-dire la contribution de chaque arbre aux prédictions finales. Une valeur plus faible (comme `0.1`) réduit le risque de surapprentissage mais peut nécessiter plus d'estimateurs (`n_estimators`).

- **n_estimators (int)** : Nombre d'arbres dans le modèle. Avec `150`, le modèle a un bon équilibre entre précision et complexité, mais il peut nécessiter un ajustement en fonction des performances pour éviter le surapprentissage.

- **max_depth (int)** : Profondeur maximale des arbres individuels. Limiter la profondeur à `3` aide à contrôler la complexité du modèle et réduit le risque de surapprentissage en évitant des arbres trop spécialisés.

- **random_state (int)** : Graines de randomisation pour garantir la reproductibilité des résultats. Avec `14`, les résultats seront identiques à chaque exécution avec les mêmes données.

- **max_features (str ou float)** : Nombre de caractéristiques maximales prises en compte pour chaque arbre. `"sqrt"` sélectionne la racine carrée du nombre total de caractéristiques, ce qui favorise la diversité des arbres et améliore la généralisation.

- **min_samples_leaf (int)** : Nombre minimum d’échantillons requis dans chaque feuille de l’arbre. Une valeur de `5` évite des feuilles trop petites, améliorant la robustesse du modèle aux variations.

- **min_samples_split (int)** : Nombre minimum d'échantillons requis pour diviser un nœud. Avec `20`, cela limite la création de sous-arbres à partir de petites quantités de données, réduisant ainsi le surapprentissage.

- **subsample (float)** : Fraction des données utilisée pour chaque arbre. Une valeur de `0.7` (70%) crée une diversité entre les arbres et agit comme une régularisation pour améliorer la généralisation.

Ces choix d'hyperparamètres ont été sélectionnés pour optimiser la performance du modèle tout en réduisant le risque de surapprentissage, garantissant ainsi une meilleure capacité de généralisation aux données nouvelles.

Après affinements de ces choix ,les metrics du modèles sont de :
    MAE: 1.2570244076612143
    MSE: 3.385375251864288
    R²: 0.9561694712256452

### Gradient Boosting Regressor : Avantages ###

Le **Gradient Boosting Regressor** est un algorithme de régression puissant, basé sur l'idée d'améliorer successivement la performance d'un ensemble de modèles faibles (typiquement des arbres de décision). Voici les principaux avantages de cet algorithme :

1. **Haute performance prédictive** : En combinant de nombreux modèles faibles, le Gradient Boosting Regressor atteint souvent des résultats de prédiction très précis, surpassant d'autres algorithmes dans de nombreux contextes (comme l'analyse prédictive ou les compétitions de données).

2. **Flexibilité** : Il est adapté aux tâches de régression complexes avec des relations non linéaires entre les variables, car les arbres de décision peuvent capturer des non-linéarités et des interactions entre les caractéristiques.

3. **Robustesse au surapprentissage** : Le Gradient Boosting Regressor applique une régularisation (souvent via le paramètre de taux d’apprentissage), ce qui aide à limiter le surapprentissage, surtout dans les versions modernes comme **XGBoost**, **LightGBM** ou **CatBoost**.

4. **Gestion des données déséquilibrées** : Même avec des distributions de données déséquilibrées, le modèle s'adapte bien en construisant progressivement une combinaison de modèles qui minimise les erreurs résiduelles à chaque itération.

5. **Prise en compte des données manquantes** : Les implémentations modernes, comme **XGBoost** ou **LightGBM**, peuvent gérer des données manquantes sans nécessiter de prétraitement exhaustif.

6. **Personnalisation par hyperparamètres** : Le Gradient Boosting Regressor propose de nombreux paramètres de réglage, permettant d'optimiser finement le modèle (taux d’apprentissage, profondeur des arbres, nombre d’itérations, etc.).

7. **Résilience aux valeurs aberrantes** : Basé sur des arbres de décision, le modèle est relativement peu sensible aux valeurs aberrantes et aux caractéristiques bruitées, en se concentrant sur la réduction des erreurs globales.

8. **Utilisation efficace de la mémoire** : Bien que les modèles d'ensemble d'arbres puissent être lourds, le Gradient Boosting est souvent plus efficace en mémoire que d’autres techniques d'ensemble comme les forêts aléatoires, car il optimise progressivement sans nécessiter un stockage massif de modèles.

En somme, le Gradient Boosting Regressor est un outil polyvalent et performant, idéal pour des problèmes de régression complexes nécessitant des prédictions fiables et optimisées.

### Prédiction et Evaluation

Le modèle effectue une prédiction du score, qui est ensuite converti en une lettre (`A`, `B`, `C`, `D`, ou `E`) via une fonction `score_to_grade` :
   - `A` : score <= -1
   - `B` : 0 <= score <= 2
   - `C` : 3 <= score <= 10
   - `D` : 11 <= score <= 18
   - `E` : score > 18

### Tests de corrélation

Des tests de corrélation ont été effectués pour analyser les relations entre les différentes variables nutritionnelles. Cela permet de repérer les variables qui présentent des dépendances ou des similitudes importantes, aidant à simplifier le modèle tout en conservant les informations les plus pertinentes. Une heatmap de corrélation est notamment utilisée pour visualiser l’intensité des relations, et une Analyse en Composantes Principales (ACP) est également appliquée pour réduire la dimensionnalité des données. ​

### Résultats

Les résultats de la prédiction sont affichés sous la forme :
- **Score prédictif** : Un score numérique est affiché pour l’utilisateur, en fonction des données saisies.
- **Nutri-Score** : La note Nutri-Score est donnée sous forme de lettre (A à E), permettant une évaluation rapide et claire.


## Interface Utilisateur

- **predict.html** : Formulaire où les utilisateurs saisissent les informations nutritionnelles du produit. Une option permet également de choisir la catégorie de produit pour affiner la prédiction.
- **results.html** : Affiche la prédiction de Nutri-Score et le score associé.
- **Modal d’avertissement** : Si certains champs sont laissés vides, une fenêtre modale avertit l’utilisateur que cela pourrait affecter le calcul.

## API Flask

Le fichier `routes.py` contient l’API pour les fonctionnalités principales :
- **Route `/`** : Renvoie la page d’accueil avec les graphiques de visualisation.
- **Route `/predict`** : Affiche le formulaire de prédiction.
- **Route `/results`** : Prend les données du formulaire et retourne les résultats de prédiction du Nutri-Score.
- **Route `\api\predict`** : Utilisation d'un POST dans Postman avec en Body un JSON de test pour tester l'appli. Celle ci renvoi un Json de réponse.
## Contributions

Les contributions sont les bienvenues ! Merci de soumettre une `pull request` pour tout ajout ou amélioration.

---

Développé dans le cadre d'un projet de formation en Machine Learning et IA.
https://trello.com/b/Q7za6Yp6/projet-fit-pour-tous
