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

## Modèle de Machine Learning

Le modèle utilisé pour la prédiction est un **Gradient Boosting Regressor**, optimisé pour minimiser l'erreur d'estimation du Nutri-Score. Voici les étapes de préparation des données et d’entraînement du modèle :

1. **Préparation des données** :
   - Les données d’origine sont chargées depuis un fichier CSV (`cleaned_data.csv`).
   - La colonne cible pour la prédiction est `nutriscore_score`.
   - Transformation des variables catégorielles avec des variables indicatrices (one-hot encoding) pour améliorer la précision du modèle.

2. **Normalisation** : Les caractéristiques sont normalisées via un `MinMaxScaler` pour garantir une échelle uniforme entre les différentes caractéristiques.

3. **Modèle** : Nous avons utilisé un **Gradient Boosting Regressor** avec les hyperparamètres suivants :
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


### Prédiction et Evaluation

Le modèle effectue une prédiction du score, qui est ensuite converti en une lettre (`A`, `B`, `C`, `D`, ou `E`) via une fonction `score_to_grade` :
   - `A` : score <= -1
   - `B` : 0 <= score <= 2
   - `C` : 3 <= score <= 10
   - `D` : 11 <= score <= 18
   - `E` : score > 18

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

## Contributions

Les contributions sont les bienvenues ! Merci de soumettre une `pull request` pour tout ajout ou amélioration.

---

Développé dans le cadre d'un projet de formation en Machine Learning et IA.
