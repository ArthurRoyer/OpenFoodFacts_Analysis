markdown
Copier le code
# Prédiction du Nutri-Score avec un Arbre de Régression

Ce projet implémente un modèle de régression supervisé pour prédire le `nutriscore_score` de produits alimentaires, basé sur leurs caractéristiques nutritionnelles. Le modèle utilise un arbre de régression de type CART (Classification And Regression Trees) pour effectuer les prédictions.

## Contexte et Objectif

La régression est une technique d'apprentissage supervisé visant à modéliser la relation entre une variable cible continue et plusieurs variables indépendantes. L'objectif de ce projet est de :
1. Charger et préparer un ensemble de données de produits alimentaires.
2. Utiliser un modèle de régression pour prédire le `nutriscore_score`.
3. Évaluer la précision des prédictions du modèle.

## Structure des Données

Le fichier de données, `cleaned_data.csv`, contient les informations nutritionnelles de divers produits, incluant des variables comme `energy-kj_100g`, `fat_100g`, `sugars_100g`, et `nutriscore_grade`.

### Colonnes sélectionnées pour la régression

- `nutriscore_score`: la variable cible, représentant le score Nutri-Score.
- Variables indépendantes sélectionnées : `energy-kj_100g`, `fat_100g`, `saturated-fat_100g`, `sugars_100g`, `fiber_100g`, `proteins_100g`, `salt_100g`, `fruits-vegetables-nuts-estimate-from-ingredients_100g`.

## Prérequis

- Python 3.x
- Bibliothèques Python :
  - pandas
  - scikit-learn

Vous pouvez installer les dépendances nécessaires avec :

pip install pandas scikit-learn

Instructions d'Exécution
Cloner le dépôt :

bash
Copier le code
git clone [https://github.com/votre-utilisateur/nutri-score-prediction.git](https://github.com/ArthurRoyer/OpenFoofFacts_Analysis.git)
cd nutri-score-prediction
Charger les données : Importez le fichier cleaned_data.csv dans un DataFrame et effectuez les traitements de nettoyage initiaux.

python
Copier le code
import pandas as pd
df = pd.read_csv("cleaned_data.csv", sep=',', on_bad_lines='skip')
Préparer les données :

Supprimer les colonnes inutiles pour la régression.
Effectuer un encodage one-hot sur la colonne nutriscore_grade.

Entraîner le modèle :

Séparer les données en ensembles d'entraînement et de test.
Normaliser les données.
Entraîner un arbre de régression.


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

Évaluer le modèle : Utilisez les métriques suivantes pour évaluer les performances :

Erreur absolue moyenne (MAE)
Erreur quadratique moyenne (MSE)
Coefficient de détermination (R²)
Résultats
Le modèle a montré de bonnes performances avec :

MAE : 0.41
MSE : 0.99
R² : 0.98
Ces résultats indiquent une faible erreur de prédiction et une haute précision du modèle.
