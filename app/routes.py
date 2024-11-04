from flask import Blueprint, render_template, request, jsonify
from app.graphs import create_graph
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

df=pd.read_csv("cleaned_data.csv",sep=',',on_bad_lines='skip', low_memory=False)

df=df.drop(['created_datetime','energy-kj_100g','code','nutrition-score-fr_100g','product_name','quantity','brands','categories','categories_en','pnns_groups_1','main_category_en','ingredients_text','countries_en','nutriscore_grade','product_name_lower','brands_lower'], axis = 1)
df = pd.get_dummies(df, columns=['pnns_groups_2'], drop_first=True)
X = df.drop("nutriscore_score", axis = 1)
y = df["nutriscore_score"]


# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des caractéristiques
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

main = Blueprint('main', __name__)

def score_to_grade(score):
    if score <= -1:
        return 'A'
    elif 0 <= score <= 2:
        return 'B'
    elif 3 <= score <= 10:
        return 'C'
    elif 11 <= score <= 18:
        return 'D'
    else:
        return 'E'

# Mapping des catégories à leurs indices dans le tableau new_data
category_mapping = {
    "pnns_groups_2_Appetizers": 8,
    "pnns_groups_2_Artificially_sweetened_beverages": 9,
    "pnns_groups_2_Biscuits_and_cakes": 10,
    "pnns_groups_2_Bread": 11,
    "pnns_groups_2_Breakfast_cereals": 12,
    "pnns_groups_2_Cereals": 13,
    "pnns_groups_2_Cheese": 14,
    "pnns_groups_2_Chocolate_products": 15,
    "pnns_groups_2_Dairy_desserts": 16,
    "pnns_groups_2_Dressings_and_sauces": 17,
    "pnns_groups_2_Dried_fruits": 18,
    "pnns_groups_2_Eggs": 19,
    "pnns_groups_2_Fats": 20,
    "pnns_groups_2_Fish_and_seafood": 21,
    "pnns_groups_2_Fruit_juices": 22,
    "pnns_groups_2_Fruit_nectars": 23,
    "pnns_groups_2_Fruits": 24,
    "pnns_groups_2_Ice_cream": 25,
    "pnns_groups_2_Legumes": 26,
    "pnns_groups_2_Meat": 27,
    "pnns_groups_2_Milk_and_yogurt": 28,
    "pnns_groups_2_Nuts": 29,
    "pnns_groups_2_Offals": 30,
    "pnns_groups_2_One_dish_meals": 31,
    "pnns_groups_2_Pastries": 32,
    "pnns_groups_2_Pizza_pies_and_quiches": 33,
    "pnns_groups_2_Plant_based_milk_substitutes": 34,
    "pnns_groups_2_Potatoes": 35,
    "pnns_groups_2_Processed_meat": 36,
    "pnns_groups_2_Salty_and_fatty_products": 37,
    "pnns_groups_2_Sandwiches": 38,
    "pnns_groups_2_Soups": 39,
    "pnns_groups_2_Sweetened_beverages": 40,
    "pnns_groups_2_Sweets": 41,
    "pnns_groups_2_Teas_and_herbal_teas_and_coffees": 42,
    "pnns_groups_2_Unsweetened_beverages": 43,
    "pnns_groups_2_Vegetables": 44,
    "pnns_groups_2_Waters_and_flavored_waters": 45,
    "pnns_groups_2_unknown": 46
}

# Ajouter cette fonction au début du fichier, après les imports et avant les routes
def parse_input(value, nom):
    global df
    return df[nom].median() if value == "" else float(value)

@main.route('/')
def dashboard():
    graph = create_graph()
    return render_template('index.html', graph=graph)

@main.route('/predict')
def predict():
    return render_template('predict.html')

@main.route('/results', methods=['POST'])
def results():
    # Récupérer et traiter les valeurs des champs du formulaire
    energy_kcal = parse_input(request.form.get('energy-kcal'),'energy-kcal_100g')
    fat = parse_input(request.form.get('fat'),'fat_100g')
    saturated_fat = parse_input(request.form.get('saturated-fat'),'saturated-fat_100g')
    sugars = parse_input(request.form.get('sugars'), 'sugars_100g')
    fiber = parse_input(request.form.get('fiber'), 'fiber_100g')
    proteins = parse_input(request.form.get('proteins'), 'proteins_100g')
    salt = parse_input(request.form.get('salt'), 'salt_100g')
    fruits_vegetables_nuts_estimate = parse_input(request.form.get('fruits-vegetables-nuts-estimate-from-ingredients'), 'fruits-vegetables-nuts-estimate-from-ingredients_100g')
    selected_name = request.form.get('selected_name')

    # Initialiser les valeurs de new_data
    new_data = np.array([[energy_kcal, fat, saturated_fat, sugars, fiber, proteins, salt, fruits_vegetables_nuts_estimate] + [0] * 39])

    # Activer l’indice correspondant à la catégorie sélectionnée
    if selected_name in category_mapping:
        category_index = category_mapping[selected_name]
        new_data[0, category_index] = 1  # On définit cette catégorie sur 1


    # Normaliser les nouvelles données avec le scaler
    new_data_scaled = scaler.transform(new_data)

    # Prédire avec le modèle entraîné
    y_new_pred = model.predict(new_data_scaled)

    score_new_pred = round(y_new_pred[0])
    y_new_pred = score_to_grade(score_new_pred)

    # Passer la prédiction au template pour affichage
    return render_template('results.html', y_new_pred=y_new_pred, score_new_pred=score_new_pred)

@main.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        
        # Validation des données
        required_fields = ['energy-kcal', 'fat', 'saturated-fat', 'sugars', 
                         'fiber', 'proteins', 'salt', 
                         'fruits-vegetables-nuts-estimate-from-ingredients',
                         'selected_name']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Le champ {field} est manquant'}), 400

        # Traitement des données
        energy_kcal = parse_input(str(data['energy-kcal']), 'energy-kcal_100g')
        fat = parse_input(str(data['fat']), 'fat_100g')
        saturated_fat = parse_input(str(data['saturated-fat']), 'saturated-fat_100g')
        sugars = parse_input(str(data['sugars']), 'sugars_100g')
        fiber = parse_input(str(data['fiber']), 'fiber_100g')
        proteins = parse_input(str(data['proteins']), 'proteins_100g')
        salt = parse_input(str(data['salt']), 'salt_100g')
        fruits_vegetables_nuts_estimate = parse_input(
            str(data['fruits-vegetables-nuts-estimate-from-ingredients']),
            'fruits-vegetables-nuts-estimate-from-ingredients_100g'
        )
        selected_name = data['selected_name']

        # Création du tableau de données
        new_data = np.array([[energy_kcal, fat, saturated_fat, sugars, fiber, 
                             proteins, salt, fruits_vegetables_nuts_estimate] + [0] * 39])

        # Activation de la catégorie sélectionnée
        if selected_name in category_mapping:
            category_index = category_mapping[selected_name]
            new_data[0, category_index] = 1

        # Normalisation et prédiction
        new_data_scaled = scaler.transform(new_data)
        y_new_pred = model.predict(new_data_scaled)
        score_new_pred = round(y_new_pred[0])
        grade = score_to_grade(score_new_pred)

        return jsonify({
            'nutriscore_grade': grade,
            'nutriscore_score': score_new_pred
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
