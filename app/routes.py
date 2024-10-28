from flask import Blueprint, render_template, request
from app.graphs import create_graph
from app import Rtr, scaler  # Importer le modèle Rtr et le scaler
import numpy as np

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

@main.route('/')
def dashboard():
    graph = create_graph()
    return render_template('index.html', graph=graph)

@main.route('/predict')
def predict():
    return render_template('predict.html')

@main.route('/results', methods=['POST'])
def results():
    # Fonction pour gérer les valeurs de formulaire vides
    def parse_input(value):
        return np.nan if value == "" else float(value)

    # Récupérer et traiter les valeurs des champs du formulaire
    energy_kcal = parse_input(request.form.get('energy-kcal'))
    fat = parse_input(request.form.get('fat'))
    saturated_fat = parse_input(request.form.get('saturated-fat'))
    sugars = parse_input(request.form.get('sugars'))
    fiber = parse_input(request.form.get('fiber'))
    proteins = parse_input(request.form.get('proteins'))
    salt = parse_input(request.form.get('salt'))
    fruits_vegetables_nuts_estimate = parse_input(request.form.get('fruits-vegetables-nuts-estimate-from-ingredients'))

    # Créer un tableau avec les données du formulaire
    new_data = np.array([[energy_kcal, fat, saturated_fat, sugars, fiber, proteins, salt, fruits_vegetables_nuts_estimate]])

    # Normaliser les nouvelles données avec le scaler
    new_data_scaled = scaler.transform(new_data)

    # Prédire avec le modèle entraîné
    y_new_pred = Rtr.predict(new_data_scaled)

    score_new_pred = y_new_pred[0]
    y_new_pred = score_to_grade(score_new_pred)

    # Passer la prédiction au template pour affichage
    return render_template('results.html', y_new_pred=y_new_pred, score_new_pred=score_new_pred)
