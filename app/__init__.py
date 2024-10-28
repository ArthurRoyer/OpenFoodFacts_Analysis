from flask import Flask
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

df=pd.read_csv("cleaned_data.csv",sep=',',on_bad_lines='skip')

df=df.drop(['code','nutrition-score-fr_100g','energy-kj_100g','product_name','nutriscore_grade','created_datetime','quantity','brands','categories','categories_en','pnns_groups_1','pnns_groups_2','main_category_en','ingredients_text','countries_en','product_name_lower','brands_lower'], axis = 1)
X = df.drop("nutriscore_score", axis = 1)
y = df["nutriscore_score"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

Rtr = DecisionTreeRegressor()
Rtr.fit(X_train_scaled,y_train)

def create_app():
    app = Flask(__name__)
    
    from app.routes import main
    app.register_blueprint(main)

    return app
