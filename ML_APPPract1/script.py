import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import os


# Funcție pentru încărcarea și preprocesarea datelor
def load_and_preprocess(filepath):
    # Verifică dacă fișierul există
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fișierul nu a fost găsit: {filepath}")

    print(f"Încărcarea fișierului: {filepath}")

    # Citește fișierul Excel fără argumentul 'dayfirst'
    try:
        data = pd.read_excel(filepath, parse_dates=['Data'])
    except Exception as e:
        raise ValueError(f"A apărut o problemă la încărcarea fișierului Excel: {e}")

    print("Fișier încărcat cu succes. Coloane disponibile:")
    print(data.columns)

    # Verifică existența coloanelor de interes
    columns_of_interest = ['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
                           'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]', 'Sold[MW]']
    missing_columns = [col for col in columns_of_interest if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Coloanele lipsă din fișier: {missing_columns}")

    # Curăță NaN-uri și păstrează coloanele relevante
    data = data.dropna(subset=columns_of_interest)
    data = data[columns_of_interest]

    print("Datele au fost curățate. Primele 5 rânduri:")
    print(data.head())

    return data


# Funcție pentru antrenarea unui arbore de decizie ID3 adaptat la regresie
def train_id3_regressor(X_train, y_train):
    regressor = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
    regressor.fit(X_train, y_train)
    print("Modelul ID3 a fost antrenat.")
    return regressor


# Funcție pentru antrenarea unui clasificator bayesian adaptat la regresie
def train_bayes_regressor(X_train, y_train, num_bins=10):
    # Discretizare a valorilor țintă (y) în intervale
    bins = np.linspace(y_train.min(), y_train.max(), num_bins)
    y_train_discretized = np.digitize(y_train, bins)

    print("Valorile țintă au fost discretizate pentru Bayes.")
    print(f"Intervale: {bins}")

    # Antrenare model GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train_discretized)

    print("Modelul Bayesian a fost antrenat.")
    return model, bins


# Funcție pentru evaluarea modelului
def evaluate_model(model, X_test, y_test, bins=None, bayes=False):
    print("Evaluarea modelului...")
    if bayes:
        # Pentru Bayes, discretizăm predicțiile înapoi
        y_pred_discretized = model.predict(X_test)
        y_pred = (bins[y_pred_discretized - 1] + bins[y_pred_discretized]) / 2  # Medie între limite
    else:
        y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Predicțiile modelului: {y_pred[:5]} (primele 5 valori)")
    print(f"Scorurile: RMSE={rmse:.2f}, MAE={mae:.2f}")

    return rmse, mae


# Funcția principală
def main():
    filepath = "D:\\Simona\\Downloads\\Grafic_SEN.xlsx"  # Asigură-te că fișierul este în aceeași locație ca scriptul

    try:
        # Încarcă și preprocesează datele
        data = load_and_preprocess(filepath)

        # Separă caracteristicile (X) de țintă (y)
        X = data[['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
                  'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']]
        y = data['Sold[MW]']

        # Împarte datele în seturi de antrenare și testare
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Datele au fost împărțite în seturi de antrenare și testare.")

        # Antrenează modelul ID3
        id3_model = train_id3_regressor(X_train, y_train)

        # Antrenează modelul Bayesian
        bayes_model, bins = train_bayes_regressor(X_train, y_train)

        # Evaluează modelele
        id3_rmse, id3_mae = evaluate_model(id3_model, X_test, y_test)
        bayes_rmse, bayes_mae = evaluate_model(bayes_model, X_test, y_test, bins=bins, bayes=True)

        # Afișează rezultatele
        print("\nRezultate ID3:")
        print(f"  RMSE: {id3_rmse:.2f}")
        print(f"  MAE: {id3_mae:.2f}")

        print("\nRezultate Bayesian:")
        print(f"  RMSE: {bayes_rmse:.2f}")
        print(f"  MAE: {bayes_mae:.2f}")

    except Exception as e:
        print(f"A apărut o eroare: {e}")


# Rulează scriptul
if __name__ == "__main__":
    main()