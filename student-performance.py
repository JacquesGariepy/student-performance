import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the CSV file https://www.kaggle.com/datasets/impapan/student-performance-data-set?resource=download
math_data = pd.read_csv(os.path.join(dir_path, "student-mat.csv"), sep=';')
port_data = pd.read_csv(os.path.join(dir_path, "student-por.csv"), sep=';')

# Combiner les datasets si nécessaire
data = pd.concat([math_data, port_data])

# Nettoyer les données (exemple de transformation de variables catégorielles)
data['school'] = data['school'].apply(lambda x: 1 if x == 'GP' else 0)
data['sex'] = data['sex'].apply(lambda x: 1 if x == 'F' else 0)
data['address'] = data['address'].apply(lambda x: 1 if x == 'U' else 0)
data['famsize'] = data['famsize'].apply(lambda x: 1 if x == 'GT3' else 0)
data['Pstatus'] = data['Pstatus'].apply(lambda x: 1 if x == 'T' else 0)
data['schoolsup'] = data['schoolsup'].apply(lambda x: 1 if x == 'yes' else 0)
data['famsup'] = data['famsup'].apply(lambda x: 1 if x == 'yes' else 0)
data['paid'] = data['paid'].apply(lambda x: 1 if x == 'yes' else 0)
data['activities'] = data['activities'].apply(lambda x: 1 if x == 'yes' else 0)
data['nursery'] = data['nursery'].apply(lambda x: 1 if x == 'yes' else 0)
data['higher'] = data['higher'].apply(lambda x: 1 if x == 'yes' else 0)
data['internet'] = data['internet'].apply(lambda x: 1 if x == 'yes' else 0)
data['romantic'] = data['romantic'].apply(lambda x: 1 if x == 'yes' else 0)

# Utiliser get_dummies pour les colonnes catégorielles restantes
categorical_columns = ['Mjob', 'Fjob', 'reason', 'guardian']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Séparer les features et le label
X = data.drop(columns=['G3'])
y = data['G3']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle prédictif
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédire les résultats sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'MAE: {mae}, RMSE: {rmse}')

# Analyser les caractéristiques importantes
feature_importances = model.feature_importances_
important_features = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
print(important_features)

# Proposer des interventions éducatives basées sur les caractéristiques influentes
def recommend_interventions(student_data):
    if student_data['studytime'] < 2:
        return "Increase weekly study time."
    if student_data['schoolsup'] == 0:
        return "Provide extra educational support."
    # Ajouter plus de recommandations basées sur les caractéristiques importantes
    return "Keep up the good work!"

# Exemple d'utilisation de l'interface utilisateur
example_student = X_test.iloc[0]
print(recommend_interventions(example_student))

# Exemple d'utilisation de l'interface utilisateur pour recommander des interventions
example_student = X_test.iloc[0]
print("Données de l'étudiant:", example_student)
print("Recommandation:", recommend_interventions(example_student))

