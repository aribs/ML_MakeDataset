import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
df = pd.read_csv('dataset_boolean.csv')

# Preparación de los datos
X = df.drop(['message'], axis=1)  # Excluir columna 'message'
X['creation_date'] = pd.to_datetime(X['creation_date'], format='%Y/%m/%d')
X['updated_date'] = pd.to_datetime(X['updated_date'], format='%Y/%m/%d')
# Extraer características de las fechas
X = X.assign(
    creation_year=X['creation_date'].dt.year,
    creation_month=X['creation_date'].dt.month,
    creation_day=X['creation_date'].dt.day,
    updated_year=X['updated_date'].dt.year,
    updated_month=X['updated_date'].dt.month,
    updated_day=X['updated_date'].dt.day,
    days_difference=(X['updated_date'] - X['creation_date']).dt.days
)
X.drop(columns=['creation_date', 'updated_date'], inplace=True)
y = X.pop('is_smsing')  # Separar la variable objetivo

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicción y evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Preparación para predicciones en el conjunto de datos de prueba
df_test = pd.read_csv('dataset_to_predict_boolean.csv')
df_test.drop(columns=['message'], inplace=True)
df_test['creation_date'] = pd.to_datetime(df_test['creation_date'], format='%Y/%m/%d')
df_test['updated_date'] = pd.to_datetime(df_test['updated_date'], format='%Y/%m/%d')
df_test = df_test.assign(
    creation_year=df_test['creation_date'].dt.year,
    creation_month=df_test['creation_date'].dt.month,
    creation_day=df_test['creation_date'].dt.day,
    updated_year=df_test['updated_date'].dt.year,
    updated_month=df_test['updated_date'].dt.month,
    updated_day=df_test['updated_date'].dt.day,
    days_difference=(df_test['updated_date'] - df_test['creation_date']).dt.days
)
df_test.drop(columns=['creation_date', 'updated_date'], inplace=True)

# Asegurar que df_test tenga las mismas columnas que X_train
df_test = df_test[X_train.columns]

# Predicciones
predicted_is_smsing = model.predict(df_test)
df_test['Predicted_is_smsing'] = predicted_is_smsing

# Guardar las predicciones en un nuevo archivo CSV
df_test.to_csv('predicted_data_test.csv', index=False)
print("Predicciones guardadas en 'predicted_data_test.csv'.")
