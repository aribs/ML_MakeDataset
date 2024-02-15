import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
df = pd.read_csv('dataset.csv')

# Quitar la columna 'message' del conjunto de datos, manteniendo el resto
X = df.drop(['message'], axis=1)

# Convertir las columnas de fecha de string a datetime y extraer características
X['creation_date'] = pd.to_datetime(X['creation_date'], format='%Y/%m/%d')
X['updated_date'] = pd.to_datetime(X['updated_date'], format='%Y/%m/%d')

# Extraer características de las fechas
X['creation_year'] = X['creation_date'].dt.year
X['creation_month'] = X['creation_date'].dt.month
X['creation_day'] = X['creation_date'].dt.day
X['updated_year'] = X['updated_date'].dt.year
X['updated_month'] = X['updated_date'].dt.month
X['updated_day'] = X['updated_date'].dt.day

# Opcional: Diferencia en días entre creation_date y updated_date
X['days_difference'] = (X['updated_date'] - X['creation_date']).dt.days

# Ahora, quita las columnas originales de tipo datetime64 para evitar problemas de tipo de dato
X = X.drop(columns=['creation_date', 'updated_date'])

# La variable objetivo 'is_smsing' es parte de X, la separamos para tener y
y = X['is_smsing']

# Ahora, prepara X para el entrenamiento excluyendo la variable objetivo 'is_smsing'
X = X.drop(['is_smsing'], axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar el DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Cargar el conjunto de datos de prueba
df_test = pd.read_csv('data_test.csv')

# Asegurarse de excluir la columna 'message' como se hizo con el conjunto de entrenamiento
df_test = df_test.drop(columns=['message'])

# Convertir las columnas de fecha de string a datetime y extraer características
df_test['creation_date'] = pd.to_datetime(df_test['creation_date'], format='%Y/%m/%d')
df_test['updated_date'] = pd.to_datetime(df_test['updated_date'], format='%Y/%m/%d')

# Extraer características de las fechas
df_test['creation_year'] = df_test['creation_date'].dt.year
df_test['creation_month'] = df_test['creation_date'].dt.month
df_test['creation_day'] = df_test['creation_date'].dt.day
df_test['updated_year'] = df_test['updated_date'].dt.year
df_test['updated_month'] = df_test['updated_date'].dt.month
df_test['updated_day'] = df_test['updated_date'].dt.day

# Opcional: Diferencia en días entre creation_date y updated_date
df_test['days_difference'] = (df_test['updated_date'] - df_test['creation_date']).dt.days

# Quitar las columnas originales de fecha para evitar problemas de tipo de dato
df_test = df_test.drop(columns=['creation_date', 'updated_date'])

# Preparar el conjunto X_test para predicción (asegúrate de que X_test solo contenga las columnas usadas por el modelo)
X_test = df_test.drop(['is_smsing'], axis=1)  # Asume que 'is_smsing' no está presente o no se usa para predicción

# Realizar predicciones con el modelo entrenado
predicted_classes = model.predict(X_test)

# Si deseas añadir estas predicciones al DataFrame df_test puedes hacerlo así
df_test['Predicted_is_smsing'] = predicted_classes

# Opcionalmente, guardar las predicciones en un nuevo archivo CSV
df_test.to_csv('predicted_data_test.csv', index=False)

print("Predicciones guardadas en 'predicted_data_test.csv'.")
