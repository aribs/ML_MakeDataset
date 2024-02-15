import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Cambio aquí para importar RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
df = pd.read_csv('dataset_boolean.csv')

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

# Configurar el RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Uso de RandomForestClassifier

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
