import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
df = pd.read_csv('dataset.csv')

# Quitar la columna 'domain_old_info' del conjunto de datos
df = df.drop(columns=['domain_old_info'])

# Convertir las columnas de fecha de string a datetime
df['creation_date'] = pd.to_datetime(df['creation_date'], format='%Y/%m/%d')
df['updated_date'] = pd.to_datetime(df['updated_date'], format='%Y/%m/%d')

# Extraer características de las fechas
df['creation_year'] = df['creation_date'].dt.year
df['creation_month'] = df['creation_date'].dt.month
df['creation_day'] = df['creation_date'].dt.day
df['updated_year'] = df['updated_date'].dt.year
df['updated_month'] = df['updated_date'].dt.month
df['updated_day'] = df['updated_date'].dt.day

# Opcional: Diferencia en días entre creation_date y updated_date
df['days_difference'] = (df['updated_date'] - df['creation_date']).dt.days

# La variable objetivo es la segunda columna, con índice 1 (ajustar si es necesario después de eliminar columnas)
y = df.iloc[:, 1]

# Excluir solo la primera columna (índice 0) del conjunto de datos para X
X = df.drop(df.columns[0], axis=1)

# Ahora, selecciona todas las columnas numéricas incluyendo las nuevas características extraídas
X_numeric = X.select_dtypes(include=['number'])

# Manejo de valores faltantes solo en columnas numéricas
X_numeric.fillna(X_numeric.mean(), inplace=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Configurar el RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Preparar el conjunto de datos de prueba siguiendo el mismo proceso
df_test = pd.read_csv('data_test.csv')
df_test = df_test.drop(columns=['domain_old_info'])  # Quitar la columna 'domain_old_info' del conjunto de prueba
df_test['creation_date'] = pd.to_datetime(df_test['creation_date'], format='%Y/%m/%d')
df_test['updated_date'] = pd.to_datetime(df_test['updated_date'], format='%Y/%m/%d')
df_test['creation_year'] = df_test['creation_date'].dt.year
df_test['creation_month'] = df_test['creation_date'].dt.month
df_test['creation_day'] = df_test['creation_date'].dt.day
df_test['updated_year'] = df_test['updated_date'].dt.year
df_test['updated_month'] = df_test['updated_date'].dt.month
df_test['updated_day'] = df_test['updated_date'].dt.day
df_test['days_difference'] = (df_test['updated_date'] - df_test['creation_date']).dt.days

X_test = df_test.drop(df_test.columns[0], axis=1)
X_test_numeric = X_test.select_dtypes(include=['number']).fillna(X_numeric.mean())

# Hacer las predicciones con el modelo entrenado
predicted_classes = model.predict(X_test_numeric)

# Añadir las predicciones al DataFrame original de test
df_test['Predicted_Class'] = predicted_classes

print(df_test[['Predicted_Class']])
