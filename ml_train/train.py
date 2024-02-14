import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
df = pd.read_csv('dataset.csv')

# La variable objetivo es la segunda columna, con índice 1
y = df.iloc[:, 1]

# Excluir solo la primera columna (índice 0) del conjunto de datos para X
# Esto mantiene la segunda columna (tu objetivo) dentro de X también
X = df.drop(df.columns[0], axis=1)

# Seleccionar solo columnas numéricas para X
X_numeric = X.select_dtypes(include=['number'])

# Manejo de valores faltantes solo en columnas numéricas
X_numeric.fillna(X_numeric.mean(), inplace=True)

# Ahora X_numeric es tu conjunto de características limpias y listo para usar
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


df_test = pd.read_csv('data_test.csv')


# Cargar el conjunto de datos de prueba
df_test = pd.read_csv('data_test.csv')

# Excluir solo la primera columna del conjunto de datos para X_test, manteniendo la segunda columna (aunque esté vacía)
X_test = df_test.drop(df_test.columns[0], axis=1)

# Seleccionar solo columnas numéricas para X_test, excluyendo la segunda columna si no es numérica
# Asumiendo que la segunda columna está vacía y no es numérica, ajusta según sea necesario
X_test_numeric = X_test.select_dtypes(include=['number'])

# Manejo de valores faltantes en el conjunto de prueba, usando la media del conjunto de entrenamiento
X_test_numeric.fillna(X_numeric.mean(), inplace=True)  # Usando X_numeric.mean() del conjunto de entrenamiento

print("Características del entrenamiento:", X_numeric.columns)
print("Características del test (numéricas):", X_test_numeric.columns)

# Hacer las predicciones con el modelo entrenado
predicted_classes = model.predict(X_test_numeric)

# Añadir las predicciones al DataFrame original de test (df_test), ajusta para insertar en la ubicación deseada si es necesario
df_test['Predicted_Class'] = predicted_classes

print(df_test[['Predicted_Class']])
