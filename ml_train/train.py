import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Importa DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
df = pd.read_csv('dataset_boolean.csv')

# Quitar la columna 'domain_old_info' del conjunto de datos
df = df.drop(columns=['domain_old_info'])

# Convertir las columnas de fecha de string a datetime y extraer características
df['creation_date'] = pd.to_datetime(df['creation_date'], format='%Y/%m/%d')
df['updated_date'] = pd.to_datetime(df['updated_date'], format='%Y/%m/%d')
df['creation_year'] = df['creation_date'].dt.year
df['creation_month'] = df['creation_date'].dt.month
df['creation_day'] = df['creation_date'].dt.day
df['updated_year'] = df['updated_date'].dt.year
df['updated_month'] = df['updated_date'].dt.month
df['updated_day'] = df['updated_date'].dt.day
df['days_difference'] = (df['updated_date'] - df['creation_date']).dt.days

# La variable objetivo es la segunda columna, con índice 1
y = df.iloc[:, 1]

# Excluir solo la primera columna (índice 0) del conjunto de datos para X
X = df.drop(df.columns[0], axis=1)

# Selecciona todas las columnas numéricas incluyendo las nuevas características extraídas
X_numeric = X.select_dtypes(include=['number'])

# Manejo de valores faltantes solo en columnas numéricas
X_numeric.fillna(X_numeric.mean(), inplace=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar el DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)  # Usa DecisionTreeClassifier

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')