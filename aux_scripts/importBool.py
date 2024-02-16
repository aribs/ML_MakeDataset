import pandas as pd

# Cargar el conjunto de datos
df = pd.read_csv('data_to_predict.csv')

# Lista de columnas para cambiar de 1/0 a True/False
columns_to_convert = ['is_smsing', 'safe_info', 'domain_old_info', 'is_ip', 'is_https',
                      'special_characters', 'tlds_blacklist', 'make_redirection', 'is_suspended']

# Verificar y corregir posible typo en el nombre de la columna 'tlds_blacklist'
# Asegúrate de que el nombre de la columna sea correcto según tu archivo CSV
if 'tlds_blacklist' not in df.columns and 'tlds_blackist' in df.columns:
    columns_to_convert[columns_to_convert.index('tlds_blacklist')] = 'tlds_blackist'

# Realizar la conversión
for column in columns_to_convert:
    if column in df.columns:  # Verifica si la columna existe en el DataFrame
        df[column] = df[column].map({1: True, 0: False})

# Guardar el resultado en un nuevo archivo CSV
df.to_csv('dataset_to_predict_boolean.csv', index=False)
