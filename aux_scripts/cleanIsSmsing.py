import pandas as pd

# Ruta al archivo CSV
archivo_csv = "./data_to_test.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Verificar si la columna "is_smsing" existe para evitar errores
if "is_smsing" in df.columns:
    # Asignar un valor vacío a toda la columna "is_smsing"
    df["is_smsing"] = ""

    # Guardar los cambios en el mismo archivo
    # Si prefieres guardar en un nuevo archivo, cambia "archivo_csv" por el nombre del nuevo archivo
    df.to_csv(archivo_csv, index=False)
    print("La columna 'is_smsing' ha sido vaciada con éxito.")
else:
    print("La columna 'is_smsing' no existe en el archivo proporcionado.")
