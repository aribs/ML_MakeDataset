import random

# Definir el cuerpo del mensaje
introductions = [
    "NETFLIX: Su suscripción ha sido",
    "Alerta NETFLIX: Detectamos un problema con su cuenta, está",
    "AVISO de Netflix: Su cuenta ahora está",
    "Urgente Neflix: información de su cuenta",
    "NETFLIX BLOQUEADO: su cuenta",
    "NETFLIX cuenta bloqueada",
    "Detectado acceso a su cuenta Netflix",
    "Problemas con su cuenta Neflix",
    "Cancelación de su Cuenta Netflix",
    "Problemas con Pago cuenta Neflix",
    "Netflix BLOQUEADO",
    "ATENCIÓN NETFLIX",
    "Acceso Externo a su cuenta Netflix"
]
reasons = [
    "temporalmente suspendida.",
    "bloqueada por seguridad.",
    "pausada. Necesita verificar su información.",
    "verificar información de pago",
    "verificar"
]
actions = [
    "Por favor, inicie sesión para actualizar su información a través de:",
    "Urgente: Acceda a su cuenta en",
    "Evite la cancelación de su servicio visitando:",
    "Inicie sesión en",
    "Revise su información en el siguiente enlace:",
    "Acceda a su info en:"
]
urls = ["netfIix-micuenta.com", "http://217.160.114.154", "http://217.160.114.154/redirection", "http://217.160.114.154/pages/special_char_а.html", "http://truchodomain.info/", "http://truchodomain.info/redirection", "http://truchodomain.info/pages/special_char_а.html"]

# Lista para almacenar las variaciones generadas
variations = []

# Generar 1000 variaciones del mensaje
for _ in range(1000):
    intro = random.choice(introductions)
    reason = random.choice(reasons)
    action = random.choice(actions)
    url = random.choice(urls)  
    complete_message = f"{intro} {reason} {action} {url}"
    variation = {"text": complete_message, "is_smsing": 1}
    variations.append(variation)

# Opcional: Imprimir las primeras 5 variaciones para verificar
for variation in variations[:5]:
    print(variation)
