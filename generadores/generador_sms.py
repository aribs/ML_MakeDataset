import json
import random
numMensajesByPacket = 500
def generate_messages(is_smishing, introductions, reasons, actions):
    # Lista para almacenar las variaciones generadas
    variations = []

    # Generar 1000 variaciones del mensaje
    for _ in range(numMensajesByPacket):
        intro = random.choice(introductions)
        reason = random.choice(reasons)
        action = random.choice(actions)
        url = random.choice(urls)  
        complete_message = f"{intro} {reason} {action} {url}"
        variation = {"text": complete_message, "is_smsing": is_smishing}
        variations.append(variation)

    # Ruta del archivo de salida
    output_netflix_path = "output_generated.json"

    try:
        # Intentar abrir el archivo existente para añadir las nuevas variaciones
        with open(output_netflix_path, "r+", encoding="utf-8") as file:
            existing_data = json.load(file)
            existing_data.extend(variations)
            file.seek(0)
            json.dump(existing_data, file, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        # Si el archivo no existe, crear uno nuevo y escribir las variaciones
        with open(output_netflix_path, "w", encoding="utf-8") as file:
            json.dump(variations, file, ensure_ascii=False, indent=4)

    # Opcional: Imprimir mensaje de confirmación
    print(f"Se han añadido {len(variations)} variaciones al archivo {output_netflix_path}.")

####MENSAJES TIPO NETFLIX
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

generate_messages(1, introductions, reasons, actions)

###MENSAJES TIPO BANCO
introductions = [
    "Se ha intentado cobrar 262,12",
    "Se le ha facturado en su cuenta 134,32",
    "Se ha registrado una compra con importe de 225,45",
    "nuevo cargo en su cuenta de 123,56",
    "Se le va a cargar en su cuenta el total de 12,23",
    "Se le han cobrado 143,22 por una compra en amazon ",
    "Tiene un aviso importante sobre su cuenta"
    "Compra efectiva 257,34€",
    "Compra aceptada AMAZON precio 332,32"
]
reasons = [
    "verificar información de pago",
    "verificar",
    "verificar compra",
]
actions = [
    "Por favor, inicie sesión para actualizar su información a través de:",
    "Urgente: Acceda a su cuenta en",
    "Inicie sesión en",
    "Revise su información en el siguiente enlace:",
    "Acceda a su info en:",
    "Si no reconoce este pago accdeda a la info en:"
]
urls = ["https://bbva.es-122.com/e?784458?", "http://217.160.114.154", "http://217.160.114.154/redirection", "http://217.160.114.154/pages/special_char_а.html", "http://truchodomain.info/", "http://truchodomain.info/redirection", "http://truchodomain.info/pages/special_char_а.html"]

generate_messages(1, introductions, reasons, actions)

###MENSAJES TIPO PAQUETERÍA
introductions = [
    "Intento de entrega",
    "Entrega EXPRESS",
    "Intento de entrega 2/2",
    "Te hemos intentado contactar para entrega",
    "SEUR entrega fallida",
    "DHL intento de entrega",
    "GLS Paquete a la espera",
    "Paquete a la espera",
    "Su paquete va a ser devuelto",
    "Envío en punto de entrega ",
    "Urgente, paquete no entregado",
    "Urgente, paquete en proceso de devolución",
    "AMAZOn, paquete en espera"
]
reasons = [
    "Confirma información",
    "Confirma datos",
    "Confirma para que no se devuelva el paquete",
    "Actualiza datos de entrega",
    "¡Evite que sea devuelto!",
    "Actualice sus datos de envío"
]
actions = [
    "Por favor, inicie sesión para actualizar su información a través de:",
    "Link",
    "Reclama paquete",
    "Tus datos",
    "Asegurate que no se devuelve el paquete",
    "Evite que sea reverso"
]
urls = ["https://ijarez.com/GaJb5q5", "uhapib.com/xuOthWt", "nfllf.info/48RQXwI", "ilorol.com/GQjn8ec", "http://217.160.114.154", "post.cfd/es","http://217.160.114.154/redirection", "http://217.160.114.154/pages/special_char_а.html", "http://truchodomain.info/", "http://truchodomain.info/redirection", "http://truchodomain.info/pages/special_char_а.html"]

#generate_messages(1, introductions, reasons, actions)

###MENSAJES TIPO RANDOM
introductions = [
    "Amazon entrega",
    "SORTEO GANADO",
    "Tienes un artículo esperando",
    "¡Felicidades! eres el ganador",
    "¡Felicidades! eres la ganadora",
    "¡Felicidades! eres el ganador de un sorteo",
    "¡Felicidades! eres la ganadora de un sorteo",
    "¡Felicidades! eres el ganador de una tarjeta AMAZON",
    "¡Felicidades! eres la ganadora de una tarjeta AMAZON",
    "SORTEO AMAZON",
    "Agencia Tributaria, tienes un aviso",
    "Pago Agencia Tributaria",
    "Aviso campaña Renta 2023",
    
]
reasons = [
    "Confirma información",
    "Revisa tus datos",
    "Confirma tus datos",
    "Confirma datos correctos",
    
]
actions = [
    "Por favor, inicie sesión para actualizar su información a través de:",
    "Inicia sesión en:",
    "Confirma datos en :"
    "Iniciar sesión: "
    "Confirma en: "
    
]
urls = ["https://ijarez.com/GaJb5q5", "uhapib.com/xuOthWt", "nfllf.info/48RQXwI", "netfIix-micuenta.com", "http://217.160.114.154", "http://217.160.114.154/redirection", "http://217.160.114.154/pages/special_char_а.html", "http://truchodomain.info/", "http://truchodomain.info/redirection", "http://truchodomain.info/pages/special_char_а.html"]

#generate_messages(1, introductions, reasons, actions)


###MENSAJES TIPO INOCUOS
introductions = [
    "¡Sorprendete esta Navidad en GAES",
    "Lo estabas esperando! ",
    "CAJAMAR",
    "Oferta Especial"
    "Descuento en compra",
    "Rebajas Online"     
]
reasons = [
    "Una MANTA y un CALENDARIO de REGALO al revisar tu audicion",
    "! Comprando uno de estos arboles de Navidad ",
    " Cuenta 360. Deja de darle vueltas a las cosas y dale una vuelta a tus ahorros",
    "Descuentos de hasta el 30%",
    "Descuentazos sólo online",
    "Sólo hoy"
   
]
actions = [
    "Hasta 15/12/23. Cita",
    "Inicia sesión en:",
    "Confirma datos en :"
    "Iniciar sesión: "
    "Confirma en: "
    "tendras un -15% en adornos y figuras. ¡Solo hasta 10 Diciembre!",
    "Entra en:",
    "Compra en: "
    
]
urls = ["https://lmes.es/arb3", "http://t6.mc.amplifon.com/r/?id=s1d5f83f7,360f71ad,10cc991d", "https://www.decathlon.es/"]

numMensajesByPacket = 1000
generate_messages(0, introductions, reasons, actions)
