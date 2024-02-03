import url_library
import word_library
import csv



bad_url = "http://malware.testing.google.test/testing/malware/"
good_url = "https://www.google.com"
special_url = "http://malware.testing.googвle.test/testing/malware/"

dictionary = ["netflix", "tributaria", "inicie", "silla"]

messageList = [{"text": "NETFLIX: Su suscripcion ha sido NETFLIX suspendida, inicie sesion para actualizar su informacion a través de: netfIix-micuenta.com", "is_smsing": 1}, 
               {"text": "AGENCIA TRIBUTARIA:su ejercicio anterior 2022/2023 a resultado favorable para usted en 286,84€ para recibir la devolucion antes del13/07/2023 pulse aqui ref076589.eu", "is_smsing": 1},
               {"text": "Paga la cuenta sin moverte de la silla, usando TheFork netflix PAY www.thefork.es/pay?utm_medium=sms&utm_source=TFPay", "is_smsing": 0},
               {'text': 'Acceso Externo a su cuenta Netflix verificar información de pago Evite la cancelación de su servicio visitando: http://truchodomain.info/redirection', 'is_smsing': 1},
                {'text': 'Acceso Externo a su cuenta Netflix bloqueada por seguridad. Evite la cancelación de su servicio visitando: http://217.160.114.154/pages/special_char_а.html', 'is_smsing': 1},
                {'text': 'Cancelación de su Cuenta Netflix temporalmente suspendida. Urgente: Acceda a su cuenta en http://truchodomain.info/pages/special_char_а.html', 'is_smsing': 1},
                {'text': 'Detectado acceso a su cuenta Netflix bloqueada por seguridad. Evite la cancelación de su servicio visitando: http://217.160.114.154', 'is_smsing': 1},
                {'text': 'Alerta NETFLIX: Detectamos un problema con su cuenta, está temporalmente suspendida. Acceda a su info en: http://truchodomain.info/redirection', 'is_smsing': 1}]
               

claves_csv = ["message", "is_smsing", "safe_info", "domain_old_info", "is_ip", "url_length", "number_of_subdomains", "is_https", "special_characters", "tlds_blackist", "make_redirection", "creation_date", "updated_date", "is_suspended", "count_dictionary"]
def setUrlDataset(url):
    return {
        "safe_info": url_library.get_domain_safe_info(url),
        "domain_old_info": url_library.get_domain_time_info(url),
        "is_ip": url_library.is_ip(url),
        "url_length": url_library.url_length(url),
        "number_of_subdomains": url_library.number_of_subdomains(url),
        "is_https": url_library.is_https(url),
        "special_characters": url_library.special_characters(url),
        "tlds_blackist": url_library.tlds_blacklist(url),
        "make_redirection": url_library.make_redirection(url)
    }

with open('data.csv', mode='w', newline='') as dataFile:
    writer_csv = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer_csv.writerow(claves_csv)
    for message in messageList:
        lineCsv = {}
        url = url_library.get_url(message["text"])
        is_smsing = message["is_smsing"]
        lineCsv["message"] = message["text"]
        lineCsv["is_smsing"] = is_smsing
        if(url):
            datasetUrl = setUrlDataset(url)
            whoisInfo = url_library.get_whois(url)
        count_dictionary = {"count_dictionary": word_library.count_words(message["text"], dictionary)}
        lineDataset = {**lineCsv, **datasetUrl, **whoisInfo, **count_dictionary}
        print("\n", lineDataset)
        valores_en_orden = [lineDataset.get(clave) for clave in claves_csv]
        writer_csv.writerow(valores_en_orden)
        
        print("coincidence words ", count_dictionary)
        





    