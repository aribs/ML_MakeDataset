import url_library
import pandas as pd


bad_url = "http://malware.testing.google.test/testing/malware/"
good_url = "https://www.google.com"
special_url = "http://malware.testing.googвle.test/testing/malware/"

messageList = [{"text": "NETFLIX: Su suscripcion ha sido suspendida, inicie sesion para actualizar su informacion a través de: netfIix-micuenta.com", "is_smsing": 1}, 
               {"text": "AGENCIA TRIBUTARIA:su ejercicio anterior 2022/2023 a resultado favorable para usted en 286,84€ para recibir la devolucion antes del13/07/2023 pulse aqui ref076589.eu", "is_smsing": 1},
               {"text": "Paga la cuenta sin moverte de la silla, usando TheFork PAY www.thefork.es/pay?utm_medium=sms&utm_source=TFPay", "is_smsing": 0}]

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



for message in messageList:
    lineCsv = {}
    url = url_library.get_url(message["text"])
    is_smsing = message["is_smsing"]
    lineCsv["message"] = message["text"]
    lineCsv["is_smsing"] = is_smsing
    if(url):
        datasetUrl = setUrlDataset(url)
        lineDataset = {**lineCsv, **datasetUrl}
        print(lineDataset)
        





    