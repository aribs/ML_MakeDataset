
import yaml
import requests
import json
import re
from urllib.parse import urlparse

with open('./config/conf.yml', 'r') as file:
    config = yaml.safe_load(file)



#Extraer URL
def get_url(text):
    pattern = r'(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]\.[^\s]{2,})'
    urls = re.findall(pattern, text)
    if(urls):
        return urls[0]

#Api Safe Browsing 
#Pasamos una URL obtenemos de la api de safe browsing si está listada como maliciosa o no
def get_domain_safe_info(url):
    print(url)
    # URL de la API de Safe Browsing
    api_url = 'https://safebrowsing.googleapis.com/v4/threatMatches:find'
    # Parámetros de la petición
    payload = {
        'client': {
            'clientId': "tu_cliente",
            'clientVersion': "1.5.2"
        },
        'threatInfo': {
            'threatTypes':      ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            'platformTypes':    ["ANY_PLATFORM"],
            'threatEntryTypes': ["URL"],
            'threatEntries': [
                {"url": url}
            ]
        }
    }
    try:
        response = requests.post(api_url, params={'key': config["safe_browsing_api_key"]}, json=payload)
        response.raise_for_status()
        result = response.json()
        if result:
            return 1
    except requests.exceptions.HTTPError as err:
        print(f'Error en la petición HTTP GET DOMAIN SAFE: {err}')
    except Exception as e:
        print(f'Error: {e}')

#Antiguedad del Dominio
#Recibe un string con el nombre de dominio y hace una petición a la api ip2whois.com
def get_domain_time_info(domain):
    try:
        response = requests.get("https://api.ip2whois.com/v2", params={'key': config["doman_info_api_key"], "domain": domain})
        response.raise_for_status()
        result = response.json()
        if result["domain_age"]:
            return result["domain_age"]
    except requests.exceptions.HTTPError as err:
        print(f'Error en la petición HTTP: {err}')
    except Exception as e:
        print(f'Error: {e}')

#Es una IP
def is_ip(url):
    pattern = r'\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    if re.search(pattern, url):
        return 1
    else:
        return 0
    
#Url Length
def url_length(url):
    return len(url)

#Cantidad subdominios
def number_of_subdomains(url):
    domain = urlparse(url).netloc
    parts = domain.split('.')
    # La cantidad de subdominios será el número de partes menos 2 (dominio y TLD)
    # Pero puede variar si se trata de un TLD de segundo nivel como .co.uk
    if len(parts) > 2:
        # Aquí se pueden agregar condiciones para TLDs especiales
        if parts[-2] in ['co', 'com', 'org', 'net']:  # Ejemplo para TLDs de segundo nivel
            return len(parts) - 3
        else:
            return len(parts) - 2
    else:
        return 0
    
#HTTPS
def is_https(url):
    result = urlparse(url)
    if result.scheme == 'https':
        return 1

#Carácteres Especiales
def special_characters(url):
    pattern = re.compile(r"[^a-zA-Z0-9/:.?#&=_%-]")
    if pattern.search(url):
        return 1
    else:
        return 0

#Dominio Sospechoso
def tlds_blacklist(url):
    tlds_blacklist = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.info', '.ru', '.cn']
    tld = '.' + urlparse(url).netloc.split('.')[-1]
    return tld in tlds_blacklist

#Comprueba redireccionamiento
def make_redirection(url):
    try:
        response = requests.get(url)
        if response.history:
            return 1
        else:
            return 0
    except requests.RequestException as e:
        print(f"Error al realizar la solicitud: {e}")
        return 0
