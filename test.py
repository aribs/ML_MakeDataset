import url_library
redirection = "http://217.160.114.154/redirection"
special_char = "http://217.160.114.154/pages/special_char_а.html"
bad_url = "http://malware.testing.google.test/testing/malware/"
good_url = "https://www.google.com"
trucho = "http://netfIix-micuenta.com"
prueba1 = "ref076589.eu"
value = url_library.make_redirection(redirection)
print(value)

value = url_library.special_characters(special_char)
print(value)

value = url_library.get_whois(prueba1)
print(value)

