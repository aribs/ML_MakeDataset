import url_library
bad_url = "http://malware.testing.google.test/testing/malware/"
good_url = "https://www.google.com"
special_url = "http://malware.testing.googвle.test/testing/malware/"



is_domain_listed = url_library.get_domain_safe_info(bad_url)
print(is_domain_listed)

domain_age  = url_library.get_domain_time_info(good_url)
print(domain_age)

#isIp = url_library.is_ip(bad_url)
#print(isIp)

#specialCharacter = url_library.special_characters(bad_url)
#print(specialCharacter)

url = url_library.get_url("texto de prueba1 " + special_url + "texto de prueba2")
print(url)
