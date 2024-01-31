import url_library
redirection = "http://217.160.114.154/redirection"
special_char = "http://217.160.114.154/pages/special_char_а.html"
value = url_library.make_redirection(redirection)
print(value)

value = url_library.special_characters(special_char)
print(value)
