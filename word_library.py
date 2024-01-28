import string

#Cuenta las coincidencias totales en un string, comparando con un array de diccionario
def count_words(text, dictionary):
    text = remove_punctuation(text)
    words = text.lower().split()   
    counter = {word: 0 for word in dictionary }
    for word in words:
        print(word)
        if word in dictionary:
            counter[word] += 1
    total_matches = sum(counter.values())
    return total_matches

#Aux Functions
#Borra los símbolos de puntuación de un string
def remove_punctuation(text):
    return ''.join(char if char not in string.punctuation else ' ' for char in text)

