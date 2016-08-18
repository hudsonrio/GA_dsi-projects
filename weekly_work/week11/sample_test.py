def rectangle_area(w,h):
    return w*h

def strip_stopwords(phrase, stopwords):
    phrase = phrase.split()
    phrase = [w for w in phrase if w not in stopwords]
    phrase = ' '.join(phrase)
    return phrase

def strip_stopwords2(phrase, stopwords):
    phrase = phrase.split()
    phrase = (w for w in phrase if w not in stopwords)
    phrase = ' '.join(phrase)
    return phrase



this = 'purple hippos fail to understand yellow roosters'
stopwords2 = ['yellow', 'purple']

print(strip_stopwords(this, stopwords2))
print(strip_stopwords2(this, stopwords2))
