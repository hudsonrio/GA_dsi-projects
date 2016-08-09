#!/Users/HudsonCavanagh/anaconda/bin/python
import string
import sys



def word_count(inputs):
    lista = []
    inputs = inputs.strip(string.punctuation)
    inputs = inputs.lower().split(" ")
    for i in inputs:
        if i != None:
            lista.append((i,int(1)))
    print(lista)



for line in sys.stdin:
    word_count(line)
