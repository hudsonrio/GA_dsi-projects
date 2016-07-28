#!/usr/bin/env python
# coding:'utf-8'

import pandas as pd
import numpy as np
import sys
import sklearn
import counter
import collections
from collections import defaultdict
import re
from re import split



# lines = sys.stdin.readlines()
master_list = []
word_dict = {}

for line in sys.stdin:
    line = line.lower()
    for lin in line.split(" "):
        lin = lin.strip('"/').replace(" ", "").replace(".", "").replace("'", "").replace("?", "").replace("!", "").replace('"', "").replace(',', "")
        if lin in word_dict.keys():
            word_dict[lin]=+1
        else:
            word_dict[lin]=1

for w, n in sorted(word_dict.items()):
    print '%s\t%s' %(w,n)
#
# df = pd.DataFrame(word_dict.items())
# return (df)


#
# cvec_counter(master_list)
#



#
# def cvec_counter(words):
#     cvec = CountVectorizer()
#     cvec.fit(words)
#     language = pd.DataFrame(cvec.transform(words).todense(),
#                        columns=cvec.get_feature_names())
#     word_counts = language.sum(axis=0)
#     counts = word_counts.sort_values(ascending = False)
#     return(counts)
