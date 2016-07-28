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
import my_cleaner

# lines = sys.stdin.readlines()
master_list = []
word_dict = {}
d = defaultdict(str)

for line in sys.stdin:
    for word in line.split(" "):
        word = my_cleaner.cleaner(word)
        d[word]=+1

for w, n in sorted(d.items()):
    print '%s\t%s' %(word,number_count)
