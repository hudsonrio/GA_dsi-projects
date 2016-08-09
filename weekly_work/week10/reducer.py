#!/Users/HudsonCavanagh/anaconda/bin/python

from collections import defaultdict
import sys
import pandas as pd


def reduction(dis_list):
    reduce_count = defaultdict(int)
    for item in dis_list:
        reduce_count[item] += 1
    # reduce_count = pd.DataFrame.from_dict(reduce_count)
    df = pd.DataFrame(reduce_count.iteritems())
    df = sorted(df) #fix
    return(df)


item_list = []
for line in sys.stdin:
    item_list.append(line)


reduction(item_list)
