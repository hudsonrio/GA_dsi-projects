

import pandas as pd
import json
import numpy as np
import logging
import os
print(os.getcwd())
log = logging.getLogger(__name__)


# with open("./log_config.json", "r") as fd:  #NOT GETTING THIS TO WORK
#     logging.config.dictConfig(json.load(fd))
logging.basicConfig(filename='test.log', filemode='w', format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


cdc_1999 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_1999.csv', delimiter='\t', encoding='utf-8')
cdc_2000 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2000.csv', delimiter='\t', encoding='utf-8')
cdc_2001 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2001.csv', delimiter='\t', encoding='utf-8')
cdc_2002 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2002.csv', delimiter='\t', encoding='utf-8')
cdc_2003 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2003.csv', delimiter='\t', encoding='utf-8')
cdc_2004 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2004.csv', delimiter='\t', encoding='utf-8')
cdc_2005 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2005.csv', delimiter='\t', encoding='utf-8')
cdc_2006 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2006.csv', delimiter='\t', encoding='utf-8')
cdc_2007 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2007.csv', delimiter='\t', encoding='utf-8')
log.info('Half of CDC files successfully read in')
cdc_2008 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2008.csv', delimiter='\t', encoding='utf-8')
cdc_2009 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2009.csv', delimiter='\t', encoding='utf-8')
cdc_2010 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2010.csv', delimiter='\t', encoding='utf-8')
cdc_2011 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2011.csv', delimiter='\t', encoding='utf-8')
cdc_2012 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2012.csv', delimiter='\t', encoding='utf-8')
cdc_2013 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2013.csv', delimiter='\t', encoding='utf-8')
cdc_2014_a = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2014_a-mississip.csv', delimiter='\t', encoding='utf-8')
cdc_2014_b = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/cdc_2014_missouri-z.csv', delimiter='\t', encoding='utf-8')
log.info('Target Data loaded')


def pct_clean(thing):
    thing = str(thing)
    x = thing.replace("%", "")
    return float(x)

def disqual(entry):
    disqual_list = ['nan', 'Not Applicable', np.NaN, 'NaN']
    if entry.isnull() == True:
        return False
    elif type(entry) == float:
        return True
    elif entry in disqual_list:
        return False
    else:
        return True

def int_erate(num):
    str(num).replace(".", "").replace(" ", "")
    return float(num)



#Workign with 2014 first, in pieces
cdc_2014_a = cdc_2014_a[cdc_2014_a['Deaths'].isnull() == False]
cdc_2014_b = cdc_2014_b[cdc_2014_b['Deaths'].isnull() == False]


cols_cdc =['Notes','County','County Code','Ten-Year Age Groups',
    'Ten-Year Age Groups Code','Gender','Gender Code', 'Race',
       'Race Code', 'Hispanic Origin', 'Hispanic Origin Code', 'Deaths',
       'Population', 'Crude Rate', '% of Total Deaths']


cdc_2014_a.columns = cols_cdc
cdc_2014_b.columns = cols_cdc

cdc_2014 = pd.concat([cdc_2014_a, cdc_2014_b], axis=0)

#succesfully merged, all dfs imported

cdc_years = (cdc_2014, cdc_2013, cdc_2012, cdc_2011, cdc_2010, cdc_2009, cdc_2008, cdc_2007, cdc_2006, cdc_2005, cdc_2004, cdc_2003, cdc_2002, cdc_2001, cdc_2000, cdc_1999)

#checking what percent of counties I am dropping below

pre_disqual = 0
post_disqual = 0

for check_pre in cdc_years:
    pre_disqual = pre_disqual + len(check_pre)


cdc_1999 = cdc_1999[cdc_1999['Deaths'].isnull() == False]
cdc_2000 = cdc_2000[cdc_2000['Deaths'].isnull() == False]
cdc_2001 = cdc_2001[cdc_2001['Deaths'].isnull() == False]
cdc_2002 = cdc_2002[cdc_2002['Deaths'].isnull() == False]
cdc_2003 = cdc_2003[cdc_2003['Deaths'].isnull() == False]
cdc_2004 = cdc_2004[cdc_2004['Deaths'].isnull() == False]
cdc_2005 = cdc_2005[cdc_2005['Deaths'].isnull() == False]
cdc_2006 = cdc_2006[cdc_2006['Deaths'].isnull() == False]
cdc_2007 = cdc_2007[cdc_2007['Deaths'].isnull() == False]
cdc_2008 = cdc_2008[cdc_2008['Deaths'].isnull() == False]
cdc_2009 = cdc_2009[cdc_2009['Deaths'].isnull() == False]
cdc_2010 = cdc_2010[cdc_2010['Deaths'].isnull() == False]
cdc_2011 = cdc_2011[cdc_2011['Deaths'].isnull() == False]
cdc_2012 = cdc_2012[cdc_2012['Deaths'].isnull() == False]
cdc_2013 = cdc_2013[cdc_2013['Deaths'].isnull() == False]

years = [2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]

lista = []

for ind, ex_df in enumerate([cdc_2014, cdc_2013, cdc_2012, cdc_2011, cdc_2010, cdc_2009, cdc_2008, cdc_2007, cdc_2006, cdc_2005, cdc_2004, cdc_2003, cdc_2002, cdc_2001, cdc_2000, cdc_1999]):
    ex_df['year'] = years[ind]
    ex_df['pct_total_deaths'] = ex_df['% of Total Deaths'].apply(lambda x: pct_clean(x))
    ex_df['deaths'] = ex_df['Deaths'].apply(lambda x: float(x))
    ex_df['population'] = ex_df['Population'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    ex_df['crude_100k'] = ex_df['deaths']/(ex_df['population']/100000)


cdc_1999.drop(cdc_1999.index[473], inplace=1)
cdc_2000.drop(cdc_2000.index[473], inplace=1)
cdc_2001.drop(cdc_2001.index[486], inplace=1)
cdc_2002.drop(cdc_2002.index[579], inplace=1)
cdc_2009.drop(cdc_2009.index[1049], inplace=1)

for check_post in cdc_years:
    post_disqual = post_disqual + len(check_post)

log.info("How many rows did I drop? Pre_clean: %s, post clen: %s", pre_disqual, post_disqual) #identical - do not appear to be dropping anything?

cdc_99_14 = pd.concat([cdc_2014, cdc_2013, cdc_2012, cdc_2011, cdc_2010, cdc_2009, cdc_2008, cdc_2007, cdc_2006, cdc_2005, cdc_2004, cdc_2003, cdc_2002, cdc_2001, cdc_2000, cdc_1999], axis=0)

cdc_99_14.drop(['Notes'], inplace=1) #need to fix this to drop notes

log.debug("number of lines in target dataframe: %s", len(cdc_99_14))
