import pandas as pd
import numpy as np
# import sklearn
import pandas as pd
import os
# import matplotlib.pyplot as plt
import math
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn import metrics
# from collections import defaultdict

# from matplotlib import *
# import sys
# from matplotlib import offsetbox
# from sklearn import (manifold, datasets, decomposition, ensemble,
#                      discriminant_analysis, random_projection)
# # from sklearn import metrics
# from sklearn.cross_validation import cross_val_score, StratifiedKFold
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
import logging
log = logging.getLogger(__name__)
logging.basicConfig(filename='test_script.log', filemode='w', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

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
    elif entry in disqual_list: #add a .lower() or two in here?
        return False
    else:
        return True

def int_erate(num):
    str(num).replace(".", "").replace(" ", "")
#     try:
    return float(num)
#     except:
#         return num


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

cdc_99_14 = pd.concat([cdc_2014, cdc_2013, cdc_2012, cdc_2011, cdc_2010, cdc_2009, cdc_2008, cdc_2007, cdc_2006, cdc_2005, cdc_2004, cdc_2003, cdc_2002, cdc_2001, cdc_2000, cdc_1999], axis=0)

cdc_99_14.drop(['Notes'], inplace=1) #need to fix this to drop notes


def county_clean(code):
    code = str(code)
    code = code.split(".")[0]
    return code


def state_extract(county):
    county = str(county)
    try:
        county = county.split(",")[1]
        county = county.replace(" ", "")
        return county
    except:
        if len(county) == 2:
            return county
        else:
            return "ERROR"


cdc_99_14['county_id'] = cdc_99_14['County Code'].apply(county_clean)
cdc_99_14['state'] = cdc_99_14['County'].apply(state_extract)

def county_cleaner(county):
    county = str(county).split(".")[0]
    if len(county) == 4:
        county = '0' + county
    return county

cdc_99_14['county_code'] = cdc_99_14['County Code'].apply(county_cleaner)
cdc_99_14['CTYNAME'] = cdc_99_14['County'].apply(lambda x: x.split(",")[0].strip(" "))
cdc_99_14['year'] = cdc_99_14['year'].apply(lambda x: str(x))
cdc_99_14['pop_merge_ind'] = cdc_99_14['CTYNAME'] + "_" + cdc_99_14['year'] #this no longer useful b/c had to groupby before using this merge method


####     Clean & merge **unemployment data ***

def unemp_col_clean(fig):
    fig = str(fig)
    fig = fig.replace(",", "").strip(" ")
    try:
        return float(fig)
    except:
        return np.nan

def unemp_fix(df):
    unemp_cols = ['unknown_code', 'state_code', 'cout_code', 'county_state', 'year', 'nan', 'working_pop', 'employed_pop', 'unemployed_pop', 'unemp_perc']
    df = df.iloc[3:,0:]
    df.columns= unemp_cols
    df = df[2:]
    df['county_code'] = df['state_code'] + df['cout_code']# need to clean this
    df['county_code'] = df['county_code'].apply(lambda x: str(x))
    df['working_pop'] = df['working_pop'].apply(unemp_col_clean)
    df['employed_pop'] = df['employed_pop'].apply(unemp_col_clean)
    df['unemployed_pop'] = df['unemployed_pop'].apply(unemp_col_clean)
    df['unemp_perc'] = df['unemp_perc'].apply(unemp_col_clean)
    return df



unemp_03 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_03.csv"))
unemp_04 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_04.csv"))
unemp_05 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_05.csv"))
unemp_06 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_06.csv"))
unemp_07 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_07.csv"))
unemp_08 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_08.csv"))
unemp_09 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_09.csv"))
unemp_10 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_10.csv"))
unemp_11 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_11.csv"))
unemp_12 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_12.csv"))
unemp_13 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_13.csv"))
unemp_14 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_14.csv"))
unemp_15 = unemp_fix(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/unemp_15.csv"))

unemp_tot = pd.concat([unemp_03,unemp_04, unemp_05,unemp_06, unemp_07,unemp_08, unemp_09,unemp_10,unemp_11,unemp_12,unemp_13,unemp_14], axis=0)
unemp_tot.dropna(axis=0, subset=['year', 'county_code', 'working_pop', 'employed_pop','unemployed_pop','unemp_perc'], inplace=1)

def state_code_zero(item):
    item = str(item)
    if item[0] == '0':
        return item[1]
    else:
        return item

#need to create this dict for merging the population data later

state_code_convert = unemp_tot.loc[:,['state_code', 'county_state']]
state_code_convert['state_code'] = state_code_convert['state_code'].apply(state_code_zero)
state_code_convert.index=state_code_convert['state_code']
state_code_convert['state_abbrev'] = state_code_convert['county_state'].apply(state_extract)
# state_code_convert['state_abbrev'].value_counts() #only 12 errors appear to be DC - can proceed
state_code_convert = state_code_convert.loc[:,['state_code', 'state_abbrev']]
state_code_convert['state_code'] = state_code_convert['state_code'].apply(lambda x: int(x)) #should be an int for merging
state_code_convert = state_code_convert.to_dict()
state_code_convert = state_code_convert['state_abbrev']
state_code_convert['11'] = 'DC' #had an error with DC


###    merging POVERTY data below


def pov_col_clean(fig):
    fig = str(fig)
    fig = fig.replace(",", "").strip(" ")
    try:
        return float(fig)
    except:
        return 0.0


def pov_clean(df):
    correct_pov_cols = ['State FIPS Code',
 'County FIPS Code',
 'Postal Code',
 'Name',
 'Poverty Estimate, All Ages',
 '90% CI Lower Bound',
 '90% CI Upper Bound',
 'Poverty Percent, All Ages',
 '90% CI Lower Bound',
 '90% CI Upper Bound',
 'Poverty Estimate, Age 0-17',
 '90% CI Lower Bound',
 '90% CI Upper Bound',
 'Poverty Percent, Age 0-17',
 '90% CI Lower Bound',
 '90% CI Upper Bound',
 'Poverty Estimate, Age 5-17 in Families',
 '90% CI Lower Bound',
 '90% CI Upper Bound',
 'Poverty Percent, Age 5-17 in Families',
 '90% CI Lower Bound',
 '90% CI Upper Bound',
 'Median Household Income',
 '90% CI Lower Bound',
 '90% CI Upper Bound',
 'Poverty Estimate, Age 0-4',
 '90% CI Lower Bound',
 '90% CI Upper Bound',
 'Poverty Percent, Age 0-4',
 '90% CI Lower Bound',
 '90% CI Upper Bound']
    df = df.iloc[2:,:]
    pov_cols = df.iloc[0]
    df = df.iloc[1:,:]
    df.columns = correct_pov_cols
    df['county_code'] = df['State FIPS Code'] + df['County FIPS Code']
    df['county_code'] = df['county_code'].apply(lambda x: str(x))
    df['pov_count_tot'] = df['Poverty Estimate, All Ages'].apply(lambda x: pov_col_clean(x))
    df['pov_rate_tot'] = df['Poverty Percent, All Ages'].apply(lambda x: pov_col_clean(x))
    df['pov_rate_tot'] = df['Poverty Percent, All Ages'].apply(lambda x: pov_col_clean(x))
    df['pov_youth_count_0-17'] = df['Poverty Estimate, Age 0-17'].apply(lambda x: pov_col_clean(x))
    df['pov_youth_rate_0-17'] = df['Poverty Percent, Age 0-17'].apply(lambda x: pov_col_clean(x))
    df['med_hh_income'] = df['Median Household Income'].apply(lambda x: pov_col_clean(x))
    return df

pov_14 = pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_14.csv")
pov_14 = pov_clean(pov_14)


pov_03 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_03.csv"))
pov_04 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_04.csv"))
pov_05 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_05.csv"))
pov_06 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_06.csv"))
pov_07 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_07.csv"))
pov_08 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_08.csv"))
pov_09 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_09.csv"))
pov_10 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_10.csv"))
pov_11 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_11.csv"))
pov_12 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_12.csv"))
pov_13 = pov_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pov_13.csv"))

pov_03['year'] = '2003'
pov_04['year'] = '2004'
pov_05['year'] = '2005'
pov_06['year'] = '2006'
pov_07['year'] = '2007'
pov_08['year'] = '2008'
pov_09['year'] = '2009'
pov_10['year'] = '2010'
pov_11['year'] = '2011'
pov_12['year'] = '2012'
pov_13['year'] = '2013'
pov_14['year'] = '2014'

# print(len(pov_03), len(pov_04), len(pov_05),len(pov_06),len(pov_07),len(pov_08),len(pov_09),len(pov_10),len(pov_11),len(pov_12),len(pov_13),len(pov_14))#insert more loaded data here

pov_03_14 = pd.concat([pov_03, pov_04, pov_05,pov_06,pov_07,pov_08,pov_09,pov_10,pov_11,pov_12,pov_13,pov_14], axis=0) #insert more loaded data here
# pov_03_14 = pov_03_14.iloc[:3191,-7:]

pov_03_14.dropna(axis=0, subset=['year','pov_count_tot', 'pov_rate_tot','pov_youth_rate_0-17','med_hh_income'], inplace=1)

#pov_03_14 is being moved below for merging


##Merging poplation/ demographic data below
def age_grp_clean(age):
    if age == 0:
        return 'total'
    if age == 1:
        return 'age_0-4'
    if age == 2:
        return 'age_5-14'
    if age == 3:
        return 'age_5-14'
    if age == 4:
        return 'age_15-24'
    if age == 5:
        return 'age_15-24'
    if age == 6:
        return 'age_25-34'
    if age == 7:
        return 'age_25-34'
    if age == 8:
        return 'age_35-44'
    if age == 9:
        return 'age_35-44'
    if age == 10:
        return 'age_45-54'
    if age == 11:
        return 'age_45-54'
    if age == 12:
        return 'age_55-64'
    if age == 13:
        return 'age_55-64'
    if age == 14:
        return 'age_65-74'
    if age == 15:
        return 'age_65-74'
    if age == 16:
        return 'age_75-84'
    if age == 17:
        return 'age_75-84'
    if age == 18:
        return 'age_85+'

def year_clean(year):
    if year == 1:
        return '2010_census'
    elif year == 2:
        return '2010_pop_est_base'
    elif year == 3:
        return '2010'
    elif year == 4:
        return '2011'
    elif year == 5:
        return '2012'
    elif year == 6:
        return '2013'
    elif year == 7:
        return '2014'
    elif year == 8:
        return '2015'


def pop_df_clean(df):
    # df['age'] = df['AGEGRP'].apply(lambda x:age_grp_clean(x))
    df['year'] = df['YEAR'].apply(lambda x: year_clean(x))
    df['AGEGRP'] = df['AGEGRP'].apply(lambda x: int(x))
    df['pop_sub_15'] = df['TOT_POP'] * df['AGEGRP'].apply(lambda x: 1 if x in (1,2,3) else 0)
    df['pop_15-34'] = df['TOT_POP'] * df['AGEGRP'].apply(lambda x: 1 if x in (4,5,6,7) else 0)
    df['pop_35-54'] = df['TOT_POP'] * df['AGEGRP'].apply(lambda x: 1 if x in (8,9,10,11) else 0)
    df['pop_55+'] = df['TOT_POP'] * df['AGEGRP'].apply(lambda x: 1 if x >= 12 else 0)
    df['pop_black'] = df['BAC_MALE'] +  df['BAC_FEMALE']
    df['pop_white'] = df['WA_MALE'] +  df['WA_FEMALE']
    df['pop_hisp'] = df['H_MALE'] +  df['H_FEMALE']
    return df

pop_15 = pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pop_data_15.csv")

pop_15['age'] = pop_15['AGEGRP'].apply(lambda x:age_grp_clean(x))
pop_15['year'] = pop_15['YEAR'].apply(lambda x: year_clean(x))
pop_15['AGEGRP'] = pop_15['AGEGRP'].apply(lambda x: int(x))
pop_15['pop_sub_15'] = pop_15['TOT_POP'] * pop_15['AGEGRP'].apply(lambda x: 1 if x in (1,2,3) else 0)
pop_15['pop_15-34'] = pop_15['TOT_POP'] * pop_15['AGEGRP'].apply(lambda x: 1 if x in (4,5,6,7) else 0)
pop_15['pop_35-54'] = pop_15['TOT_POP'] * pop_15['AGEGRP'].apply(lambda x: 1 if x in (8,9,10,11) else 0)
pop_15['pop_55+'] = pop_15['TOT_POP'] * pop_15['AGEGRP'].apply(lambda x: 1 if x >= 12 else 0)
pop_15['pop_black'] = pop_15['BAC_MALE'] +  pop_15['BAC_FEMALE']
pop_15['pop_white'] = pop_15['WA_MALE'] +  pop_15['WA_FEMALE']
pop_15['pop_hisp'] = pop_15['H_MALE'] +  pop_15['H_FEMALE']
pop_15['pop_asian'] = pop_15['AA_MALE'] +  pop_15['AA_FEMALE']
pop_15['pop_merge_ind'] = pop_15['CTYNAME'] + "_" + pop_15['year']

pop_10 = pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pop_data_10.csv")
# pop_10.head(20)
pop_10_age = pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pop_10_age_sex.csv")
# pop_test = pop_df_clean(pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pop_mass.csv"))

# tri= [(pop_04, 'POPESTIMATE2004', 2004),(pop_05, 'POPESTIMATE2005', 2005)]

def pop_clean_analyze(pop_10, pop_10_age, df_name, col, year):
    df_name = pop_10_age[['CTYNAME', 'STNAME', 'STATE', 'COUNTY', 'SEX', 'AGEGRP', col]]
    pop_race = pop_10[['CTYNAME', 'STNAME', 'STATE', 'COUNTY', 'ORIGIN', 'RACE', col]]
    df_name = pd.merge(df_name, pop_race, how='left', on=['COUNTY', 'STATE'], copy=False)
    df_name['year'] = year
    df_name.columns = ['CTYNAME', 'STNAME_x', 'STATE', 'COUNTY', 'SEX', 'AGEGRP','population_age_sex', 'del', 'del', 'ORIGIN', 'RACE','population_race_hisp', u'year']
    df_name['population_age_sex'] = df_name['population_age_sex'].apply(lambda x: float(x))
    df_name['population_race_hisp'] = df_name['population_race_hisp'].apply(lambda x: float(x))
    df_name['sex'] = df_name['SEX'].apply(lambda x: int(x))
    df_name['hisp'] = df_name['ORIGIN'].apply(lambda x: int(x))
    df_name['race'] = df_name['RACE'].apply(lambda x: int(x))
    df_name['year_str'] = df_name['year'].apply(lambda x: str(x))
    df_name['pop_merge_ind'] = df_name['CTYNAME'] + "_" + df_name['year_str']
    df_name['age_group'] = df_name['AGEGRP'].apply(lambda x: int(x))
    df_name['pop_sub_15'] = df_name['population_age_sex'] * df_name['age_group'].apply(lambda x: 1 if x in (1,2,3) else 0)
    df_name['pop_15-34'] = df_name['population_age_sex'] * df_name['age_group'].apply(lambda x: 1 if x in (4,5,6,7) else 0)
    df_name['pop_35-54'] = df_name['population_age_sex'] * df_name['age_group'].apply(lambda x: 1 if x in (8,9,10,11) else 0)
    df_name['pop_55+'] = df_name['population_age_sex'] * df_name['age_group'].apply(lambda x: 1 if x >= 12 else 0)
    df_name['pop_male'] = df_name['population_age_sex'] * df_name['sex'].apply(lambda x: 1 if x == 1 else 0)
    df_name['pop_white'] = df_name['population_race_hisp'] * df_name['race'].apply(lambda x: 1 if x == 1 else 0)
    df_name['pop_black'] = df_name['population_race_hisp'] * df_name['race'].apply(lambda x: 1 if x == 2 else 0)
    df_name['pop_hisp'] = df_name['population_race_hisp'] * df_name['hisp'].apply(lambda x: 1 if x == 1 else 0)
    df_name['pop_asian'] = df_name['population_race_hisp'] * df_name['race'].apply(lambda x: 1 if x == 4 else 0)
    return df_name


pop_03 = pd.DataFrame()
pop_03 = pop_clean_analyze(pop_10, pop_10_age, pop_03, 'POPESTIMATE2003', 2003)

pop_04 = pd.DataFrame()
pop_04 = pop_clean_analyze(pop_10, pop_10_age, pop_04, 'POPESTIMATE2004', 2004)

pop_05 = pd.DataFrame()
pop_05 = pop_clean_analyze(pop_10, pop_10_age, pop_05, 'POPESTIMATE2005', 2005)

pop_06 = pd.DataFrame()
pop_06 = pop_clean_analyze(pop_10, pop_10_age, pop_06, 'POPESTIMATE2006', 2006)

pop_07 = pd.DataFrame()
pop_07 = pop_clean_analyze(pop_10, pop_10_age, pop_07, 'POPESTIMATE2007', 2007)

pop_08 = pd.DataFrame()
pop_08 = pop_clean_analyze(pop_10, pop_10_age, pop_08, 'POPESTIMATE2008', 2008)

pop_09 = pd.DataFrame()
pop_09 = pop_clean_analyze(pop_10, pop_10_age, pop_09, 'POPESTIMATE2009', 2009)

def state_coder_gen(df):
    a= df['STATE']
    b= df['COUNTY']
    a,b = str(a), str(b)
    if len(a) < 2:
        a = "0" + a
    while len(b) < 3:
        b = "0" + b
    df['county_code'] = a+b
    df['county_code'].apply(lambda x: str(x))
    return df

pop_03 = state_coder_gen(pop_03)
pop_04 = state_coder_gen(pop_04)
pop_05 = state_coder_gen(pop_05)
pop_06 = state_coder_gen(pop_06)
pop_07 = state_coder_gen(pop_07)
pop_08 = state_coder_gen(pop_08)
pop_09 = state_coder_gen(pop_09)
pop_15 = state_coder_gen(pop_15)


#THIS IS STILL PRE GROUPBY -


pop_03_g = pop_03.groupby(['year', 'county_code'], axis=0).sum()
pop_04_g = pop_04.groupby(['year', 'county_code'], axis=0).sum()
pop_05_g = pop_05.groupby(['year', 'county_code'], axis=0).sum()
pop_06_g = pop_06.groupby(['year', 'county_code'], axis=0).sum()
pop_07_g = pop_07.groupby(['year', 'county_code'], axis=0).sum()
pop_08_g = pop_08.groupby(['year', 'county_code'], axis=0).sum()
pop_09_g = pop_09.groupby(['year', 'county_code'], axis=0).sum()
pop_10_15 = pop_15.groupby(['year', 'county_code'], axis=0).sum()

pop_03_15 = pd.concat([pop_03_g, pop_04_g, pop_05_g, pop_06_g, pop_07_g, pop_08_g, pop_09_g, pop_10_15]) #this works 1) need to make floats 2) need more data


pop_03_15['pop%_men'] = pop_03_15['pop_male']/  pop_03_15['population_age_sex']
pop_03_15['pop%_sub_15'] = pop_03_15['pop_sub_15']/ pop_03_15['population_age_sex']
pop_03_15['pop%_15-34'] = pop_03_15['pop_15-34']/ pop_03_15['population_age_sex']
pop_03_15['pop%_35-54'] = pop_03_15['pop_35-54']/ pop_03_15['population_age_sex']
pop_03_15['pop%_55+'] = pop_03_15['pop_55+']/ pop_03_15['population_age_sex']
pop_03_15['pop%_black'] = pop_03_15['pop_black']/ pop_03_15['population_race_hisp']
pop_03_15['pop%_white'] = pop_03_15['pop_white']/ pop_03_15['population_race_hisp']
pop_03_15['pop%_hisp'] = pop_03_15['pop_hisp']/ pop_03_15['population_race_hisp']
pop_03_15['pop%_asian'] = pop_03_15['pop_asian']/ pop_03_15['population_race_hisp']


## Creating target groupby for merging

deaths_pop = cdc_99_14.groupby(by=['year', 'county_code']).sum()
deaths_pop.reset_index(inplace=1)
deaths_pop.to_csv('deaths_pop_inc.csv')


#MERGING POPULATION DATA

# # -- already grouped by year, county name immediately above
pop_03_15.reset_index(inplace=1)
pop_03_15.to_csv('pop_03_15.csv')

deaths_pop = pd.merge(deaths_pop, pop_03_15, how='left', left_on=['year','county_code'], right_on=['year','county_code'])



#UNEMPLOYMENT DATA groupby & reset


unemp_tot = unemp_tot.groupby(by=['year', 'county_code']).sum()
unemp_tot.reset_index(inplace=1)
unemp_tot.to_csv('unemp_tot_03_14.csv')

deaths_pop = pd.merge(deaths_pop, unemp_tot, how='left', left_on=['year', 'county_code'], right_on=['year', 'county_code'])


#POVERTY DATA

pov_county_year = pov_03_14.groupby(by=['year', 'county_code']).mean()
pov_county_year.reset_index(inplace=1)
pov_county_year.to_csv('pov_county_year_03_14.csv')


deaths_pop = pd.merge(deaths_pop, pov_county_year, how='left', left_on=['year','county_code'], right_on=['year','county_code'])

#CALCULATIONS REQUIRING UNEMP & CDC

deaths_pop['perc_pop_employed'] = deaths_pop['employed_pop']/deaths_pop['population']
deaths_pop['unemp_rate'] = deaths_pop['employed_pop']/deaths_pop['working_pop']
deaths_pop['perc_pop_not_working'] = (deaths_pop['population'] -deaths_pop['working_pop'])/deaths_pop['population']

deaths_pop.to_csv('drug_mortality_03_14.csv')

print(deaths_pop.head())




#STILL NEED TO MAKE COUNTY_STATE FOR  pop_15

#
