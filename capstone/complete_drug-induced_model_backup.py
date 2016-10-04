
# coding: utf-8

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import r2_score
import sys
import matplotlib.patheffects as path_effects

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn import metrics
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel


# some helper functions

def pop_year_clean(year):
    year = str(year)
    year = year.replace(" ", "")
    try:
        year = int(year)
    except:
        year = year
    return year


def state_fix(code):
    code = str(code)
    if len(code) == 1:
        code = "0" + code
    return code

def county_fix(code):
    code = str(code)
    if len(code) == 1:
        code = "00" + code
    elif len(code) ==2:
        code = "0" + code
    return code

def col_includer(year, col):
    if year in col:
        return col
    else:
        return 0

#used for growth data
def growth_convert(df, year):
    final_cols = []
    df_cols = df.columns
    last_year = str(year - 1)
    year = str(year)
    df_cols_this = df_cols.map(lambda x: col_includer(year, x))
    df_cols_last = df_cols.map(lambda x: col_includer(last_year, x))

    #cleaning this year col
    df_cols_this = pd.DataFrame(df_cols_this)
    df_cols_this = df_cols_this[df_cols_this[0]!=0]
    final_cols.append(df_cols_this.iloc[0,0])
    #pulling last year's columns
    df_cols_last = pd.DataFrame(df_cols_last)
    df_cols_last = df_cols_last[df_cols_last[0]!=0]
    final_cols.append(df_cols_last.iloc[1,0])
    final_cols.append(df_cols_last.iloc[12,0])
    final_cols.append(df_cols_last.iloc[13,0])
    final_cols.append(df_cols_last.iloc[14,0])
    final_cols.append('STATE')
    final_cols.append('COUNTY')
    df = df[final_cols]
    df['STATE'] = df['STATE'].apply(lambda x: state_fix(x))
    df['COUNTY'] = df['COUNTY'].apply(lambda x: county_fix(x))
    df['string_county_code'] = df['STATE'] + df['COUNTY']
    df['county_code'] = df['string_county_code'].apply(lambda x: int(x))
    df['year'] = int(year)
    df.columns = ['population_est', 'net_pop_change_raw','natural_pop_growth_rate', 'intl_migrate_rate', 'dom_migrate_rate', 'state_num', 'county_num', 'string_county_code', 'county_code', 'year']
    return df

def yearify(year):
    year = float(year)
    try:
        return int(year)
    except:
        return year

def interator(example):
    try:
        example = str(example).split('.')[0]
        return int(example)
    except:
        return example

def death_clean(death):
    try:
        return float(death)
    except:
        return np.nan

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


def floater(df):
    for col in df:
        try:
            col = col.apply(lambda x: float(x))
        except:
            continue
    return df

def dict_applier(df):
    dict_ex = defaultdict(lambda: 0)
    for i in range(len(df)):
        dict_ex[df['county_code'][i]] +=1
        df['count_county_code'][i] = dict_ex[df['county_code'][i]]
    return df

# not needed unless producing visuals
# def plot_confusion_matrix(cm, title='Confusion matrix', labels=['Decreasing','Increasing'], cmap=plt.cm.Blues):
#
#    plt.figure(figsize=(7,7))
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#
#    tick_marks = np.arange(2)
#    plt.xticks(tick_marks, labels)
#    plt.yticks(tick_marks, labels)
#
#    plt.title(title)
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.colorbar()
#    plt.tight_layout()
#
#    width, height = cm.shape
#
#    for x in xrange(width):
#        for y in xrange(height):
#            plt.annotate(str(cm[x][y]), xy=(y, x),
#                        horizontalalignment='center',
#                        verticalalignment='center',
#                        color = 'white',
#                        fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
#                                                       path_effects.Normal()]) #The last line here adds a text outline



#reading in unemployment data

unemp_03_14 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/unemp_tot_03_14.csv')
unemp_03_14 = unemp_03_14.iloc[:,1:]

#reading in poverty data
pov_county_year_03_14 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pov_county_year_03_14.csv')
pov_county_year_03_14 = pov_county_year_03_14.iloc[:,1:]


#  name, state conversion for later on (need to merge Naxo features)

name_code_convert = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/deaths_simple.csv', delimiter='\t', encoding='utf-8')
name_code_convert = name_code_convert[['County Code', 'County']]
name_code_convert.columns = ['county_code', 'County']
name_code_convert['county_code'] = name_code_convert['county_code'].apply(lambda x: yearify(x))
name_code_convert.index=name_code_convert['county_code']
name_code_convert = name_code_convert.iloc[:,1:]
name_code_convert = name_code_convert.to_dict()
name_code_convert = name_code_convert['County']


#these files were large, had to be groupby'd in seperate scripts on EC2 before reading in
pop_03 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pop_03_output.csv')
pop_04 =pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pop_04_output.csv')
pop_05 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pop_05_output.csv')
pop_06 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pop_06_output.csv')
pop_07 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pop_07_output.csv')
pop_08 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pop_08_output.csv')
pop_09 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pop_09_output.csv')
pop_10_14 = pd.read_csv('/Users/HudsonCavanagh/Documents/csv_out_duplicate/pop_10_15_output.csv')

#standardizing formatting to merge
pop_10_14 = pop_10_14.loc[:,['year', 'county_code', 'pop_sub_15', 'pop_15-34', 'pop_35-54', 'pop_55+','pop_black','pop_white','pop_hisp','pop_asian', 'pop_male', 'population', 'TOT_POP']]

#merged population data
pop_03_14 = pd.concat([pop_03, pop_04, pop_05, pop_06, pop_07, pop_08, pop_09, pop_10_14], axis=0) #this works 1) need to make floats 2) need more data
pop_03_14['year'] = pop_03_14['year'].apply(lambda x: pop_year_clean(x)) #fixing integer conversion issue


### Making caluclations with pop DataFrame

pop_03_14['pop_total_age'] = pop_03_14['pop_sub_15'] + pop_03_14['pop_15-34'] + pop_03_14['pop_35-54'] + pop_03_14['pop_55+']
pop_03_14['pop_sub_15_prop'] = pop_03_14['pop_sub_15']/pop_03_14['pop_total_age']
pop_03_14['pop_15-34_prop'] = pop_03_14['pop_sub_15']/pop_03_14['pop_total_age']
pop_03_14['pop_35-54_prop'] = pop_03_14['pop_sub_15']/pop_03_14['pop_total_age']
pop_03_14['pop_55+_prop'] = pop_03_14['pop_sub_15']/pop_03_14['pop_total_age']
pop_03_14['pop_big_3_races'] = pop_03_14['pop_black'] + pop_03_14['pop_white'] + pop_03_14['pop_asian']
pop_03_14['pop_asian_prop'] = pop_03_14['pop_asian']/pop_03_14['pop_big_3_races']
pop_03_14['pop_white_prop'] = pop_03_14['pop_white']/pop_03_14['pop_big_3_races']
pop_03_14['pop_black_prop'] = pop_03_14['pop_black']/pop_03_14['pop_big_3_races']
# pop_03_14['pop_hisp_prop'] = pop_03_14['pop_hisp']/pop_03_14['pop_big_3_races'] #don't have good relative comparison
#excluded hispanic status is not mutually exclusive, no denominator to compare to &
pop_03_14['pop_male_prop'] = pop_03_14['pop_male'] / pop_03_14['pop_total_age']


# ## Merging Fentanyl and Naxolone data here
fent_13_14 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/drug_seizures_year.csv')
fent_13_14 = fent_13_14.iloc[:,1:]
fent_13_14.columns = ['year', 'national_opiate_seizures', 'national_fentanyl_seizures']

naxo = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/naxo_years_clean.csv')
naxo.columns = ['year', 'state', 'naxo_crim', 'naxo_civ', 'naxo_third']

#### Merge Population, Population Growth Below

pop_grow_10_14 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pop_dense_10_14.csv') #need to find where this file actually is
pop_grow_00_10 = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pop_grow_00_10.csv') #need to find where this file actually is


#for 2011 population growth, need to do it in 2 parts b/c different CSV layout
#first get 2011 pop estimate

pop_grow_11 = pop_grow_10_14
pop_grow_11['STATE'] = pop_grow_11['STATE'].apply(lambda x: state_fix(x))
pop_grow_11['COUNTY'] = pop_grow_11['COUNTY'].apply(lambda x: county_fix(x))
pop_grow_11['string_county_code'] = pop_grow_11['STATE'] + pop_grow_11['COUNTY']
pop_grow_11['county_code'] = pop_grow_11['string_county_code'].apply(lambda x: int(x))
pop_grow_11['year'] = 2011
pop_grow_11 = pop_grow_11[['POPESTIMATE2011', 'county_code', 'year']]
pop_grow_11.columns = ['population_est', 'county_code', 'year']

#second, get pop change in 2010
pop_grow_2010 = pop_grow_00_10
pop_grow_2010['year'] = 2011
pop_grow_2010['STATE'] = pop_grow_2010['STATE'].apply(lambda x: state_fix(x))
pop_grow_2010['COUNTY'] = pop_grow_2010['COUNTY'].apply(lambda x: county_fix(x))
pop_grow_2010['string_county_code'] = pop_grow_2010['STATE'] + pop_grow_2010['COUNTY']
pop_grow_2010['county_code'] = pop_grow_2010['string_county_code'].apply(lambda x: int(x))
pop_grow_2010 = pop_grow_2010[['year', 'county_code', 'NPOPCHG_2010','RNATURALINC2010','RINTERNATIONALMIG2010','RDOMESTICMIG2010']]
pop_grow_2010.columns = ['year', 'county_code', 'net_pop_change_raw', 'natural_pop_growth_rate', 'intl_migrate_rate', 'dom_migrate_rate']

#and merge
pop_grow_11 = pd.merge(pop_grow_11, pop_grow_2010, how='left', left_on=['year', 'county_code'], right_on=['year', 'county_code'])

#apply function to each year other than 2011 on two .csvs
growth_03 = growth_convert(pop_grow_00_10, 2003)
growth_04 = growth_convert(pop_grow_00_10, 2004)
growth_05 = growth_convert(pop_grow_00_10, 2005)
growth_06 = growth_convert(pop_grow_00_10, 2006)
growth_07 = growth_convert(pop_grow_00_10, 2007)
growth_08 = growth_convert(pop_grow_00_10, 2008)
growth_09 = growth_convert(pop_grow_00_10, 2009)
growth_10 = growth_convert(pop_grow_00_10, 2010)
# growth_11 = growth_convert(pop_grow_00_14, 2011) ## note this requires merging the two csvs because spans the difference
growth_12 = growth_convert(pop_grow_10_14, 2012)
growth_13 = growth_convert(pop_grow_10_14, 2013)
growth_14 = growth_convert(pop_grow_10_14, 2014)
#merge all population growth features - including special 2011
grow_03_14 = pd.concat([growth_03, growth_04, growth_05, growth_06, growth_07, growth_08, growth_09, growth_10, pop_grow_11, growth_12, growth_13, growth_14], axis=0)


# ## Re-merging deaths to make sure no errors above

deaths_simple = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/cdc_data/deaths_simple.csv', delimiter='\t', encoding='utf-8')
#put last years alcohol
deaths_simple = deaths_simple.applymap(lambda x: str(x).replace(",", ""))
deaths_simple = deaths_simple[['County Code', 'Year', 'Deaths', 'Population', 'Crude Rate', 'Age Adjusted Rate']]
deaths_simple.columns = ['county_code', 'year', 'deaths_raw', 'cdc_population', 'death_rate_100k_cdc','death_rate_age_adjusted_cdc']

deaths_simple['year'] = deaths_simple['year'].apply(lambda x: yearify(x))
deaths_simple['county_code'] = deaths_simple['county_code'].apply(lambda x: yearify(x))
deaths_simple['deaths_raw'] = deaths_simple['deaths_raw'].apply(lambda x: death_clean(x))
deaths_simple['cdc_population'] = deaths_simple['cdc_population'].apply(lambda x: death_clean(x))
deaths_simple['death_rate_100k_cdc'] = deaths_simple['death_rate_100k_cdc'].apply(lambda x: death_clean(x))

deaths_pop = deaths_simple.groupby(by=['year', 'county_code']).sum()
deaths_pop.reset_index(inplace=1)

#merging population
deaths_pop = pd.merge(deaths_pop, pop_03_14, how='left', left_on=['year', 'county_code'], right_on=['year', 'county_code'])
# deaths_pop['population'] = deaths_pop['population_x'] #removed b/c order changed, no longer necessary

#merging poverty

deaths_pop = pd.merge(deaths_pop, pov_county_year_03_14, how='left', left_on=['year','county_code'], right_on=['year','county_code'])

#merging unemployment

deaths_pop = pd.merge(deaths_pop, unemp_03_14, how='left', left_on=['year', 'county_code'], right_on=['year', 'county_code'])

# merging fentanyl

deaths_pop = pd.merge(deaths_pop, fent_13_14, how='left', left_on=['year'], right_on=['year'])

#merging population growth

deaths_pop = pd.merge(deaths_pop, grow_03_14, how='left', left_on=['year', 'county_code'], right_on=['year', 'county_code'])
deaths_pop.dropna(axis=0, subset=['working_pop', 'employed_pop', 'deaths_raw', 'population_est'], inplace=1)

#note - need to extract state (str) to merge narcan/ naxo features (below)

###calculations that required merge
deaths_pop['drug_death_rate_100k_cdc']  = deaths_pop['deaths_raw']/(deaths_pop['cdc_population']/100000)
deaths_pop['drug_death_rate_100k']  = deaths_pop['deaths_raw']/(deaths_pop['population_est']/100000) #accepting this as target
deaths_pop['perc_pop_employed'] = deaths_pop['employed_pop']/deaths_pop['population_est']
deaths_pop['unemp_rate'] = 1 - (deaths_pop['employed_pop']/deaths_pop['working_pop']) #this is good
deaths_pop['perc_pop_working'] = deaths_pop['working_pop']/deaths_pop['population_est']
deaths_pop['pov_rate'] = deaths_pop['pov_count_tot']/ deaths_pop['population_est']
deaths_pop['pov_rate_youth'] = deaths_pop['pov_youth_count_0-17']/ deaths_pop['population_est']
deaths_pop['pop_change_rate'] = deaths_pop['net_pop_change_raw'] / deaths_pop['population_est']
deaths_pop['drug_death_rate_100k_diff_cdc_pop'] = deaths_pop['drug_death_rate_100k_cdc'] - deaths_pop['drug_death_rate_100k']
deaths_pop['constant'] = 1.0

deaths_pop.drop(1391, inplace=1)#noticed this is an outliar in plot below Coconino, AZ  pop listed as 1196.0
deaths_pop.dropna(axis=0, subset=['working_pop', 'employed_pop'], inplace=1)

##### Readding county names & states, adding state and year as dummy variables
deaths_pop['county_code'] = deaths_pop['county_code'].apply(lambda x: int(x)) ## this was int 10/1

def county_name_adder(code_to_convert, dict_ref=name_code_convert):
    try:
        return dict_ref[code_to_convert]
    except:
        return np.nan


deaths_pop['county_name'] = deaths_pop['county_code'].apply(lambda x: county_name_adder(x))
### this is an ongoing issue - not sure why all these are not appropriately converting
# I don't want to have tp drop any nulls below
deaths_pop.dropna(axis=0, subset=['county_name'], inplace=1)

deaths_pop['state'] = deaths_pop['county_name'].apply(lambda x: state_extract(x))

state_dums = pd.get_dummies(deaths_pop['state'])
deaths_pop = pd.concat([deaths_pop,state_dums], axis = 1)

deaths_pop = floater(deaths_pop) #convert all remaining fields to float
deaths_pop['year'].apply(lambda x: str(x)) #need year to be in string for dummies
year_dums = pd.get_dummies(deaths_pop['year'])
deaths_pop = pd.concat([deaths_pop,year_dums], axis = 1) #not using these anymore in features

#merging naxolone access policies (Narcan)

deaths_pop = pd.merge(deaths_pop, naxo, how='left', left_on=['year', 'state'], right_on=['year', 'state'])


## Adding auto-regressive features

deaths_pop = deaths_pop.sort_values(by=['county_code', 'year'])
deaths_pop['prior_year_death_rate_100k'] = deaths_pop['drug_death_rate_100k'].shift(1)
deaths_pop['pre_prior_year_death_rate_100k'] = deaths_pop['drug_death_rate_100k'].shift(2)
deaths_pop['prior_year_death_growth_rate'] = (deaths_pop['prior_year_death_rate_100k'] - deaths_pop['pre_prior_year_death_rate_100k'])/deaths_pop['pre_prior_year_death_rate_100k']
deaths_pop['three_back_year_death_rate_100k'] = deaths_pop['drug_death_rate_100k'].shift(3)
deaths_pop['two_year_period_death_growth_rate'] = (deaths_pop['prior_year_death_rate_100k'] - deaths_pop['three_back_year_death_rate_100k'])/ deaths_pop['three_back_year_death_rate_100k']
deaths_pop['two_three_year_death_growth_rate'] = (deaths_pop['pre_prior_year_death_rate_100k']-deaths_pop['three_back_year_death_rate_100k'])/ deaths_pop['three_back_year_death_rate_100k']
deaths_pop['years_since_14'] = 2014 - deaths_pop['year']

deaths_pop.drop(886, inplace=1)
deaths_pop = deaths_pop.sort_index() #to undo sorting
death_df = deaths_pop  ## going to clean death_df to include only counties w/ auto-regressive features


death_df['count_county_code'] = np.nan
death_df.reset_index(inplace=1)
death_df = dict_applier(death_df)

deaths_pop = death_df[death_df['count_county_code']>=4] # 4 b/c 3 auto-regressive features


##### Implementing Model Below


median_drug_rate = np.median(deaths_pop['drug_death_rate_100k'])
avg_drug_rate = np.mean(deaths_pop['drug_death_rate_100k'])

y = deaths_pop['drug_death_rate_100k']

# v1 features = deaths_pop[['year','pop_sub_15_prop','pop_15-34_prop','pop_35-54_prop','pop_55+_prop','pop_asian_prop', 'pop_white_prop','pop_black_prop','pop_hisp_prop','population','med_hh_income', 'unemp_rate', 'perc_pop_working', 'pov_rate', 'pov_rate_youth', 'constant', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO','CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME','MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NH','NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI','SC','SD', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY']]
# v2 features, removed hisp, years, population below = deaths_pop[['pop_sub_15_prop','pop_15-34_prop','pop_35-54_prop','pop_55+_prop','pop_asian_prop', 'pop_white_prop','pop_black_prop','pop_hisp_prop','population','med_hh_income', 'unemp_rate', 'perc_pop_working', 'pov_rate', 'pov_rate_youth', 'constant', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO','CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME','MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NH','NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI','SC','SD', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY', 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]]

features = deaths_pop[['prior_year_death_rate_100k', 'pre_prior_year_death_rate_100k', 'prior_year_death_growth_rate','three_back_year_death_rate_100k', 'two_year_period_death_growth_rate', 'two_three_year_death_growth_rate', 'national_fentanyl_seizures', 'naxo_crim', 'naxo_civ', 'naxo_third', 'pop_sub_15_prop','pop_15-34_prop','pop_35-54_prop','pop_55+_prop','pop_asian_prop', 'pop_white_prop','pop_black_prop','med_hh_income', 'unemp_rate', 'perc_pop_working', 'pov_rate', 'pov_rate_youth', 'population_est', 'pop_change_rate','natural_pop_growth_rate', 'intl_migrate_rate', 'dom_migrate_rate', 'constant', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO','CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME','MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NH','NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI','SC','SD', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY']]
#8/25 added autoregressive features

features.to_csv('/Users/HudsonCavanagh/Dropbox/Capstone/csv_output/full_features_cdc_states_v7.csv')

X = StandardScaler().fit_transform(features)
X = pd.DataFrame(X, columns=features.columns)

#will be using cross_val, this is to validate most successful model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)


#PCA for later plots & model testing

pca_2 = PCA(n_components=2) # for visualization
pca_3 = PCA(n_components=3)
pca_4 = PCA(n_components=4)
xPC_2 = pca_2.fit_transform(X) # for visualization
xPC_3 = pca_3.fit_transform(X)
xPC_4 = pca_4.fit_transform(X)

stratk = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=66) #for cross_val

dt_simple = DecisionTreeRegressor(min_samples_leaf=4, min_samples_split=6, max_depth=6)
# dt_simple.fit(X,y)
s_dt = cross_val_score(dt_simple, X, y, cv=stratk, n_jobs=-1)
# scoring='roc_auc'

print("Decision Tree CV score", s_dt.mean())

rf = RandomForestRegressor()
rf_cv = RandomForestRegressor()

rf.fit(X_train, y_train)
ypred_rf = rf.predict(X_test)
s_rf_cv = cross_val_score(rf_cv, X, y, cv=stratk, n_jobs=-1)
 # scoring='roc_auc'

print("Random Forest CV score", s_rf_cv.mean())

# code for GS RF model

# gs_rf = GridSearchCV(rf, {'min_samples_split': split_vals, 'max_depth': depths, 'max_features': max_feats}, cv=15)
# s_gs_rf_cv = cross_val_score(gs_rf, X, y, cv=stratk, n_jobs=-1)
# s_gs_rf_cv.mean()


#Bagging/ Bootstrapped Model
bc_cv = BaggingRegressor(n_estimators=100)
# bc_cv.fit(X_train,y_train)
s_bc = cross_val_score(bc_cv, X, y, cv=stratk, n_jobs=-1)

print("{} Score:\t{:0.3} ± {:0.3}".format("Bagging Trees", s_bc.mean(), s_bc.std()))

# ## Looking at Feature Importance
rel_feature_import = sorted(zip(rf.feature_importances_, features.columns), reverse=True)
rel_feature_import = pd.DataFrame(rel_feature_import)
print("RF feature importance:", rel_feature_import) #pop was .74 before outliar removal


# ## Implementing Kaggle-Darling XG Boost

xg_model = xgboost.XGBRegressor()
xg_model_cv = xgboost.XGBRegressor()
xg_model.fit(X_train, y_train)
s_xg_cv = cross_val_score(xg_model_cv, X, y, cv=stratk, n_jobs=-1)

# make predictions for test data
y_pred_xg = xg_model.predict(X_test)
print("XG Boost cross_val", s_xg_cv.mean())


# ### Implementing GradientBoosting Regressor to see if this is diff from XG Boost

gb_tree = GradientBoostingRegressor(n_estimators=1000)
gb_tree_cv = GradientBoostingRegressor(n_estimators=1000, max_depth=10)

gb_tree.fit(X_train, y_train)
ypred_gbtree = gb_tree.predict(X_test)


s_gb_tree_cv = cross_val_score(gb_tree_cv, X, y, cv=stratk, n_jobs=-1)

print("Gradient Boosted cross_val", s_gb_tree_cv.mean())


## Now going to use Classification models - Random Forest and XG Boost

y_bin = y.apply(lambda x: 1 if x > median_drug_rate else 0) # new target for classification
rf_c_cv = RandomForestClassifier()
s_rf_cv = cross_val_score(rf_c_cv, X, y_bin, cv=stratk, n_jobs=-1, scoring='roc_auc')
print("random forest, median drug rate classifier AUC", s_rf_cv.mean())

xg_class_model_cv = xgboost.XGBClassifier()
s_xg_cv = cross_val_score(xg_class_model_cv, X, y_bin, cv=stratk, n_jobs=-1, scoring='roc_auc')
print("XG Boost, median drug rate classifier AUC", s_xg_cv.mean())


## Predicting change in 2014, using 2003-2013 data (hardest model to get accurate)
print("moving into 2014 predictions")

input_2013 = deaths_pop[deaths_pop['year']<2014]
input_2013.reset_index(inplace=1) # added this 8/23 to alleviate nans below
test_2014 = deaths_pop[deaths_pop['year']==2014]
test_2014.reset_index(inplace=1) # added this 8/23 to alleviate nans below

#creating regression training
y_03_13 = input_2013['drug_death_rate_100k']

#regression target
y_14 = test_2014['drug_death_rate_100k']

#creating classification (bigger/ smaller) train & target
y_delta_bin_03_13 = (input_2013['drug_death_rate_100k']-input_2013['prior_year_death_rate_100k']).apply(lambda x: 1 if x >= 0 else 0) #one means increase
y_delta_bin_14 = (test_2014['drug_death_rate_100k']-test_2014['prior_year_death_rate_100k']).apply(lambda x: 1 if x >= 0 else 0) #one means increase

#creating regression train & target for drug rate growth
y_delta_grow_03_13 = ((input_2013['drug_death_rate_100k']-input_2013['prior_year_death_rate_100k'])/input_2013['prior_year_death_rate_100k'])
y_delta_grow_14 = ((test_2014['drug_death_rate_100k']-test_2014['prior_year_death_rate_100k'])/test_2014['prior_year_death_rate_100k'])


#8/22 added prior year death rate as feature
features_03_13 = input_2013[['prior_year_death_rate_100k','pre_prior_year_death_rate_100k', 'prior_year_death_growth_rate','three_back_year_death_rate_100k', 'two_year_period_death_growth_rate', 'two_three_year_death_growth_rate', 'national_fentanyl_seizures','naxo_crim', 'naxo_civ', 'naxo_third', 'pop_sub_15_prop','pop_15-34_prop','pop_35-54_prop','pop_55+_prop','pop_asian_prop', 'pop_white_prop','pop_black_prop','med_hh_income', 'unemp_rate', 'perc_pop_working', 'pov_rate', 'pov_rate_youth', 'population_est', 'pop_change_rate','natural_pop_growth_rate', 'intl_migrate_rate', 'dom_migrate_rate', 'constant', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO','CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME','MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NH','NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI','SC','SD', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY']]
features_14 = test_2014[['prior_year_death_rate_100k','pre_prior_year_death_rate_100k', 'prior_year_death_growth_rate','three_back_year_death_rate_100k', 'two_year_period_death_growth_rate', 'two_three_year_death_growth_rate','national_fentanyl_seizures','naxo_crim', 'naxo_civ', 'naxo_third', 'pop_sub_15_prop','pop_15-34_prop','pop_35-54_prop','pop_55+_prop','pop_asian_prop', 'pop_white_prop','pop_black_prop','med_hh_income', 'unemp_rate', 'perc_pop_working', 'pov_rate', 'pov_rate_youth', 'population_est', 'pop_change_rate','natural_pop_growth_rate', 'intl_migrate_rate', 'dom_migrate_rate', 'constant', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO','CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME','MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NH','NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI','SC','SD', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY']]
X_03_13 = StandardScaler().fit_transform(features_03_13)
X_03_13 = pd.DataFrame(X_03_13, columns=features_03_13.columns)

X_14 = StandardScaler().fit_transform(features_14)
X_14 = pd.DataFrame(X_14, columns=features_14.columns)

gb_14 = GradientBoostingRegressor(n_estimators=1000, max_depth=10)
gb_14.fit(X_03_13, y_03_13)
ypred_14 = gb_14.predict(X_14)

print("GBoosted rsq score, overall drug death rate regression", r2_score(y_14, ypred_14))

#.6075 before fent feature


#logistic regression to test direction of  features

logreg_basic = LogisticRegression(solver='liblinear')
logreg_basic.fit(X_03_13, y_delta_bin_03_13)
y_bin_preds_log = logreg_basic.predict(X_14)



# X_03_13_ = pd.merge(X_03_13_mod, state_features_train, how='left', left_index=1, right_index=1) -- pretty sure to be removed


# Grid Search implementation on LogisticRegression

logreg = LogisticRegression(solver='liblinear')
# C_vals = [0.01, 0.1, .25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
# penalties = ['l1','l2']
#
# gs = GridSearchCV(logreg, {'penalty': penalties, 'C': C_vals}, verbose=False, cv=15)
# gs.fit(X_03_13, y_delta_bin_03_13)
# y_bin_preds_gs_logis = gs.predict(X_14)
logreg.fit(X_03_13, y_delta_bin_03_13)
y_bin_preds_log = logreg.predict(X_14)

coefs_variables = pd.DataFrame(zip(logreg.coef_[0], X_03_13.columns))
coefs_variables.columns = ['coef', 'feature']
coefs_variables['abs_coef'] = abs(coefs_variables['coef'])
coefs_variables = coefs_variables.sort_values(by='abs_coef', ascending=0)
print("coefficients of logistic regressions:", coefs_variables)

# don't need another CM, commented out
# lr_basic_cm = confusion_matrix(y_delta_bin_14, y_bin_preds_gs_logis)
# lr_basic_cm_df = pd.DataFrame(lr_basic_cm)
# print(lr_basic_cm_df)


# ## How accurately can we classify communities as compared to the median rate?

y_bin_03_13 = y_03_13.apply(lambda x: 1 if x > median_drug_rate else 0)
y_bin_14_obs = y_14.apply(lambda x: 1 if x > median_drug_rate else 0)


xg_class_model_14 = xgboost.XGBClassifier()
xg_class_model_14.fit(X_03_13, y_bin_03_13)
y_class_14 = xg_class_model_14.predict(X_14)

xg_14_cm = confusion_matrix(y_bin_14_obs, y_class_14, labels=xg_class_model_14.classes_)
print("confusion matrix, classification against median rate (XG Boost)", xg_14_cm)

# xg_14_cm = pd.DataFrame(xg_14_cm, columns=xg_class_model_14.classes_, index=xg_class_model_14.classes_)
feat_import_XG_class_14 = pd.Series(zip(X_03_13, xg_class_model_14.feature_importances_))
print("relative feature importance, classification against median rate (XG Boost)", feat_import_XG_class_14)

#Using tree-based feature importance to reduce my features -- predicting against median

xg_class_model_14_select = SelectFromModel(xg_class_model_14, prefit=True)
X_03_13_new = xg_class_model_14_select.transform(X_03_13)
X_14_new = xg_class_model_14_select.transform(X_14)


xg_class_model_14_mod = xgboost.XGBClassifier()
xg_class_model_14_mod.fit(X_03_13_new, y_bin_03_13)
y_class_xg_14_mod = xg_class_model_14_mod.predict(X_14_new)
xg_14_mod_cm = confusion_matrix(y_class_xg_14_mod, y_bin_14_obs, labels=xg_class_model_14.classes_)


print('f1:', sklearn.metrics.f1_score(y_bin_14_obs,y_class_xg_14_mod),
'accuracy:', accuracy_score(y_bin_14_obs,y_class_xg_14_mod))


# plot_confusion_matrix(xg_14_mod_cm)


print('f1:', sklearn.metrics.f1_score(y_bin_14_obs,y_class_14),
'accuracy:', accuracy_score(y_bin_14_obs,y_class_14))


# ### How accurately can we predict the directional movement of communities in 2014, e.g. can our model predict which communties were going to get worse in 2014?

print("this is the hard part, predicting the 2014 delta...")

#creating naive guesses as baseline
naive_guesses = y_delta_bin_14.apply(lambda x: 1)

#xg classificaiton model 1
xg_delta_model_14 = xgboost.XGBClassifier()
xg_delta_model_14.fit(X_03_13, y_delta_bin_03_13)
y_delta_14_pred = xg_delta_model_14.predict(X_14)

xg_delta_14_cm = confusion_matrix(y_delta_bin_14, y_delta_14_pred)
# xg_14_cm = pd.DataFrame(xg_14_cm, columns=xg_class_model_14.classes_, index=xg_class_model_14.classes_)
# plot_confusion_matrix(xg_delta_14_cm)


print("XG Boost classification model auc score ('14 delta'):", sklearn.metrics.roc_auc_score(y_delta_bin_14,y_delta_14_pred),
      "naive auc score:", sklearn.metrics.roc_auc_score(y_delta_bin_14, naive_guesses))


#xg classificaiton model 2 w/ feature transformation

xg_delta_model_14_select = SelectFromModel(xg_delta_model_14, prefit=True)
X_03_13_new_delta = xg_delta_model_14_select.transform(X_03_13)
X_14_new_delta = xg_delta_model_14_select.transform(X_14)

xg_delta_model_14_mod = xgboost.XGBClassifier()
xg_delta_model_14_mod.fit(X_03_13_new_delta, y_delta_bin_03_13)
y_delta_14_ypred = xg_delta_model_14_mod.predict(X_14_new_delta)
xg_14_mod_cm = confusion_matrix(y_delta_bin_14, y_delta_14_ypred, labels=xg_class_model_14.classes_)

print("XG Boost classification model auc score ('14 delta') with feature transformation:", sklearn.metrics.roc_auc_score(y_delta_bin_14,y_delta_14_ypred),
      "naive auc score:", sklearn.metrics.roc_auc_score(y_delta_bin_14, naive_guesses))


#model number 3 - Regression

gb_delta_14 = GradientBoostingRegressor(n_estimators=1000)

gb_delta_14.fit(X_03_13, y_delta_grow_03_13)
gb_delta_14_ypred = gb_delta_14.predict(X_14)

print("GB Regression r2 score for 2014 delta predictions:", r2_score(y_delta_grow_14, gb_delta_14_ypred))

#model number 4

gb_delta_model_14_select = SelectFromModel(gb_delta_14, prefit=True)
X_03_13_new_delta_gb = gb_delta_model_14_select.transform(X_03_13)
X_14_new_delta_gb = gb_delta_model_14_select.transform(X_14)

#lets try this with both the features from X_03_13_new_delta_gb and X_03_13_new_delta (same for test)


xg_delta_model_14_mod = xgboost.XGBClassifier()
xg_delta_model_14_mod.fit(X_03_13_new_delta_gb, y_delta_bin_03_13)
y_delta_gb_14_mod = xg_delta_model_14_mod.predict(X_14_new_delta_gb)
# gb_14_mod_cm = confusion_matrix(y_delta_grow_14, y_delta_gb_14_mod, labels=xg_class_model_14.classes_)

print("XG Classify AUC score, delta '14 with feature selection", sklearn.metrics.roc_auc_score(y_delta_grow_14, y_delta_gb_14_mod)
