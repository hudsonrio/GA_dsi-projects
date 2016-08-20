import pandas as pd

pop_10_age = pd.read_csv("https://s3.amazonaws.com/hudsonbucket/pop_10_age_sex.csv")
pop_10 = pd.read_csv("https://s3.amazonaws.com/hudsonbucket/pop_data_10.csv")

#might be as simple as: bc already in same s3 bucket
# pop_10 = pd.read_csv("./pop_data_10.csv")
# pop_10_age = pd.read_csv("./pop_10_age_sex.csv")

# pop_10 = pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pop_data_10.csv")
# pop_10_age = pd.read_csv("/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/pop_10_age_sex.csv")

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



pop_07 = pd.DataFrame()
pop_07 = pop_clean_analyze(pop_10, pop_10_age, pop_07, 'POPESTIMATE2007', 2007)
pop_07 = pop_07.iloc[:,:100]


def state_coder_gen(df):
    a= df['STATE']
    b= df['COUNTY']
    a,b = str(a), str(b)
    if len(a) < 2:
        a = "0" + a
    while len(b) < 3:
        b = "0" + b
    df['county_code'] = a+b
    return df

test = state_coder_gen(pop_07)
# test.head(15)
test.to_csv('test_pop.csv')
pop_07_g1 = test.groupby(['year', 'county_code'], axis=0).sum()
pop_07_g1.to_csv('test1_pop_07.csv')
print(pop_07_g1.head())

#STILL NEED TO MAKE COUNTY_STATE FOR  pop_15
def state_coder_gen2(df):
    a= df['STATE']
    b= df['COUNTY']
    a,b = str(a), str(b)
    if len(a) < 2:
        a = "0" + a
    if len(b) < 3:
        b = "0" + b
    elif len(b) < 2:
        b = "00" + b
    df['county_code'] = a+b
    df['county_code'] = df['county_code'].apply(lambda x: str(x))
    return df

test2 = state_coder_gen2(pop_07)
pop_07_g = test2.groupby(['year', 'county_code'], axis=0).sum()
pop_07_g.to_csv('test2_pop_07.csv')
print(pop_07_g.head())
# test.head(25)
