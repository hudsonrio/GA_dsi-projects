import pandas as pd

#might be as simple as: bc already in same s3 bucket
pop_10 = pd.read_csv("pop_10_race_age.csv")
pop_10_age = pd.read_csv("pop_10_age_sex.csv")
# pop_15 = pd.read_csv("pop_data_15.csv")

#this CSV (2010-15) has different col names, so applying changes manually

# pop_15['year'] = pop_15['YEAR'].apply(lambda x: year_clean(x))
# pop_15['AGEGRP'] = pop_15['AGEGRP'].apply(lambda x: int(x))
# pop_15['pop_sub_15'] = pop_15['TOT_POP'] * pop_15['AGEGRP'].apply(lambda x: 1 if x in (1,2,3) else 0)
# pop_15['pop_15-34'] = pop_15['TOT_POP'] * pop_15['AGEGRP'].apply(lambda x: 1 if x in (4,5,6,7) else 0)
# pop_15['pop_35-54'] = pop_15['TOT_POP'] * pop_15['AGEGRP'].apply(lambda x: 1 if x in (8,9,10,11) else 0)
# pop_15['pop_55+'] = pop_15['TOT_POP'] * pop_15['AGEGRP'].apply(lambda x: 1 if x >= 12 else 0)
# pop_15['pop_black'] = pop_15['BAC_MALE'] +  pop_15['BAC_FEMALE']
# pop_15['pop_white'] = pop_15['WA_MALE'] +  pop_15['WA_FEMALE']
# pop_15['pop_hisp'] = pop_15['H_MALE'] +  pop_15['H_FEMALE']
# pop_15['pop_asian'] = pop_15['AA_MALE'] +  pop_15['AA_FEMALE']
# pop_15['pop_merge_ind'] = pop_15['CTYNAME'] + "_" + pop_15['year']
#


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


# pop_03 = pd.DataFrame()
# pop_03 = pop_clean_analyze(pop_10, pop_10_age, pop_03, 'POPESTIMATE2003', 2003)

pop_04 = pd.DataFrame()
pop_04 = pop_clean_analyze(pop_10, pop_10_age, pop_04, 'POPESTIMATE2004', 2004)

# pop_05 = pd.DataFrame()
# pop_05 = pop_clean_analyze(pop_10, pop_10_age, pop_05, 'POPESTIMATE2005', 2005)
#
# pop_06 = pd.DataFrame()
# pop_06 = pop_clean_analyze(pop_10, pop_10_age, pop_06, 'POPESTIMATE2006', 2006)
#
# pop_07 = pd.DataFrame()
# pop_07 = pop_clean_analyze(pop_10, pop_10_age, pop_07, 'POPESTIMATE2007', 2007)
#
# pop_08 = pd.DataFrame()
# pop_08 = pop_clean_analyze(pop_10, pop_10_age, pop_08, 'POPESTIMATE2008', 2008)
#
# pop_09 = pd.DataFrame()
# pop_09 = pop_clean_analyze(pop_10, pop_10_age, pop_09, 'POPESTIMATE2009', 2009)
#



def state_coder_gen(df):
    a = df['STATE'].apply(lambda x: str(x))
    b = df['COUNTY'].apply(lambda x: str(x))
    for xa in a:
        if len(xa) < 2:
            xa = "0" + xa
    for xb in b:
        if len(xb) < 3:
            xb = "0" + xb
        elif len(xb) < 2:
            xb = "00" + xb
    df['county_code'] = a+b
    return df

# pop_03 = state_coder_gen(pop_03)
pop_04 = state_coder_gen(pop_04)
# pop_05 = state_coder_gen(pop_05)
# pop_06 = state_coder_gen(pop_06)
# pop_07 = state_coder_gen(pop_07)
# pop_08 = state_coder_gen(pop_08)
# pop_09 = state_coder_gen(pop_09)
# pop_15 = state_coder_gen(pop_15)


# pop_03_g = pop_03.groupby(['year', 'county_code'], axis=0).sum()
pop_04_g = pop_04.groupby(['year', 'county_code'], axis=0).sum()
# pop_05_g = pop_05.groupby(['year', 'county_code'], axis=0).sum()
# pop_06_g = pop_06.groupby(['year', 'county_code'], axis=0).sum()
# pop_07_g = pop_07.groupby(['year', 'county_code'], axis=0).sum()
# pop_08_g = pop_08.groupby(['year', 'county_code'], axis=0).sum()
# pop_09_g = pop_09.groupby(['year', 'county_code'], axis=0).sum()
# pop_10_15 = pop_15.groupby(['year', 'county_code'], axis=0).sum()




#combining after the groupby
# pop_03_15 = pd.concat([pop_03_g, pop_04_g, pop_05_g, pop_06_g, pop_07_g, pop_08_g, pop_09_g, pop_10_15]) #this works 1) need to make floats 2) need more data

#applying post groupby calculations for rates elsewhere
# pop_03_15['pop%_men'] = pop_03_15['pop_male']/  pop_03_15['population_age_sex']
# pop_03_15['pop%_sub_15'] = pop_03_15['pop_sub_15']/ pop_03_15['population_age_sex']
# pop_03_15['pop%_15-34'] = pop_03_15['pop_15-34']/ pop_03_15['population_age_sex']
# pop_03_15['pop%_35-54'] = pop_03_15['pop_35-54']/ pop_03_15['population_age_sex']
# pop_03_15['pop%_55+'] = pop_03_15['pop_55+']/ pop_03_15['population_age_sex']
# pop_03_15['pop%_black'] = pop_03_15['pop_black']/ pop_03_15['population_race_hisp']
# pop_03_15['pop%_white'] = pop_03_15['pop_white']/ pop_03_15['population_race_hisp']
# pop_03_15['pop%_hisp'] = pop_03_15['pop_hisp']/ pop_03_15['population_race_hisp']
# pop_03_15['pop%_asian'] = pop_03_15['pop_asian']/ pop_03_15['population_race_hisp']

#reseting index for merge and attempting to write to csv



pop_04_g.reset_index(inplace=1)
pop_04_g.to_csv('pop_04_output.csv')

# pop_03_15.reset_index(inplace=1)
# pop_03_15.to_csv('pop_03_ouput.csv')


print(pop_04_g.head())
# test.head(25)
