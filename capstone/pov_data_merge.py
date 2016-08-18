import pandas as pd
import numpy as np


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
pov_county_year = pov_03_14.groupby(by=['year', 'county_code']).mean()
pov_county_year.reset_index(inplace=1)
pov_county_year.to_csv('pov_county_year_03_14.csv')
