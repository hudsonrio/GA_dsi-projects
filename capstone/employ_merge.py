import pandas as pd
import numpy as np


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


unemp_tot = unemp_tot.groupby(by=['year', 'county_code']).sum()
unemp_tot.reset_index(inplace=1)
unemp_tot.to_csv('unemp_tot_03_14.csv')
