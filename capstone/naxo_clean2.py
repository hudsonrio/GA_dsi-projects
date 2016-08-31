import pandas as pd
import numpy as np

naxo = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/naxolone_policy_data.csv') #need to find where this file actually is

naxo['begin_date'] = pd.to_datetime(naxo['Begin Date'])
naxo['end_date'] = pd.to_datetime(naxo['End Date'])

# naxo['nalo_crim_2003'] = naxo['naloxone-crimpossesion'] *

years_list = pd.Series([2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014])
# years_list = years_list.apply(lambda x: pd.to_datetime(x))

def naxo_clean(naxo, years):
#     years = years.apply(lambda x: pd.to_datetime(x))
    cols = ['state', 'naxo_crim_code', 'naxo_civ_code', 'naxo_third_code', 'year', 'valid']
    new_df = pd.DataFrame(columns = cols)
    for i in range(len(naxo)):
        for year in years:
            print("beginning", year)
            # for col in [naxo['naloxone-crimpossesion'], naxo['naimmcivlialpyn'], naxo['naloxone-thirdcare']]:
            if naxo['begin_date'][i].year <= year <= naxo['end_date'][i].year:
                naxo_crim = naxo['naloxone-crimpossesion'][i]
                naxo_civ = naxo['naloxone-crimpossesion'][i]
                naxo_third = naxo['naloxone-crimpossesion'][i]
                mini_df = pd.DataFrame([[naxo['State'][i], naxo_crim, naxo_civ, naxo_third, year, 1]])
                mini_df = mini_df.T
                print(mini_df)
                mini_df = pd.DataFrame(mini_df, columns = cols)
                new_df.append(mini_df, ignore_index=True)
                print(new_df)
            else:
                # f.append({'Animal':'mouse', 'Color':'black'}, ignore_index=True)

                mini_df = pd.DataFrame([[naxo['State'][i], naxo_crim, naxo_civ, naxo_third, year, 0]])
                mini_df = mini_df.T
                # print(mini_df)
                mini_df = pd.DataFrame(mini_df, columns = cols)
                new_df.append(mini_df, ignore_index=True)
    return new_df

df = naxo_clean(naxo, years_list)
df.to_csv('/Users/HudsonCavanagh/Dropbox/Capstone/csv_output/naxo_years.csv')
