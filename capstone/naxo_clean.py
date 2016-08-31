import pandas as pd

naxo = pd.read_csv('/Users/HudsonCavanagh/Dropbox/Capstone/non_cdc_data/naxolone_policy_data.csv') #need to find where this file actually is

naxo['begin_date'] = pd.to_datetime(naxo['Begin Date'])
naxo['end_date'] = pd.to_datetime(naxo['End Date'])



# naxo['nalo_crim_2003'] = naxo['naloxone-crimpossesion'] *

years_list = pd.Series([2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014])
# years_list = years_list.apply(lambda x: pd.to_datetime(x))

def naxo_clean(naxo, years):
#     years = years.apply(lambda x: pd.to_datetime(x))
    new_df = pd.DataFrame(columns = ['state', 'code', 'year', 'law'])
    for i in range(len(naxo)):
        for year in years:
            print("beginning", year)
            for col in [naxo['naloxone-crimpossesion'], naxo['naimmcivlialpyn'], naxo['naloxone-thirdcare']]:
                if naxo['begin_date'][i].year <= year <= naxo['end_date'][i].year:
                    
                    mini_df = pd.DataFrame([naxo['State'][i], col[i], year, 1])
                    print mini_df
                    new_df =  pd.concat([new_df, mini_df], axis=0)
                else:
                    mini2_df = pd.DataFrame([naxo['State'][i], col[i], year, 0])
                    print mini_df2
                    new_df =  pd.concat([new_df, mini_df2], axis=0)

    return new_df

df = naxo_clean(naxo, years_list)
df.to_csv('/Users/HudsonCavanagh/Dropbox/Capstone/csv_output/naxo_years.csv')
