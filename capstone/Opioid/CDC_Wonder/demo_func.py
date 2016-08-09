def demo_counter(df, age, gender_code, race, hisp):
    for i in df.iloc[i,:]:
        if df['Year Age Groups Code'] == age:
            if df['Gender Code'] == gender_code:
                if df['Race'] == race:
                    if df['Hispanic Origin']==hisp:
                        df['demo_id'] = (age, gender_code, race, hisp)
                        df['demo_pop'] = df.ix[i,'population']
                        df['demo_deaths'] = df.ix[i,'deaths']
                        df['demo_per_cap_100k'] = df.ix[i,'crude_100k']
