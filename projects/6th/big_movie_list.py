import pandas as pd
import omdb

def big_movie_extractor(series, keys):
    movie_list = []
    for i in set(series):
        try:
            movie_list.append((omdb.get(title=i, tomatoes=True).values()))
        except:
            print("er ")
    df_full = pd.DataFrame(movie_list, columns=keys)
    return(df_full)


keys_wrapper = (omdb.get(title='True Grit', tomatoes=True)).keys()
all_movies = pd.read_csv("/Users/HudsonCavanagh/Documents/All U.S. Released Movies- 1972-2016.csv")
big_df = big_movie_extractor(all_movies['Title'].values, keys_wrapper)
big_df.to_csv("/Users/HudsonCavanagh/Documents/df_full.csv", encoding='utf-8')
