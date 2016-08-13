import pandas as pd

cols_bike = ['tripduration', 'starttime', 'stoptime', 'start_station_id',
       'start_station_name', 'start_station_latitude',
       'start_station_longitude', 'end_station_id', 'end_station_name',
       'end_station_latitude', 'end_station longitude', 'bikeid',
       'usertype', 'birth_year', 'gender']

bike = pd.read_csv('https://s3.amazonaws.com/citibikeprojectbucket/big_bike.csv', engine='python', names=cols_bike)

import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold


bike = bike[bike['usertype']=='Subscriber']
bike = bike[bike['birth_year'].isnull() == False]
bike = bike[bike['gender'] != 0]


day_dums = pd.get_dummies(bike['day_of_week'])
day_list = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
day_dums.columns = day_list
bike = pd.concat([bike, day_dums], axis=1)

gender_features = ['date', 'start_station_id','monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
gen_y = int(bike['gender'])
gen = bike.loc[:,gender_features]

rfc = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=6, min_samples_leaf=5)

rfc.fit(gen, gen_y)

s_rfc = cross_val_score(rfc, gen, gen_y, n_jobs=-1).mean()
print(s_rfc)

import statsmodels.api as sm


bike['date'] = pd.to_datetime(bike['starttime'])
bike['day'] = bike['date'].day()

rides_by_day = bike.groupby(by=['day', 'start station id']).count()
rides_by_day.head()

# june16.iloc[:, :]

####Basic model

df = rides_by_day

df.set_index('date', inplace=True)

#transform it by week
df_weeks = df_log.diff(periods=52)[52:]


#look at an arbitrary lag to see if this looks right
plot_acf(df_weeks, lags=30)
