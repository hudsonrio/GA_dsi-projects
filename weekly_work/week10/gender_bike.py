import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold

cols_bike = ['tripduration', 'starttime', 'stoptime', 'start_station_id',
       'start_station_name', 'start_station_latitude',
       'start_station_longitude', 'end_station_id', 'end_station_name',
       'end_station_latitude', 'end_station longitude', 'bikeid',
       'usertype', 'birth_year', 'gender']

bike = pd.read_csv('/Users/HudsonCavanagh/Desktop/big_bike.csv', engine='python', names=cols_bike)

bike = bike[bike['usertype']=='Subscriber']
bike = bike[bike['birth_year'].isnull() == False]
bike = bike[bike['gender'] != 0]

bike['date'] = pd.to_datetime(bike['starttime'])
bike['day'] = bike['date'].apply(lambda x: day(x))
bike['day_of_week'] = bike['date'].apply(lambda x: dayofweek(x))

day_dums = pd.get_dummies(bike['day_of_week'])
day_dums.columns = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
bike = pd.concat([bike, day_dums], axis=1)

gender_features = ['date', 'start_station_id','monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
gen_y = bike['gender'].apply(lambda x: int(x))
gen = bike.loc[:,gender_features]

rfc = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=6, min_samples_leaf=5)

rfc.fit(gen, gen_y)

s_rfc = cross_val_score(rfc, gen, gen_y, n_jobs=-1).mean()
print(s_rfc)
