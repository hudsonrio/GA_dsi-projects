import pandas as pd
import sklearn
import numpy as np


q1 = pd.read_csv('ri_stops_q1.csv')
q2 = pd.read_csv('ri_stops_q2.csv')

q1['violation_rate'] = q1['Violation']/ q1['Stops']
q1['search_rate'] = q1['Searches Conducted']/ q1['Stops']
q1['warning_rate'] = q1['Warning']/ q1['Stops']
q1['pass_arrest_rate'] = q1['Arrest Driver']/ q1['Stops']
q1['driver_arrest_rate'] = q1['Arrest Passenger']/ q1['Stops']
q1['MV_citation_rate'] = q1['M V Citation']/ q1['Stops']


dpt_list = list(q1['Department'].unique())
q1['driver_arrest_rate_tot'] = np.nan

for i in dpt_list:
    a = q1[q1['Department'] == i]
    a = a[a['Driver Race'] == 'Total']
#     print(a)
    for ind in range(len(q1)):
        if q1.ix[ind, 'Driver Race'] != 'Total':
#             print(type(a['driver_arrest_rate']))
            q1.ix[ind, 'driver_arrest_rate_tot'] = list(a['driver_arrest_rate'])[0]
            q1.ix[ind, 'violation_rate_tot'] = list(a['violation_rate'])[0]
            q1.ix[ind, 'search_rate_tot'] = list(a['search_rate'])[0]
            q1.ix[ind, 'warning_rate_tot'] = list(a['warning_rate'])[0]
            q1.ix[ind, 'pass_arrest_rate_tot'] = list(a['pass_arrest_rate'])[0]
            q1.ix[ind, 'MV_citation_rate_tot'] = list(a['MV_citation_rate'])[0]

print(q1.head())
q1.to_csv('stops_q1_tots.csv')
