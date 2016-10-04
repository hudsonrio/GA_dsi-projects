import pandas as pd
import sklearn
import numpy as np


q2 = pd.read_csv('ri_stops_q2.csv')

q2.iloc[290,0] = "Rhode_Island_Total"


q2['violation_rate'] = q2['Violation']/ q2['Stops']
q2['search_rate'] = q2['Searches Conducted']/ q2['Stops']
q2['warning_rate'] = q2['Warning']/ q2['Stops']
q2['pass_arrest_rate'] = q2['Arrest Driver']/ q2['Stops']
q2['driver_arrest_rate'] = q2['Arrest Passenger']/ q2['Stops']
q2['MV_citation_rate'] = q2['M V Citation']/ q2['Stops']

ref = q2.iloc[290,:]
print(ref.head())

dpt_list = list(q2['Department'].unique())

q2['driver_arrest_rate_tot'] = np.nan
q2['violation_rate_tot'] = np.nan
q2['search_rate_tot'] = np.nan
q2['warning_rate_tot'] = np.nan
q2['pass_arrest_rate_tot'] = np.nan
q2['MV_citation_rate_tot'] = np.nan


for ind in range(len(q2)):
    q2['violation_rate_tot'] = ref['violation_rate']
    q2['search_rate_tot'] = ref['search_rate']
    q2['warning_rate_tot'] = ref['warning_rate']
    q2['pass_arrest_rate_tot'] = ref['pass_arrest_rate']
    q2['MV_citation_rate_tot'] = ref['MV_citation_rate']

print(q2.head())
q2.to_csv('stops_q2_state_tot.csv')
