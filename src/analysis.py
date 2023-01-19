import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from sklearn.utils import shuffle



sections_data = pd.read_csv('../input/smalldataset/acts_sections1000000.csv')
cases_2010_data = pd.read_csv('../input/smalldataset/cases_2010_1000000.csv')
print("============ ACTS SECTIONS DATA ================")
print()
print(sections_data.head())
print("================================================")
print()
print()
print("=============== CASES 2010 DATA ================")
print()
print(cases_2010_data.head())
print("================================================")


# Most filed ACTS, SECTIONS and whether they are BAILABLE
sections_size = len(sections_data)
cases_2010_size = len(cases_2010_data)

criminal_data = sections_data.loc[sections_data['criminal']==1]
num_criminal_cases = len(criminal_data)

criminal_sections = criminal_data['section'].value_counts()
criminal_sections.head(10).plot(kind='bar', xlabel='section', ylabel='frequency')
plt.show()

criminal_acts = criminal_data['act'].value_counts()
criminal_acts.head(10).plot(kind='bar', xlabel='acts', ylabel='frequency')
plt.show()

criminal_bailable = criminal_data['bailable_ipc'].value_counts()
criminal_bailable.head(10).plot(kind='bar', xlabel='bailable', ylabel='frequency')
plt.show()



# Acts, Section distribution of non-bailable cases
non_bailable_data = sections_data.loc[sections_data['bailable_ipc']=='non-bailable']
non_bailable_sections = non_bailable_data['section'].value_counts()
non_bailable_sections.head(10).plot(kind='bar', xlabel='section', ylabel='frequency')
plt.show()

non_bailable_acts = non_bailable_data['act'].value_counts()
non_bailable_acts.head(10).plot(kind='bar', xlabel='act', ylabel='frequency')
plt.show()

non_bailable_acts = non_bailable_data['criminal'].value_counts()
non_bailable_acts.head(10).plot(kind='bar', xlabel='criminal', ylabel='frequency')
plt.show()



# dispositions announced by the courts
disp_name_data = pd.read_csv('../input/precog-data-zip/data/keys/disp_name_key.csv')
print("============ DISP NAME DATA ================")
print()
print(disp_name_data.head())
print("================================================")

disp_name_2010_data = disp_name_data.loc[disp_name_data['year']==2010]
print(disp_name_2010_data.head(10))
disp_name_2010_data = disp_name_2010_data[['disp_name_s', 'count']]
disp_name_2010_data.head(10).plot(kind='bar', xlabel='disp_name_s', ylabel='frequency')
plt.show()



# Most common disposition over the period 2010-18
total_yearwise_disps = disp_name_data.groupby('year', as_index=False)['count'].sum().values.tolist()
top_disps = disp_name_data.groupby('disp_name_s', as_index=False)['count'].sum()
top_disps = top_disps.sort_values(by='count', ascending = False)

top_disps = pd.DataFrame(top_disps)

top_disps = top_disps.head(10)
top_disps = top_disps[['disp_name_s']]
top_disps = top_disps.values.tolist()

T = []
for i in top_disps:
    T.extend(i)

top_disps = tuple(T)
top_disp_name_data = disp_name_data[disp_name_data['disp_name_s'].isin(top_disps)]

top_disp_name_data_2010 = top_disp_name_data.loc[top_disp_name_data['year']==2010]
top_disp_name_data_2014 = top_disp_name_data.loc[top_disp_name_data['year']==2014]
top_disp_name_data_2018 = top_disp_name_data.loc[top_disp_name_data['year']==2018]


top_disp_name_data_2010['count'] = (top_disp_name_data_2010['count']*100/((total_yearwise_disps[0][1]))).round(2)
top_disp_name_data_2014['count'] = (top_disp_name_data_2014['count']*100/((total_yearwise_disps[4][1]))).round(2)
top_disp_name_data_2018['count'] = (top_disp_name_data_2018['count']*100/((total_yearwise_disps[8][1]))).round(2)

top_disp_name_data_2010 = top_disp_name_data_2010[['year','disp_name_s', 'count']]
top_disp_name_data_2014 = top_disp_name_data_2014[['year','disp_name_s', 'count']]
top_disp_name_data_2018 = top_disp_name_data_2018[['year','disp_name_s', 'count']]

# Storing the analysed data into csv files
top_disp_name_data_2010.to_csv('top_disps_2010.csv')
top_disp_name_data_2014.to_csv('top_disps_2014.csv')
top_disp_name_data_2018.to_csv('top_disps_2018.csv')



# Share of most popular dispositions for the years 2010, 2014, 2018
x = top_disp_name_data_2010['disp_name_s']
width = 0.1
plt.plot(x, top_disp_name_data_2010['count'], color='g', label=2010)
plt.plot(x, top_disp_name_data_2014['count'], color='b', label=2014)
plt.plot(x, top_disp_name_data_2018['count'], color='r', label=2018)
plt.xticks(rotation = 90) 

plt.legend()
plt.show()



# District-wise female ratio
df_judges = pd.read_csv('../input/precog-data-zip/data/judges_clean.csv')

#preprocessing the data
df_judges['female_judge'] = df_judges['female_judge'].replace('1 female',1)
df_judges['female_judge'] = df_judges['female_judge'].replace('0 nonfemale',0)
df_judges['female_judge'] = df_judges['female_judge'].replace('-9998 unclear',np.nan)
df_judges = df_judges.dropna()

#grouping by districts
female_rep = df_judges.groupby(['state_code', 'dist_code'], as_index=False)['female_judge'].mean()
female_rep['female_judge']*=100

#making a unique hash for each district
female_rep['dist_hash'] = female_rep['state_code']*80 + female_rep['dist_code']

# this has the IDs corrected to directly plot on map 
df_flourish = pd.read_csv('../input/1601-flourish-corrected/districts_sorted.csv')

female_rep = female_rep[['female_judge', 'dist_hash']]
df_flourish = df_flourish.set_index('dist_hash').join(female_rep.set_index('dist_hash'))
# stroing analysed data into a csv file
df_flourish.to_csv('dist_female_rep.csv')




# Analysing the time to process a case district-wise

# function to mean and average processing time
def return_avg_std_process(inp_fileName):
    df = pd.read_stata(inp_fileName)
    df = df[['year', 'state_code', 'dist_code' ,'disp_name', 'date_of_filing', 'date_of_decision']]
    
    # data preprocessing
    df = df.dropna()
    df = df.loc[df['date_of_decision'] < '2020-01-01']
    
    # converting to date format
    df['date_of_decision'] = pd.to_datetime(df['date_of_decision'], format="%Y-%m-%d", errors='coerce')
    df['date_of_filing'] = pd.to_datetime(df['date_of_filing'], format="%Y-%m-%d", errors='coerce')
    df['process_time'] = df['date_of_decision'] - df['date_of_filing']
    
    df_process_time_district = df[['year', 'state_code', 'dist_code', 'process_time']]    
    df_process_time_district = df_process_time_district.dropna()
    df_process_time_district = df_process_time_district.loc[df_process_time_district['process_time'] >= '1 days']
    
    # converting to integer
    df_process_time_district['process_time'] = df_process_time_district['process_time'].dt.days
    
    process_time_std_div = df_process_time_district.groupby(['state_code', 'dist_code'], as_index=False)['process_time'].std()
    process_time_std_avg = df_process_time_district.groupby(['state_code', 'dist_code'], as_index=False)['process_time'].mean()

    return [process_time_std_div, process_time_std_avg]


# Analysing the datasets
fileName_2011 = '../input/dta-data/dta/cases/cases/cases_2011.dta'
fileName_2015 = '../input/dta-data/dta/cases/cases/cases_2012.dta'

process_time_std_div_2011, process_time_std_avg_2011 = return_avg_std_process(fileName_2011)
process_time_std_div_2015, process_time_std_avg_2015 = return_avg_std_process(fileName_2015)


# finding the std deviation and mean disposition time
process_time_std_div = process_time_std_div_2011
process_time_std_avg = process_time_std_avg_2011

process_time_std_div['process_time'] = (process_time_std_div_2011['process_time'] + process_time_std_div_2015['process_time'])/2
process_time_std_div['state_code'] = pd.to_numeric(process_time_std_div['state_code'])
process_time_std_div['dist_code'] = pd.to_numeric(process_time_std_div['dist_code'])
process_time_std_div['dist_hash'] = process_time_std_div['state_code']*80 + process_time_std_div['dist_code']

process_time_std_avg['process_time'] = (process_time_std_avg_2011['process_time'] + process_time_std_avg_2015['process_time'])/2
process_time_std_avg['state_code'] = pd.to_numeric(process_time_std_avg['state_code'])
process_time_std_avg['dist_code'] = pd.to_numeric(process_time_std_avg['dist_code'])
process_time_std_avg['dist_hash'] = process_time_std_avg['state_code']*80 + process_time_std_avg['dist_code']

process_time_std_avg.rename(columns = {'process_time':'avg_process_time'}, inplace=True)
process_time_std_div.rename(columns = {'process_time':'std_dev_process_time'}, inplace=True)


# Combining the data of average and std deviation
process_time_std_div = process_time_std_div[['std_dev_process_time', 'dist_hash']]
process_time_std_avg = process_time_std_avg[['avg_process_time', 'dist_hash']]
df_proc_time = process_time_std_div
df_proc_time = df_proc_time.join(process_time_std_avg.set_index('dist_hash'), on='dist_hash')


# Combining with map regions to plot on map
df_flourish = pd.read_csv('../input/1601-flourish-corrected/districts_sorted.csv')
df_flourish = df_flourish.set_index('dist_hash').join(df_proc_time.set_index('dist_hash'))

# saving the data analysed
df_flourish.to_csv('dist_proc_time.csv')


# Analysing the increase in cases district wise
df_2010 = pd.read_csv('../input/precog-data-zip/data/cases/cases_2010.csv')
df_2018 = pd.read_csv('../input/precog-data-zip/data/cases/cases_2018.csv')
df_flourish = pd.read_csv('../input/1601-flourish-corrected/districts_sorted.csv')

# grouping by district
df_dist_2010 = df_2010.groupby(['state_code', 'dist_code'], as_index=False).count()
df_dist_2018 = df_2018.groupby(['state_code', 'dist_code'], as_index=False).count()

# creating a unique hash for each district
df_dist_2010['dist_hash'] = df_dist_2010['state_code']*80 + df_dist_2010['dist_code']
df_dist_2018['dist_hash'] = df_dist_2018['state_code']*80 + df_dist_2018['dist_code']

# Data clearing
df_dist_2010 = df_dist_2010[['dist_hash', 'ddl_case_id']]  # count is stored in ddl_case_id
df_dist_2010.rename(columns = {'ddl_case_id':'count_2010'}, inplace = True)
df_dist_2018 = df_dist_2018[['dist_hash', 'ddl_case_id']]
df_dist_2018.rename(columns = {'ddl_case_id':'count_2018'}, inplace = True)

# combining data to plot maps
df_flourish = df_flourish.join(df_dist_2018.set_index('dist_hash'), on='dist_hash')
df_flourish = df_flourish.join(df_dist_2010.set_index('dist_hash'), on='dist_hash')

# finding the increase in ratio cases
df_flourish['increase'] = 100.0*(df_flourish['count_2018']-(3.17*df_flourish['count_2010']))/df_flourish['count_2010'] # 3.17 is to adjust the data as there is more data for 2018
df_flourish = df_flourish.loc[df_flourish['increase']<5000]

# storing the analysed data
df_flourish.to_csv('inc_cases_districtwise.csv')