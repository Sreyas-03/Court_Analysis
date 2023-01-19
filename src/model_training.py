import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


# Function to build the model
def build_model(input_d, l1_nodes, l2_nodes):
    model = Sequential()
    model.add(Dense(l1_nodes, input_dim=input_d, activation='relu'))
    model.add(Dense(l2_nodes, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


######################## MODEL 1 #############################
# Load the data
df = pd.read_csv('../input/smalldataset/acts_sections1000000.csv')
# Preprocess the data
df['bailable_ipc'] = df["bailable_ipc"].fillna('3')
df = df[['section', 'act', 'bailable_ipc','criminal']]
df = df.dropna()  # Drop rows with missing values
X = df[['section', 'act', 'bailable_ipc']]  # Select features
y = df['criminal']  # Select target
print(X.head(10))



# Convert categorical variables to numerical form
le = LabelEncoder()
X['section'] = le.fit_transform(X['section'])
X['act'] = le.fit_transform(X['act'])
X['bailable_ipc'] = le.fit_transform(X['bailable_ipc'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Build the model
model = build_model(3, 10, 10)

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=10)

# Evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save("criminal_ActsSections.h5")


# Loading model from directory
model = load_model('../input/model-asbc/model_criminal_ActsSections.h5')

# Unbiased data to test on
df = pd.read_csv('../input/smalldataset/acts_sections1000000.csv')

# Preprocess the data
df['bailable_ipc'] = df["bailable_ipc"].fillna('3')
df = df[['section', 'act', 'bailable_ipc','criminal']]
df = df.dropna()  # Drop rows with missing values
X = df[['section', 'act', 'bailable_ipc']]  # Select features
y = df['criminal']  # Select target

le = LabelEncoder()
X['section'] = le.fit_transform(X['section'])
X['act'] = le.fit_transform(X['act'])
X['bailable_ipc'] = le.fit_transform(X['bailable_ipc'])

scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

###############################################################


##################### MODEL 2 #################################

# creating the dataframe
df_cases = pd.read_csv('../input/precog-data-zip/data/cases/cases_2018.csv')

# Preprocessing the data
df_cases = df_cases.loc[df_cases['disp_name'] != 1]

# Defining the punishable and non-punishable dispositions
df_cases['disp_name'] = df_cases['disp_name'].replace([4, 23, 12, 16, 44, 52],[0, 0, 0, 0, 0, 0])
df_cases['disp_name'] = df_cases['disp_name'].replace([9, 10, 11, 17, 20, 29, 30, 39, 40],[1, 1, 1, 1, 1, 1, 1, 1, 1])
df_cases = df_cases.loc[df_cases['disp_name'].isin([0, 1])]


#linking case data with judge data
df_merge_jc = pd.read_csv('../input/precog-data-zip/data/keys/judge_case_merge_key.csv')
df_cases = df_cases.join(df_merge_jc.set_index('ddl_case_id'), on='ddl_case_id')

df_judge = pd.read_csv('../input/precog-data-zip/data/judges_clean.csv')
df_judge = df_judge[['ddl_judge_id','female_judge', 'start_date', 'end_date']]
df_cases = df_cases.join(df_judge.set_index('ddl_judge_id'), on='ddl_filing_judge_id')


# picking the necessary attributes to train the model
X = df_cases[['state_code', 'dist_code', 'court_no',
 'judge_position', 'female_defendant', 'female_petitioner', 'female_adv_def',
 'female_adv_pet', 'date_of_filing', 'date_first_list', 'date_last_list', 'date_next_list',
 'female_judge']]
y= df_cases['disp_name']


# Convert categorical variables to numerical form
le = LabelEncoder()
X['state_code'] = le.fit_transform(X['state_code'])
X['dist_code'] = le.fit_transform(X['dist_code'])
X['court_no'] = le.fit_transform(X['court_no'])
X['judge_position'] = le.fit_transform(X['judge_position'])
X['female_defendant'] = le.fit_transform(X['female_defendant'])
X['female_petitioner'] = le.fit_transform(X['female_petitioner'])
X['female_adv_def'] = le.fit_transform(X['female_adv_def'])
X['female_adv_pet'] = le.fit_transform(X['female_adv_pet'])
X['date_of_filing'] = le.fit_transform(X['date_of_filing'])
X['date_first_list'] = le.fit_transform(X['date_first_list'])
X['date_last_list'] = le.fit_transform(X['date_last_list'])
X['date_next_list'] = le.fit_transform(X['date_next_list'])
X['female_judge'] = le.fit_transform(X['female_judge'])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build the model
build_model(13, 20, 10)

# Train the model
model.fit(X_train, y_train, epochs=3, batch_size=20)

# Evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save("punishable.h5")