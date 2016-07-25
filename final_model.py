'''
author : jalFaizy
'''

# define modules
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split

# define data directories
root_dir = os.path.abspath('..')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

train = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))
sample_sub = pd.read_csv(os.path.join(sub_dir, 'Sample_Submission.csv'))

################################
## make feature Applicant_Experience
repeat_app = []
for idx, row in train.iterrows():
    if row.Applicant_BirthDate in train.ix[0:idx-1, 'Applicant_BirthDate'].values:
        repeat_app.append(1)
    else:
        repeat_app.append(0)
        
repeat_app = np.array(repeat_app, dtype=int)

train['Applicant_Experience'] = repeat_app

repeat_app = []
for idx, row in test.iterrows():
    if row.Applicant_BirthDate in train.ix[:, 'Applicant_BirthDate'].values or             row.Applicant_BirthDate in test.ix[:idx-1, 'Applicant_BirthDate'].values:
        repeat_app.append(1)
    else:
        repeat_app.append(0)
        
repeat_app = np.array(repeat_app, dtype=int)
test['Applicant_Experience'] = repeat_app
###################################

# drop missing rows
train = train.ix[~train.Manager_Business.isnull(), :]

######################################
## make feature Manager_Experience, Manager_AllTime_Business
repeat_man = []
repeat_man_sum = []
for idx, row in train.iterrows():
    if row.Manager_DoB in train.ix[0:idx-1, 'Manager_DoB'].values:
        repeat_man.append(1)
        
        current_train = train.ix[0:idx-1, :]
        temp_sum = current_train[current_train.Manager_DoB == row.Manager_DoB].Manager_Business.sum()
        repeat_man_sum.append(temp_sum)
    else:
        repeat_man.append(0)
        repeat_man_sum.append(0)
        
repeat_man = np.array(repeat_man, dtype=int)
repeat_man_sum = np.array(repeat_man_sum, dtype=int)

train['Manager_Experience'] = repeat_man
train['Manager_AllTime_Business'] = repeat_man_sum

repeat_man = []
repeat_man_sum = []
for idx, row in test.iterrows():
    if row.Manager_DoB in test.ix[0:idx-1, 'Manager_DoB'].values or             row.Manager_DoB in train.ix[:, 'Manager_DoB'].values:
        repeat_man.append(1)
        
        current_test = test.ix[0:idx-1, :]
        current_all = pd.concat([train, current_test], axis = 0)
        temp_sum = current_all[current_all.Manager_DoB == row.Manager_DoB].Manager_Business.sum()
        repeat_man_sum.append(temp_sum)
    else:
        repeat_man.append(0)
        repeat_man_sum.append(0)
        
repeat_man = np.array(repeat_man, dtype=int)
repeat_man_sum = np.array(repeat_man_sum, dtype=int)

test['Manager_Experience'] = repeat_man
test['Manager_AllTime_Business'] = repeat_man_sum

# define features to include
train_cols = ['Office_PIN', 'Applicant_City_PIN', 'Applicant_Gender', 'Applicant_Marital_Status', 'Applicant_Occupation', 
             'Applicant_Qualification', 'Manager_Joining_Designation', 'Manager_Current_Designation', 
             'Manager_Status', 'Manager_Gender', 'Manager_Num_Application', 'Manager_Num_Coded', 
             'Manager_Business', 'Manager_Num_Products', 'Manager_Business2', 'Manager_Num_Products2',
             'Applicant_BirthDate', 'Manager_DOJ', 'Manager_DoB',
             'Applicant_Experience', 'Manager_Experience', 'Manager_AllTime_Business']

data_x = train.ix[:, train_cols]
data_y = train.Business_Sourced.values

data_x_test = test.ix[:, train_cols]

# label encode categorical columns
cat_cols = data_x.columns[data_x.dtypes == 'object']
cat_cols = cat_cols.drop(['Applicant_BirthDate', 'Manager_DOJ', 'Manager_DoB',])

lb = LabelEncoder()
for var in cat_cols:
    full_data = pd.concat((data_x[var],data_x_test[var]),axis=0).astype('str')
    lb.fit(full_data )
    data_x[var] = lb.transform(data_x[var].astype('str'))
    data_x_test[var] = lb.transform(data_x_test[var].astype('str'))


# fill remaining missing values	
data_x.fillna(data_x.mean(), inplace=True);
data_x_test.fillna(data_x.mean(), inplace=True);

# one hot encode categorical columns
for var in cat_cols:
    enc = OneHotEncoder(sparse=False)
    
    var_temp = data_x[var].reshape(-1, 1)
    var_temp_test = data_x_test[var].reshape(-1, 1)
    
    full_data = pd.concat((data_x[var],data_x_test[var]),axis=0).reshape(-1, 1)
    enc.fit(full_data)
    
    temp = enc.transform(var_temp)
    temp_test = enc.transform(var_temp_test)
    
    temp_cols = []
    for col_name in enc.active_features_:
        temp_cols.append(var + str(col_name))
    
    temp = pd.DataFrame(temp, columns=temp_cols, index=data_x.index)
    temp_test = pd.DataFrame(temp_test, columns=temp_cols, index=data_x_test.index)
    
    data_x = pd.concat([data_x, temp], axis=1)
    data_x_test = pd.concat([data_x_test, temp_test], axis=1)
    
data_x.drop(cat_cols, axis=1, inplace=True)
data_x_test.drop(cat_cols, axis=1, inplace=True)

# parse time columns
time_cols = ['Applicant_BirthDate', 'Manager_DOJ', 'Manager_DoB']

for var in time_cols:
    data_x[var] = pd.to_datetime(data_x[var])
    data_x_test[var] = pd.to_datetime(data_x_test[var])

#########################################
##make features years till now
now = pd.datetime.now()
for var in time_cols:
    data_x[var + '_Year'] = (now - data_x[var]).astype('<m8[Y]')
    data_x_test[var + '_Year'] = (now - data_x_test[var]).astype('<m8[Y]')

data_x.drop(time_cols, axis=1, inplace=True)
data_x_test.drop(time_cols, axis=1, inplace=True)
############################################

# do filling again! (just to be sure)
data_x.fillna(data_x.mean(), inplace=True);
data_x_test.fillna(data_x.mean(), inplace=True);

# clean number 2 businesses
data_x.Manager_Business2 = data_x.Manager_Business2 - data_x.Manager_Business
data_x.Manager_Num_Products2 = data_x.Manager_Num_Products2 - data_x.Manager_Num_Products

data_x_test.Manager_Business2 = data_x_test.Manager_Business2 - data_x_test.Manager_Business
data_x_test.Manager_Num_Products2 = data_x_test.Manager_Num_Products2 - data_x_test.Manager_Num_Products

# change Applicant_City_PIN to distance from office
data_x.Applicant_City_PIN = np.sqrt(abs(np.power(data_x.Applicant_City_PIN, 2) - np.power(data_x.Office_PIN, 2)))
data_x_test.Applicant_City_PIN = np.sqrt(abs(np.power(data_x_test.Applicant_City_PIN, 2) - np.power(data_x_test.Office_PIN, 2)))

###############################################
##make feature "have they worked for the company before?"
data_x['feat3'] = data_x.Applicant_Experience + data_x.Manager_Experience
data_x_test['feat3'] = data_x_test.Applicant_Experience + data_x_test.Manager_Experience
###############################################

# local validation
split_data = data_x
split_size = int(data_x.shape[0]*0.8)

x_train, y_train = split_data.iloc[:split_size, :], data_y[:split_size]
x_val, y_val = split_data.iloc[split_size:, :], data_y[split_size:]

# define test model
clf1 = GradientBoostingClassifier(n_estimators=250)

clf1.fit(x_train, y_train);
print roc_auc_score(y_val, clf1.predict_proba(x_val)[:, 1])

# define real model and predict
clf = GradientBoostingClassifier(n_estimators=250)
clf.fit(data_x, data_y);
pred = clf.predict_proba(data_x_test)
pd.DataFrame({'ID':test.ID, 'Business_Sourced':pred[:, 1]}).to_csv(os.path.join(sub_dir, 'sub_final.csv'), index=False)