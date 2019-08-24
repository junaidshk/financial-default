# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:03:33 2019

@author: junaid shaikh
"""

# =============================================================================
# Importing required libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#import xgboost
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,f1_score, recall_score, classification_report
from IPython.display import display # Allows the use of display() for DataFrames

# for jupyter -> %matplotlib inline

# =============================================================================
# Importing / Extracting data
# =============================================================================

try:
    loan_data = pd.read_csv(r'XYZCorp_LendingData.txt',
                    delimiter = "\t", engine = 'python')
    print("The loan dataset has {} samples with {} features.".format(*loan_data.shape))
except:
    print("The loan dataset could not be loaded. Is the dataset missing?")
    
# =============================================================================
# Analyzing / Exploring data
# Here we analyze the data to understand what necessary actions need to be taken
# so as to make the data suitable for our main objective.
# =============================================================================
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

loan_data.shape
loan_data.columns
loan_data.dtypes
loan_data.head()
loan_data.iloc[0]
#visualize value distribution of each column
loan_data.hist()
plt.show() 

labels = 'Default', 'Non-Default'
sizes = [1-(809502/855969), (809502/855969)]
colors = ['lightcoral', 'lightblue'] 
# Plot
plt.figure(figsize=(10,10))
plt.pie(sizes, colors=colors, autopct='%1.2f%%', shadow=False, startangle=0)
plt.title('Defaulted Vs Non-Defaulted', fontsize=12) 
plt.legend(labels, loc='lower left', fontsize=10)
plt.axis('equal')
plt.show()

####################################################
# Compute the correlation matrix
corr = loan_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": 0.7})
#######################################################

loan_data['default_ind'].value_counts()
#graphical representation of defaulters and non-defaulters in original dataset
loan_data['default_ind'].value_counts().plot(kind= 'barh', color = 'purple', title = 'Historical Loan Default Count', alpha = 0.55)
plt.show()

#analyzing columns with date type value
datecols = ['issue_d','last_credit_pull_d','last_pymnt_d','next_pymnt_d','earliest_cr_line']
loan_data[datecols].head(10)

# =============================================================================
# Treating improper or working on required data
# =============================================================================

## dropping desc because it describes information that is not necessary for our analysis.
loan_data_1 = loan_data.drop(['desc'],axis=1)

loan_data_1.describe()

## remove columns with more than 30% missing values as it would be time consuming
# and inefficient to deal with the tremendous amount of missing values from these columns.
## count 70% of the dataset and make it threshold for dropping column.
thresh_point = len(loan_data_1) * 0.7
loan_data_1 = loan_data_1.dropna(thresh=thresh_point, axis=1)
#drop duplicates if any
loan_data_1 = loan_data_1.drop_duplicates()
#52 coulms rem
loan_data_1.iloc[0]

# =============================================================================
# Feature Selection
# =============================================================================

#we understand that we can remove the id and member_id feature because it does not tell us anything about the person.
#funded_amnt and funded_amnt_inv are both features about the future the loan has been approved
#also grade, sub_grade is used for determining int_rate
#and also drop emp_title as it is unstructured column which requires different kind of algorithm altogether.

loan_data_1 = loan_data_1.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv','grade', 'sub_grade', 'emp_title','zip_code'], axis =1)
#drop other colums also which represent future value w.r.t the model
loan_data_1 = loan_data_1.drop(['out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv','total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'], axis =1)
loan_data_1 = loan_data_1.drop(['recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt'], axis =1)

#Single value columns are not information that help our model or minimum value count difference
orig_columns = loan_data_1.columns

drop_columns = []
for col in orig_columns:
    col_1 = loan_data_1[col].dropna().unique()
    if len(col_1) == 1:
        drop_columns.append(col)
        
loan_data_1['pymnt_plan'].value_counts()
drop_columns.append('pymnt_plan')
loan_data_1['collections_12_mths_ex_med'].value_counts()
drop_columns.append('collections_12_mths_ex_med')
loan_data_1['acc_now_delinq'].value_counts()
drop_columns.append('acc_now_delinq')

loan_data_1['initial_list_status'].value_counts()
loan_data_1['application_type'].value_counts()

drop_columns
loan_data_1 = loan_data_1.drop(drop_columns, axis = 1)


#Now we treat the remaining Null values by removing the rows
#as these are important feature without which the loan application is not meaningful.
null_counts = loan_data_1.isnull().sum()
null_counts

loan_data_3 = loan_data_1

loan_data_1 = loan_data_1.dropna(axis=0)
loan_data_1.shape

#Handling Non-Numeric Data Types
print(loan_data_1.dtypes.value_counts())

object_columns_df = loan_data_1.select_dtypes(include=["object"])
print(object_columns_df.iloc[0])

columns = ['term', 'emp_length', 'home_ownership', 'verification_status',
           'addr_state','initial_list_status','application_type']
for col in columns:
    print(loan_data_1[col].value_counts())
    print(" ")


## addr_state has many values and title is custom entry made by applicant hence both column to be removed
## and last_credit_pull_d and earliest_cr_line are date type. For now remove them and check.    
loan_data_1 = loan_data_1.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
    
loan_data_1 = loan_data_1.replace(mapping_dict)
print(loan_data_1['emp_length'].value_counts())
#note that now its dtype is int and not object.

object_columns_df = loan_data_1.select_dtypes(include=["object"])
print(object_columns_df.iloc[0])

##
#Encoding variables / creating dummy variable for categorical columns
##
categorical_columns = ["home_ownership", "verification_status", "purpose", "term", 
                       'initial_list_status', 'application_type']
dummy_df = pd.get_dummies(loan_data_1[categorical_columns])
loan_data_2 = pd.concat([loan_data_1, dummy_df], axis=1)
loan_data_2= loan_data_2.drop(categorical_columns, axis=1)

loan_data_2 = loan_data_2.replace(['?'], np.nan)
loan_data_2.isnull().sum()
# =============================================================================
# Data Partition as per date
# =============================================================================
loan_data_2.shape
loan_data_2.info()
loan_data_2.dtypes

#Below we check again, for count of defaulted and non-defaulted accounts by using the default_ind flag.
#loan_data_1.groupby(['default_ind'])['default_ind'].count()
loan_data_2['default_ind'].value_counts()

loan_data_2['default_ind'].value_counts().plot(kind= 'barh', color = 'purple', title = 'Historical Loan Default Count', alpha = 0.55)
plt.show()

labels = 'Default', 'Non-Default'
sizes = [1-(712962/747197), (712962/747197)]
colors = ['lightcoral', 'lightblue'] 
# Plot
plt.figure(figsize=(10,10))
plt.pie(sizes, colors=colors, autopct='%1.2f%%', shadow=False, startangle=0)
plt.title('Defaulted Vs Non-Defaulted', fontsize=12) 
plt.legend(labels, loc='lower left', fontsize=10)
plt.axis('equal')
plt.show()

corr = loan_data_2.corr()
#loan_data_2.groupby(['default_ind', 'issue_d'])['default_ind'].count()

loan_data_2['issue_date'] = pd.to_datetime(loan_data_2['issue_d'],infer_datetime_format=True, yearfirst=False)
train = loan_data_2[loan_data_2['issue_date'] < '2015-06-01'].drop(['issue_d'],axis=1)
test = loan_data_2[loan_data_2['issue_date'] >= '2015-06-01'].drop(['issue_d'],axis=1)

#loan_data_2 = loan_data_2.set_index(loan_data_2['issue_date'])
#loan_data_2 = loan_data_2.sort_index()

# create train test partition based on date condition.
#train_df = loan_data_2[:'2015-05-31']
#test_df  = loan_data_2['2015-06-01':]
train_df= train.drop('issue_date', axis=1)
test_df= test.drop('issue_date', axis=1)

#exporting clean and split data to local csv files.
#loan_data_2.to_csv(r'C:\Users\suhail shaikh\Desktop\Imarticus\Project\Python\loan_data_clean.csv')
#train_df.to_csv(r'C:\Users\suhail shaikh\Desktop\Imarticus\Project\Python\loan_data_train.csv')
#test_df.to_csv(r'C:\Users\suhail shaikh\Desktop\Imarticus\Project\Python\loan_data_test.csv')

print('Train Dataset:',train_df.shape)
print('Test Dataset:',test_df.shape)
train_df['default_ind'].value_counts()
test_df['default_ind'].value_counts()

X_train_df = train_df.drop('default_ind', axis=1)
X_test_df = test_df.drop('default_ind', axis=1)

#X_train = X_train_df.values[:,:] #excludes default_ind
#Y_train = train_df.values[:,[16]]  #include only default_ind
X_train = X_train_df
Y_train = train_df['default_ind']
#type(Y_train)

#X_test = X_test_df.values[:,:] #excludes default_ind
#Y_test = test_df.values[:,[16]]
X_test = X_test_df
Y_test = test_df['default_ind']
#type(Y_test)

Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)
#Y_train = pd.Series((train_df.values[:,[16]]).tolist())
#Y_test = pd.Series((test_df.values[:,[16]]).tolist())

# =============================================================================
# Scaling data
# =============================================================================
##Scaling -> Standardization
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

#scaler = StandardScaler()
scaler = RobustScaler()

#scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
type(X_train)

# =============================================================================
# Building Models
# =============================================================================
## 1.Logistic Regression
from sklearn.linear_model import LogisticRegression as lr
#X_train = downsampled.drop('default_ind',axis=1)
#Y_train = downsampled['default_ind']
#create a model
classifier = lr()
#fitting training data to the model
classifier.fit(X_train, Y_train)

#using trained data to predcit on test data
Y_pred = classifier.predict(X_test)
print(list(zip(Y_test, Y_pred)))
#above result can be checked on probabiluty matrix 'y_pred_prob' below.

print(classifier.coef_)
print(classifier.intercept_)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## 2.SVM

from sklearn import svm
classifier = svm.SVC(kernel = 'rbf', C = 1.0, gamma = 0.1)

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(list(Y_pred))

#exporting the list of prediction values
Y_pred_col = list(Y_pred)
#print(Y_Pred_col)
print(classifier.coef_)
print(classifier.intercept_)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## 3.Random Forest
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

# Train the model on training data
classifier = RandomForestClassifier(n_estimators=10).fit(X_train, Y_train)

# Use the forest's predict method on the test data
Y_pred = classifier.predict(X_test)

acc = accuracy_score(Y_test, Y_pred)
acc # 0.9977843274234952

f1 = f1_score(Y_test, Y_pred)
f1 # 014705882352941176

recall = recall_score(Y_test, Y_pred)
recall # 0.013937282229965157

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## 4. XGBoost

print('Initializing xgboost.sklearn.XGBClassifier and starting training...')

st = datetime.now()

clf = xgboost.sklearn.XGBClassifier(
    objective="binary:logistic", 
    learning_rate=0.05, 
    seed=9616, 
    max_depth=20, 
    gamma=10, 
    n_estimators=500)

clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=True)

print(datetime.now()-st)

y_pred = clf.predict(X_test)
submission_file_name = 'Submission_'

accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
print("Accuracy: %.10f%%" % (accuracy * 100.0))
submission_file_name = submission_file_name + ("_Accuracy_%.6f" % (accuracy * 100)) + '_'

accuracy_per_roc_auc = roc_auc_score(np.array(y_test).flatten(), y_pred)
print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))
submission_file_name = submission_file_name + ("_ROC-AUC_%.6f" % (accuracy_per_roc_auc * 100))

final_pred = pd.DataFrame(clf.predict_proba(np.array(finalTest)))
dfSub = pd.concat([test_member_id, final_pred.ix[:, 1:2]], axis=1)
dfSub.rename(columns={1:'loan_status'}, inplace=True)
dfSub.to_csv((('%s.csv') % (submission_file_name)), index=False)

import matplotlib.pyplot as plt
print(clf.feature_importances_)
idx = 0
for x in list(finalTrain):
    print('%d %s' % (idx, x))
    idx = idx + 1
plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# =============================================================================
# Evaluation of model 
# =============================================================================
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,f1_score, recall_score, classification_report

cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)
#LR    [[241385    241]
#       [   286      1]]
#LR_oversampling [[124666 116960]
#                 [    71    216]]
#RF    [[241373    253]
#       [   283      4]]


print("Classification report: ")

print(classification_report(Y_test, y_pred_class))

acc = accuracy_score(Y_test, y_pred_class)
print("Accuracy of the model: ", acc) # Common for all algorithm 0.9978215308809365
                                      # LR_sampling 0.5162269080206521
                                      # LR_downsampling 0.9368367966996399
                                      # RF_sampling 0.9988136230793715
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

# Adjusting the threshold / tuning the model #

#change the threshold
y_pred_class = []
for value in y_pred_prob[:,1]:
    if value > 0.55:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
        
#print(y_pred_class)

#check 
cfm = confusion_matrix(Y_test, y_pred_class)
print(cfm)
#LR [[241185    441]
#    [   283      4]]
#sampling_LR 
 #           [[226575  15051]
  #          [   229     58]]
#RF [[241373    253]
#    [   283      4]]

##
for a in np.arange(0,1,0.05):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])


# AUROC #

from sklearn import metrics

fpr, tpr, z = metrics.roc_curve(Y_test, y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc) #0.7074403667764139
 
#plot
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()

# =============================================================================
# Cross Validation
# =============================================================================

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean()) # LR 0.9328162417807061
                              # LR_sampling 0.5051218068287204
                              # RF 0.9321275206917525

#for evaluation run only till above.do not run below code as it takes more time
#for train_value, test_value in kfold_cv.split(X_train):
#    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])


#Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

# =============================================================================
# Oversampling for Overfiting
# =============================================================================
from sklearn.utils import resample


X = train_df

not_default = X[X.default_ind==0]
default = X[X.default_ind==1]

# upsample minority
default_upsampled = resample(default,

                          replace=True, # sample with replacement

                          n_samples=len(not_default), # match number in majority class

                          random_state=27) # reproducible results


# combine majority and upsampled minority
upsampled = pd.concat([not_default, default_upsampled])

# check new class counts
upsampled.default_ind.value_counts()

# =============================================================================
# Undersampling 
# =============================================================================
not_default_downsampled = resample(not_default,

                                replace = False, # sample without replacement

                                n_samples = len(default), # match minority n

                                random_state = 27) # reproducible results

# combine minority and downsampled majority

downsampled = pd.concat([not_default_downsampled, default])
# checking counts

downsampled.default_ind.value_counts()
# =============================================================================
# SMOTE
# =============================================================================
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=27, ratio=1.0)

X_train, y_train = sm.fit_sample(X_train, X_train)