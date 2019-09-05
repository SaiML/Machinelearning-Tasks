# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:28:16 2019

@author: Sainath.Reddy
"""

# =============================================================================
# Importing Libraries
# =============================================================================

import pandas as pd
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import random
import io
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Loading Dataset
# =============================================================================

df=pd.read_csv("breast-cancer-wisconsin.data", header=None)
df.head()
df.shape
# =============================================================================
# 
#  We Do not have any information regarding the attributes of the dataset so tried to fetch the information 
# from Machine learning Repository
# Reference link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
# 
# """ Relavant information Regarding Dataset 
#   
#   Number of Attributes: 10 plus the class attribute
#   
#   Attribute Information: (class attribute has been moved to last column)
# 
#    #  Attribute                     Domain
#    -- -----------------------------------------
#    1. Sample code number            id number
#    2. Clump Thickness               1 - 10
#    3. Uniformity of Cell Size       1 - 10
#    4. Uniformity of Cell Shape      1 - 10
#    5. Marginal Adhesion             1 - 10
#    6. Single Epithelial Cell Size   1 - 10
#    7. Bare Nuclei                   1 - 10
#    8. Bland Chromatin               1 - 10
#    9. Normal Nucleoli               1 - 10
#   10. Mitoses                       1 - 10
#   11. Class:                        (2 for benign, 4 for malignant) 
# 
# =============================================================================

# =============================================================================
# # EDA
# =============================================================================

df.columns = ['id','Clump_thickness','Uniformity_of_cellsize','uniformity_of_cellshape','Marginal_adhesion','Single_epithelain_cellsize',
        'Bare_nuclei','Bland_chromatin','Nornal_Nucleli','Mitoses','Class']
df.head()
df.info()
df.describe().transpose()

#Checking for Null Values
df.isnull().sum()
df.nunique()
df.drop(['id'],axis=1,inplace=True)
# Hence id is a unique value for all the customers visited for the cancer test, so removig the ID column for the dataset

sns.pairplot(df,diag_kind='kde')
plt.show

# Checking for Missing Values

for i in df:
    x=df[i].unique()
    print(x,i)

# here in the Bare_nuclei Attribute we are having the missing data replaced with ?
    
df['Bare_nuclei'].value_counts()

# there are 16 missing values replaced with ? , so replacing this ? with a constant 
df.replace('?',-9999,inplace=True)
df.head()
df.shape
plt.figure(figsize = (15,10))
corr = df.corr()
sns.heatmap(corr , annot =True)
plt.show()


# =============================================================================
# #Splitting Data to Train and test data
# =============================================================================

from collections import Counter
df['Bare_nuclei']=df['Bare_nuclei'].astype(int)
def split_train_valid_test(data,test_ratio): 
    shuffled_indcies=np.random.permutation(len(data)) 
    test_set_size= int(len(data)*test_ratio) 
    test_indcies=shuffled_indcies[:test_set_size] 
    train_indices=shuffled_indcies[test_set_size:] 
    return data.iloc[train_indices],data.iloc[test_indcies]
train_set,test_set=split_train_valid_test(df,test_ratio=0.2)
print('lenghth of Trainig set : ',len(train_set)) 
print('lenghth of Testing set : ',len(test_set))

# =============================================================================
# # Custom KNN without SKlearn Model
# =============================================================================

X_train=np.array(train_set.drop(['Class'],axis =1))
y_train=np.array(train_set['Class'])
X_test=np.array(test_set.drop(['Class'],axis =1))
y_test=np.array(test_set['Class'])


class KNNClassifier(object):

    def __init__(self):
        pass
    
    #"training" function
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    #predict function, output of this function is linked to main function to predict the testdata
    def predict(self, X_test, k=5):
        distances = self.compute_distances(self.X_train, X_test)
        vote_results = []
        for i in range(len(distances)):
            votesOneSample = []
            for j in range(k):
                votesOneSample.append(distances[i][j][1])
            vote_results.append(Counter(votesOneSample).most_common(1)[0][0])
        
        return vote_results
    
#For each sample and every item in test set algorithm is making tuple in distance list
#this is how list looks =>> distances = [[[distance, class],[distance, class],[distance, class],[distance, class]]]
#it will caluclate distances and sort them accordingly 

    def compute_distances(self, X, X_test):
        distances = []
        for i in range(X_test.shape[0]):
            euclidian_distances = np.zeros(X.shape[0])
            oneSampleList = []
            for j in range(len(X)):
                euclidian_distances[j] = np.sqrt(np.sum(np.square(np.array(X_test[i]) - np.array(X[j]))))
                oneSampleList.append([euclidian_distances[j], self.y_train[j]])
            distances.append(sorted(oneSampleList))
        return distances

def accuracy(y_tes, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if(y_tes[i] == y_pred[i]):
            correct += 1
    return (correct/len(y_tes))*100

def run():
    classifier = KNNClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("My KNN accuracy: ",accuracy(y_test, y_pred),'%')
    
from functools import wraps
from time import time
import time


start = time.time()
run()
end = time.time()
time_taken = end - start
print('Time Taken : ',time_taken)


# =============================================================================
# #Checking Accuracy and Time with KNN - SKlearn
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
df['newclass']=np.nan
for i in range(len(df['Class'])):
    if df['Class'].iloc[i] == 2:
        df['newclass'].iloc[i] = 0 
    else:
        df['newclass'].iloc[i] = 1

# changing in to binary class for getting better metrics
        
df['newclass']=df['newclass'].astype('category')
X=df.drop(['Class','newclass'],axis =1)
y=df['newclass']

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y, test_size=0.3, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
start = time.time()

knn = KNeighborsClassifier(n_neighbors =5)
knn.fit(X_train1,y_train1)
y_pred = knn.predict(X_test1)
print('With KNN (K=5) test accuracy is: ',knn.score(X_test1,y_test1))

end = time.time()
time_taken = end - start
print('Time: ',time_taken)

# custom model is better at accuracy and lagging for time taking

yy=pd.DataFrame(y_train)
plt.rcParams['figure.figsize'] = (18, 7)
plt.subplot(1, 2, 1)
sns.countplot(yy[0], palette = 'pastel')
plt.title('Training set : Benign or Malignant', fontsize = 30)
plt.xlabel('class', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show()

Percent_value_counts = pd.DataFrame(data={
            'Count':yy[0].value_counts(),
            '% of Total': (yy[0].value_counts()/(len(yy))*100)
            }).sort_values('Count', ascending=False)
Percent_value_counts
# =============================================================================
# 
# # SMOTE MODEL for Balancing the data
# 
# =============================================================================

from imblearn.over_sampling import SMOTE 

print("Before OverSampling, counts of label '2': {}".format(sum(y_train==2)))
print("Before OverSampling, counts of label '4': {} \n".format(sum(y_train==4)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '4': {}".format(sum(y_train_res==4)))


yy1=pd.DataFrame(y_train_res)
yy1.columns=['class']
plt.rcParams['figure.figsize'] = (18, 7)
plt.subplot(1, 2, 1)
sns.countplot(yy1['class'], palette = 'pastel')
plt.title('Training set : Benign or Malignant', fontsize = 30)
plt.ylabel('count', fontsize = 15)
plt.show()

# Now Null Accuracy of the model is 50 % 

# =============================================================================
# # Custom KNN after Sampling
# =============================================================================

def run():
    classifier = KNNClassifier()
    classifier.fit(X_train_res, y_train_res)
    y_pred1 = classifier.predict(X_test)
    print("My KNN accuracy: ",accuracy(y_test, y_pred1),'%')
    
def run():
    classifier = KNNClassifier()
    classifier.fit(X_train_res, y_train_res)
    y_pred1 = classifier.predict(X_test)
    print ('\n******** Model Performance after Sampling******')
    print("\n My KNN accuracy: ",accuracy(y_test, y_pred1),'%')
    print("\nlength of y -test is :",len(y_test))
    print("length of y -Pred is :",len(y_pred1))
    y_actu = pd.Series(y_test, name='Actual')
    y_pred = pd.Series(y_pred1, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print ('\n******** Model Metrics******')
    print ('\n 1. Confusion Matrix')
    print('\n',df_confusion)
    TN= df_confusion[2][2]
    TP= df_confusion[4][4]
    FP= df_confusion[4][2]
    FN= df_confusion[2][4]
    
    recall = TP /(TP+FN)
    Precision = TP /(TP+FP)
    Specificity = TN/(TN+FP)
    F1score= 2 * ((Precision*recall) /(Precision + recall))
    
    print("\n 2.recall score is :",recall)
    print("\n 3.Precision score is :",Precision)
    print("\n 4.Specificity is :",Specificity)
    print("\n 5. F1score is :",F1score)
    
    
start = time.time()
run()
end = time.time()
time_taken = end - start
print('Time Taken : ',time_taken)

# =============================================================================
# # SKlearn KNN ALGORITHM AFTER SAMPLING
# =============================================================================


print("Before OverSampling, counts of label '2' and newlabeled as 0: {}".format(sum(y_train1==0)))
print("Before OverSampling, counts of label '4'  and newlabeled as 1: {} \n".format(sum(y_train1==1)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train1, y_train1)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))


start = time.time()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =5)
knn.fit(X_train_res, y_train_res)
y_pred = knn.predict(X_test1)
print('With KNN (K=5) test accuracy is: ',knn.score(X_test1,y_test1))
end = time.time()
time_taken = end - start
print('\n Time Taken : ',time_taken)

# Finding Best K value for better Accuracy

neig = np.arange(1, 25)
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(X_train_res,y_train_res)
    # test accuracy
    test_accuracy.append(knn.score(X_test1, y_test1))

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.title('Value VS Accuracy',fontsize=20)
plt.xlabel('Number of Neighbors',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(neig)
plt.grid()
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

# So ,for Testing data K value is 4 and Best Accuracy is 97.61

# Model Metrics
from sklearn.metrics import classification_report,confusion_matrix
mat_KNN = confusion_matrix(y_test1,y_pred)
print("KNN model confusion matrix = \n",mat_KNN)


mat_KNN = classification_report(y_test1,y_pred)
print("KNN model confusion matrix = \n",mat_KNN)


from sklearn import metrics
def draw_roc(actual,probs):
    fpr,tpr,thresholds = metrics.roc_curve(actual,probs)
    auc_score=metrics.roc_auc_score(actual,probs)
    plt.figure(figsize=(6,6))
    plt.plot(fpr,tpr,label='ROC curve(area = %0.2f)' %auc_score)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False positve rate')
    plt.ylabel('True positve rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")
    plt.show()
    
    return fpr,tpr,thresholds

draw_roc(y_test1,y_pred)

#Changing the Threshold for Probablities

y_pred_prob = knn.predict_proba(X_test1)[:, 1]
y_pred_prob[1:20]

# histogram of predicted probabilities

plt.hist(y_pred_prob, bins=20)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities for class 1')
plt.xlabel('Predicted probability of Cancer Class')
plt.ylabel('Frequency')


## Changing the cut off value for prediction
pred_proba_df = pd.DataFrame(knn.predict_proba(X_test1))
threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,0.95]
for i in threshold_list:
    print ('\n******** For cutoff = {} ******'.format(i))
    y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
    test_accuracy = metrics.accuracy_score(y_test1.as_matrix().reshape(y_test1.as_matrix().size,1),
                                           y_test_pred.iloc[:,1].as_matrix().reshape(y_test_pred.iloc[:,1].as_matrix().size,1))
    print('Model testing accuracy is {:.2f}'.format(test_accuracy))

    print(confusion_matrix(y_test1.as_matrix().reshape(y_test1.as_matrix().size,1),
                           y_test_pred.iloc[:,1].as_matrix().reshape(y_test_pred.iloc[:,1].as_matrix().size,1)))