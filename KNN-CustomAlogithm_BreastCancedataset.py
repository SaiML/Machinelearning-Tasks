# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:28:16 2019

@author: Sainath.Reddy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random,time,warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE 
from sklearn import metrics
warnings.filterwarnings('ignore')


Data_cancer = pd.read_csv("breast-cancer-wisconsin.data", header=None)
Data_cancer.columns = ['id','Clump_thickness','Uniformity_of_cellsize','uniformity_of_cellshape','Marginal_adhesion','Single_epithelain_cellsize',
                       'Bare_nuclei','Bland_chromatin','Nornal_Nucleli','Mitoses','Class']
Data_cancer.drop(['id'],axis=1,inplace=True)
Data_cancer.replace('?',-9999,inplace=True)
Data_cancer['Bare_nuclei']=Data_cancer['Bare_nuclei'].astype(int)

# #Splitting Data to Train and test data

def split_train_valid_test(data,test_ratio): 
    shuffled_indcies=np.random.permutation(len(data)) 
    test_set_size= int(len(data)*test_ratio) 
    test_indcies=shuffled_indcies[:test_set_size] 
    train_indices=shuffled_indcies[test_set_size:] 
    return data.iloc[train_indices],data.iloc[test_indcies]

train_set,test_set=split_train_valid_test(Data_cancer,test_ratio=0.2)
X_train=np.array(train_set.drop(['Class'],axis =1))
y_train=np.array(train_set['Class'])
X_test=np.array(test_set.drop(['Class'],axis =1))
y_test=np.array(test_set['Class'])

# =============================================================================
# Custom KNN without SKlearn Model
# =============================================================================

class KNNClassifier():

    def __init__(self):
        pass
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X_test, k=5):
        distances = self.compute_distances(self.X_train, X_test)
        vote_results = []
        for i in range(len(distances)):
            votesOneSample = []
            for j in range(k):
                votesOneSample.append(distances[i][j][1])
            vote_results.append(Counter(votesOneSample).most_common(1)[0][0])
        return vote_results
    def compute_distances(self, X, X_test):
        distances = []
        for i in range(X_test.shape[0]):
            euclidian_distances = np.zeros(X.shape[0])
            oneSampleList = []
            for j in range(len(X)):
                euclidian_distances[j] = np.linalg.norm(np.array(X_test[i])-np.array(X[j])) 
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
    print(f"My KNN accuracy: {accuracy(y_test, y_pred)}")
    y_actu = pd.Series(y_test, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print('\n',df_confusion)
    TN= df_confusion[2][2]
    TP= df_confusion[4][4]
    FP= df_confusion[4][2]
    FN= df_confusion[2][4]
    recall = TP /(TP+FN)
    Precision = TP /(TP+FP)
    Specificity = TN/(TN+FP)
    F1score= 2 * ((Precision*recall) /(Precision + recall))
    
    return {'Recall score':recall, 'Precision score':Precision,'Specificity':Specificity,'F1Score' :F1score}
   

start = time.time()
run()
end = time.time()
time_taken = end - start
print(f"Time Taken for Custom KNN: {time_taken}")


# =============================================================================
# #Checking Accuracy and Time with KNN - SKlearn
# =============================================================================

Data_cancer['newclass']=np.nan
for i in range(len(Data_cancer['Class'])):
    if Data_cancer['Class'].iloc[i] == 2:
        Data_cancer['newclass'].iloc[i] = 0 
    else:
        Data_cancer['newclass'].iloc[i] = 1

# changing in to binary class for getting better metrics
        
Data_cancer['newclass']=Data_cancer['newclass'].astype('category')
X=Data_cancer.drop(['Class','newclass'],axis =1)
y=Data_cancer['newclass']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y, test_size=0.3, random_state=0)

start = time.time()
knn = KNeighborsClassifier(n_neighbors =5)
knn.fit(X_train1,y_train1)
y_pred = knn.predict(X_test1)
print(f"With SKlEARN KNN (K=5) test accuracy is {knn.score(X_test1,y_test1):.2f}")

end = time.time()
time_taken = end - start
print(f"Time taken for SK learn: {time_taken}:.2f")

# custom model is better at accuracy and lagging for time taking

# =============================================================================
# SMOTE MODEL for Balancing the data
# =============================================================================

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print(f"After OverSampling, counts of label '2': {sum(y_train_res==2)}")
print(f"After OverSampling, counts of label '4': {sum(y_train_res==4)}")

# =============================================================================
# Custom KNN after Sampling
# =============================================================================

def run():
    classifier = KNNClassifier()
    classifier.fit(X_train_res, y_train_res)
    y_pred1 = classifier.predict(X_test)
    print(f"My KNN accuracy after smapling : {accuracy(y_test, y_pred1):.2f}")

run()

# Model Metrics
mat_KNN = metrics.confusion_matrix(y_test1,y_pred)
print(f"KNN model confusion matrix :{mat_KNN}")

mat_KNN1 = metrics.classification_report(y_test1,y_pred)
print(f"KNN model confusion matrix : {mat_KNN1}")

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
    print(f"Model testing accuracy is {test_accuracy:.2f}")

    print(confusion_matrix(y_test1.as_matrix().reshape(y_test1.as_matrix().size,1),
                           y_test_pred.iloc[:,1].as_matrix().reshape(y_test_pred.iloc[:,1].as_matrix().size,1)))
