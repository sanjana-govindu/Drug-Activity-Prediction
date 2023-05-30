#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.combine import SMOTEENN,SMOTETomek

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[2]:


#Loading Training dataset
with open("/Users/sanjanagovindu/Downloads/584_HW3/DM Asst3/traindata.csv", "r") as trainData:
    train = trainData.readlines()

#Loading Test dataset
with open("/Users/sanjanagovindu/Downloads/584_HW3/DM Asst3/testdata.csv", "r") as testData:
    test = testData.readlines()


# In[3]:


train_list = []
train_label = []


# In[4]:


for data in train:
    train_label.append(data[0])
#     print(data[0])
    #Remove new line and activity label - 0/1 from each row
    data = data.replace("\n", "")
    data = data.replace("0\t", "")
    data = data.replace("1\t", "")
    train_list.append(data)
print(np.array(train_list).shape)


# In[5]:


#Convert the given training and test data into matrix
def convert_matrix_split(data):
    feat_range = 100000
    sp_matrix = pd.DataFrame(columns=range(feat_range))
    l = len(data)
    for i in range(l):
        xarr = [0 for j in range(feat_range)]
        for k in np.fromstring(data[i], dtype=int, sep=' '):
            xarr[k-1] = 1
        sp_matrix.loc[i] = xarr
    return sp_matrix


# In[6]:


train_data = convert_matrix_split(train_list)
test_data = convert_matrix_split(test)
y_train = np.asarray(train_label) #convert the input train_label into an array


# In[7]:


Y_train = [int(k) for k in y_train]
#Converting the 0s in train file to -1s 
#-> The target variable has to be binary so it has to be encoded and the values are either -1 or 1 
for i in range(len(y_train)):
    if(Y_train[i]==0):
        Y_train[i]=-1


# In[8]:


y_train = Y_train


# In[9]:


len(train_data)


# In[10]:


len(y_train)


# In[11]:


def reduceDimentionality(train_data, test_data):
    #Applying PCA - Principal Component Analysis to reduce dimentionality
    #red_dim = PCA(n_components=1000)
    
    #Applying SVD -  Singular Value Decomposition to reduce dimentionality
    #svd = TruncatedSVD(n_components=1000)
    red_dim = TruncatedSVD(n_components=500, random_state=42)

    train_vector = red_dim.fit_transform(train_data)
    test_vector = red_dim.transform(test_data)
    return train_vector, test_vector


# In[12]:


def cal_err(x, x_pred, y_i):
    return (sum(y_i * (np.not_equal(x, x_pred)).astype(int)))/sum(y_i) #Calculating the error rate

def new_weigh(y_i, alpha, x, x_pred): 
     return y_i * np.exp(alpha * (np.not_equal(x, x_pred)).astype(int)) 


# In[37]:


#Implementation of Adaboost code
class AdaBoost:
    
    def __init__(self):
        self.G_M = []
        self.M = None
        self.alpha = []
        self.predict_err = []
        self.train_error = []
        
    def fit(self, X, y, M = 100):
       
        self.M = M
        self.alpha = [] 
        self.train_error = []

        for k in range(0, M): #iterate with M weak classifiers
            if k != 0:
                weigh = new_weigh(weigh, alpha, y, y_pred)
            else:
                weigh = np.ones(len(y)) * 1 / len(y)  
            
            classify = DecisionTreeClassifier(max_depth = 5)   #usiing decision tree classifier
            classify.fit(X, y, sample_weight = weigh)
            y_pred = classify.predict(X)
            
            self.G_M.append(classify) #append the list with weak classifiers

            error = cal_err(y, y_pred, weigh) #computer error
            alpha = np.log((1 - error) / error) 
            self.train_error.append(error)
            self.alpha.append(alpha) 

    def predict(self, X):
        preds_w = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 
        for i in range(self.M):
            y_pred = self.G_M[i].predict(X) * self.alpha[i] #class label for each classifier
            preds_w.iloc[:,i] = y_pred
        return (1 * np.sign(preds_w.T.sum())).astype(int) #return final predictions


# In[14]:


#Dealing with imbalanced data using SMOTE and SMOTE Tomek
smoteTomek = SMOTETomek(random_state=42)
train_vector, y_train = smoteTomek.fit_resample(train_data, y_train)

# smote = SMOTE(random_state = 42)
# train_vector, y_train = smote.fit_resample(train_vector, y_train)

#Dimentionality Reduction using SVD technique
train_vector, test_vector = reduceDimentionality(train_vector, test_data)


# In[38]:


#Implementing AdaBoost class here
model = AdaBoost()
model.fit(train_vector, y_train, M = 200)

#Predict on test set
y_pred = model.predict(test_vector)


# In[35]:


for i in range(len(y_pred)):  #Converting back the -1s to 0s in final predictions file 
    if(y_pred[i]==-1):
        y_pred[i]=0


# In[36]:


y_pred.tolist().count(1) #Counting the number of 1s in the predictions file


# In[18]:


#Cross validation 
# scores_list=[]
# for k in [1,42,55,173,100]:
#     X_train, X_test, y_train, y_test = train_test_split(train_vector, y_train, random_state=k)
#     boost = AdaBoostFunction()
#     score = f1_score(y_test, boost['y_pred'])
#     scores_list.append(score)
# sum = sum(scores_list)/5


# In[19]:


#Output file with predictions for F1-score calculation
out = open('/Users/sanjanagovindu/Downloads/output.csv', 'w')

out.writelines( "%s\n" % x for x in y_pred)

out.close()


# In[20]:


print(len(y_pred)) #No of rows in output file - 350 rows

