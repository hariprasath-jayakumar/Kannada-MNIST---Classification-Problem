#!/usr/bin/env python
# coding: utf-8

# In[14]:


#-----------------------------------Importing Require Tools--------------------------------#

import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score,precision_score, f1_score,confusion_matrix,roc_curve, roc_auc_score
from yellowbrick.classifier import ROCAUC


# In[15]:


#URL = 'https://www.kaggle.com/datasets/higgstachyon/kannada-mnist'
#save the file in Local dir.


# In[16]:


#-------------------------------- Loading DataSet from npz file -----------------------------#

data_train = np.load(r"E:\Projects\Kannada MNIST\DataSet\X_train.npz")
data_test = np.load(r"E:\Projects\Kannada MNIST\DataSet\X_test.npz")
labels_train = np.load(r"E:\Projects\Kannada MNIST\DataSet\y_train.npz")
labels_test = np.load(r"E:\Projects\Kannada MNIST\DataSet\y_test.npz")


# In[17]:


#------------------------------- Extractig the arrays from the data---------------------------#

x_train = data_train['arr_0']
x_test = data_test['arr_0']
y_train = labels_train['arr_0']
y_test = labels_test['arr_0']


# In[18]:


#-------------------------------Reshaping the images to 1D Arrays ----------------------------#
# Loading DataSet from npz file.
# Reshape the images to 1D arrays (28x28 to 784)
X_train = x_train.reshape(-1, 28*28)
X_test = x_test.reshape(-1, 28*28)


# In[19]:


#------------------------------- Perform PCA to 10 components -------------------------------#
pca = PCA(n_components=10)
x_train_pca = pca.fit_transform(X_train)
x_test_pca = pca.transform(X_test)


# In[20]:


#-------------------- To perform in Random Componet sizes use Below -------------------------#

"""
component_sizes = [15, 20, 25, 30]

for size in component_sizes:
    pca = PCA(n_components=size)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
"""


# In[21]:


#DecisionTree
dt = DecisionTreeClassifier()
dt.fit(x_train_pca, y_train)

#RandomForest
rf = RandomForestClassifier()
rf.fit(x_train_pca, y_train)

#GaussianNB
gnb = GaussianNB()
gnb.fit(x_train_pca, y_train)

#K-NearestNeighbor
knn = KNeighborsClassifier()
knn.fit(x_train_pca, y_train)

#SVM
svm = SVC(probability=True)
svm.fit(x_train_pca, y_train)


# In[22]:


y_pred_dt = dt.predict(x_test_pca)
y_pred_rf = rf.predict(x_test_pca)
y_pred_gnb = gnb.predict(x_test_pca)
y_pred_knn = knn.predict(x_test_pca)
y_pred_svm = svm.predict(x_test_pca)


# To calculate the ROC-AUC score for a multiclass problem, you need to obtain the predicted probabilities for each class instead of the direct class predictions

# In[23]:


y_pred_dt_probabilities = dt.predict_proba(x_test_pca)
y_pred_rf_probabilities = rf.predict_proba(x_test_pca)
y_pred_gnb_probabilities = gnb.predict_proba(x_test_pca)
y_pred_knn_probabilities = knn.predict_proba(x_test_pca)
y_pred_svm_probabilities = svm.predict_proba(x_test_pca)


# In[24]:


def print_metrics(y_true, y_pred_probabilities):
    y_pred = np.argmax(y_pred_probabilities, axis=1)
    print("Precision:", precision_score(y_true, y_pred,average='micro'))
    print("Recall:", recall_score(y_true, y_pred,average='micro'))
    print("F1 Score:", f1_score(y_true, y_pred,average='micro'))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_true, y_pred_probabilities,multi_class='ovr'))


# In[25]:


# Decision Trees
print("Decision Trees:")
print_metrics(y_test, y_pred_dt_probabilities)

# Random Forest
print("\nRandom Forest:")
print_metrics(y_test, y_pred_rf_probabilities)

# GaussianNB
print("\nGaussianNB:")
print_metrics(y_test, y_pred_gnb_probabilities)

# K-NN Classifier
print("\nK-NN Classifier:")
print_metrics(y_test, y_pred_knn_probabilities)

# SVM Classifier
print("\nSVM Classifier:")
print_metrics(y_test, y_pred_svm_probabilities)


# In[27]:


#DecisionTree ROCAUC CURVE'
visualizer = ROCAUC(dt)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show()  

#RandomForest ROCAUC CURVE
visualizer = ROCAUC(rf)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show()  

#GaussianNB ROCAUC CURVE
visualizer = ROCAUC(gnb)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show() 

#K-NearestNeighbor ROCAUC CURVE
visualizer = ROCAUC(knn)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show() 

#SVM ROCAUC CURVE
visualizer = ROCAUC(svm)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)    
visualizer.show() 



# In[ ]:




