#-----------------------------------Importing Require Tools--------------------------------#

import numpy as np
import os
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score, recall_score,precision_score, f1_score,confusion_matrix,roc_curve, roc_auc_score
from yellowbrick.classifier import ROCAUC


#URL = 'https://www.kaggle.com/datasets/higgstachyon/kannada-mnist'
#save the file in Local dir.




#-------------------------------- Loading DataSet from npz file -----------------------------#

data_train = np.load(r"E:\nlp\Kannada_MNIST\X_train.npz")
data_test = np.load(r"E:\nlp\Kannada_MNIST\X_test.npz")
labels_train = np.load(r"E:\nlp\Kannada_MNIST\y_train.npz")
labels_test = np.load(r"E:\nlp\Kannada_MNIST\y_test.npz")


#------------------------------- Extractig the arrays from the data---------------------------#

x_train = data_train['arr_0']
x_test = data_test['arr_0']
y_train = labels_train['arr_0']
y_test = labels_test['arr_0']


#-------------------------------Reshaping the images to 1D Arrays ----------------------------#
# Loading DataSet from npz file.
# Reshape the images to 1D arrays (28x28 to 784)
X_train = x_train.reshape(-1, 28*28)
X_test = x_test.reshape(-1, 28*28)


#------------------------------- Perform PCA to 10 components -------------------------------#
pca = PCA(n_components=10)
x_train_pca = pca.fit_transform(X_train)
x_test_pca = pca.transform(X_test)


#-------------------- To perform in Random Componet sizes use Below -------------------------#

"""
component_sizes = [15, 20, 25, 30]

for size in component_sizes:
    pca = PCA(n_components=size)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
"""


#-------------------------------- Decision Tree Classifier ----------------------------------#

dtc = DecisionTreeClassifier()

dtc.fit(x_train_pca, y_train)


#-------- > Training < -----------#

train_pred = dtc.predict(x_train_pca)

#--------- > Evaultion < ----------#

print('Train_Accuracy_Score  =',accuracy_score(y_train, train_pred))
print('Train_Precision_Score  =',precision_score(y_train,train_pred,average='weighted'))
print('Train_Recall_Score  =',recall_score(y_train,train_pred,average='weighted'))
print('Train_F1_Score  =',f1_score(y_train,train_pred,average='weighted'))
print('Train_Confusion_matrix  =',confusion_matrix(y_train,train_pred))
print('Train_ROC_AUC_Score  = ',roc_auc_score(y_train, dtc.predict_proba(x_train_pca), multi_class='ovr'))




#-------- > Testing  < -----------#

test_pred = dtc.predict(x_test_pca)


#--------- > Evaultion < ----------#

print('Test _Accuracy_Score  =',accuracy_score(y_test, test_pred))
print('Test_Precision_Score  =',precision_score(y_test,test_pred,average='weighted'))
print('Test_Recall_Score  =',recall_score(y_test,test_pred,average='weighted'))
print('Test_F1_Score  =',f1_score(y_test,test_pred,average='weighted'))
print('Test_Confusion_matrix  =',confusion_matrix(y_test,test_pred))
print('Test_ROC_AUC_Score  = ',roc_auc_score(y_test, dtc.predict_proba(x_test_pca), multi_class='ovr'))

visualizer = ROCAUC(dtc)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show()  


#-------------------------------- RandomForest Classifier ------------------------------------#


rf = RandomForestClassifier()

rf.fit(x_train_pca, y_train)


#-------- > Training < -----------#

train_pred = rf.predict(x_train_pca)

#--------- > Evaultion < ----------#

print('Train__Accuracy_Score  =',accuracy_score(y_train, train_pred))
print('Train_Precision_Score  =',precision_score(y_train,train_pred,average='weighted'))
print('Train_Recall_Score  =',recall_score(y_train,train_pred,average='weighted'))
print('Train_F1_Score  =',f1_score(y_train,train_pred,average='weighted'))
print('Train_Confusion_matrix  =',confusion_matrix(y_train,train_pred))
print('Train_ROC_AUC_Score  = ',roc_auc_score(y_train, rf.predict_proba(x_train_pca), multi_class='ovr'))


#-------- > Testing  < -----------#

test_pred = rf.predict(x_test_pca)

#--------- > Evaultion < ----------#

print('Test _Accuracy_Score  =',accuracy_score(y_test, test_pred))
print('Test_Precision_Score  =',precision_score(y_test,test_pred,average='weighted'))
print('Test_Recall_Score  =',recall_score(y_test,test_pred,average='weighted'))
print('Test_F1_Score  =',f1_score(y_test,test_pred,average='weighted'))
print('Test_Confusion_matrix  =',confusion_matrix(y_test,test_pred))
print('Test_ROC_AUC_Score  = ',roc_auc_score(y_test, rf.predict_proba(x_test_pca), multi_class='ovr'))

visualizer = ROCAUC(rf)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show()  


 
#--------------------------------------- GaussianNB --------------------------------------------#

GNB = GaussianNB()

GNB.fit(x_train_pca, y_train)


#-------- > Training < -----------#

train_pred = GNB.predict(x_train_pca)

#--------- > Evaultion < ----------#

print('Train _Accuracy_Score  =',accuracy_score(y_train, train_pred))
print('Train_Precision_Score  =',precision_score(y_train,train_pred,average='weighted'))
print('Train_Recall_Score  =',recall_score(y_train,train_pred,average='weighted'))
print('Train_F1_Score  =',f1_score(y_train,train_pred,average='weighted'))
print('Train_Confusion_matrix  =',confusion_matrix(y_train,train_pred))
print('Train_ROC_AUC_Score  = ',roc_auc_score(y_train, GNB.predict_proba(x_train_pca), multi_class='ovr'))


#-------- > Testing  < -----------#

test_pred = GNB.predict(x_test_pca)

#--------- > Evaultion < ----------#

print('Test _Accuracy_Score  =',accuracy_score(y_test, test_pred))
print('Test_Precision_Score  =',precision_score(y_test,test_pred,average='weighted'))
print('Test_Recall_Score  =',recall_score(y_test,test_pred,average='weighted'))
print('Test_F1_Score  =',f1_score(y_test,test_pred,average='weighted'))
print('Test_Confusion_matrix  =',confusion_matrix(y_test,test_pred))
print('Test_ROC_AUC_Score  = ',roc_auc_score(y_test, GNB.predict_proba(x_test_pca), multi_class='ovr'))


visualizer = ROCAUC(GNB)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show()  


#------------------------------------- KNN lassifier ------------------------------------------#

knn = KNeighborsClassifier()

knn.fit(x_train_pca, y_train)

#-------- > Training < -----------#

train_pred = knn.predict(x_train_pca)

#--------- > Evaultion < ----------#

print('Train_Accuracy_Score  =',accuracy_score(y_train, train_pred))
print('Train_Precision_Score  =',precision_score(y_train,train_pred,average='weighted'))
print('Train_Recall_Score  =',recall_score(y_train,train_pred,average='weighted'))
print('Train_F1_Score  =',f1_score(y_train,train_pred,average='weighted'))
print('Train_Confusion_matrix  =',confusion_matrix(y_train,train_pred))
print('Train_ROC_AUC_Score  = ',roc_auc_score(y_train, knn.predict_proba(x_train_pca), multi_class='ovr'))


#-------- > Testing  < -----------#

test_pred = knn.predict(x_test_pca)

#--------- > Evaultion < ----------#

print('Test _Accuracy_Score  =',accuracy_score(y_test, test_pred))
print('Test_Precision_Score  =',precision_score(y_test,test_pred,average='weighted'))
print('Test_Recall_Score  =',recall_score(y_test,test_pred,average='weighted'))
print('Test_F1_Score  =',f1_score(y_test,test_pred,average='weighted'))
print('Test_Confusion_matrix  =',confusion_matrix(y_test,test_pred))
print('Test_ROC_AUC_Score  = ',roc_auc_score(y_test, knn.predict_proba(x_test_pca), multi_class='ovr'))

visualizer = ROCAUC(knn)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show()  


#------------------------------------ SVC Classifier -----------------------------------------#

svm = SVC(probability=True)

svm.fit(x_train_pca, y_train)


#-------- > Training < -----------#

train_pred = svm.predict(x_train_pca)

#--------- > Evaultion < ----------#

print('Train _Accuracy_Score  =',accuracy_score(y_train, train_pred))
print('Train_Precision_Score  =',precision_score(y_train,train_pred,average='weighted'))
print('Train_Recall_Score  =',recall_score(y_train,train_pred,average='weighted'))
print('Train_F1_Score  =',f1_score(y_train,train_pred,average='weighted'))
print('Train_Confusion_matrix  =',confusion_matrix(y_train,train_pred))
print('Train_ROC_AUC_Score  = ',roc_auc_score(y_train, svm.predict_proba(x_train_pca), multi_class='ovr'))


#-------- > Testing  < -----------#

test_pred = svm.predict(x_test_pca)

#--------- > Evaultion < ----------#

print('Test _Accuracy_Score  =',accuracy_score(y_test, test_pred))
print('Test_Precision_Score  =',precision_score(y_test,test_pred,average='weighted'))
print('Test_Recall_Score  =',recall_score(y_test,test_pred,average='weighted'))
print('Test_F1_Score  =',f1_score(y_test,test_pred,average='weighted'))
print('Test_Confusion_matrix  =',confusion_matrix(y_test,test_pred))



visualizer = ROCAUC(svm)
visualizer.fit(x_train_pca, y_train)        
visualizer.score(x_test_pca, y_test)        
visualizer.show()  
