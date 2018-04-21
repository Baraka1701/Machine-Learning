
# coding: utf-8

# # Supervised methods checking different basic concepts for introduction ML course:

#    #####  __**Model Selection**__: Cross Validation types, Grid Search, some metrics.
# 
#    #####  __**Classification**__: Nearest Neighbors (Specially KNN), Naives Bayes Gaussian, SVM, Data Three and  Ensemble Methods: Random Forest and Boosting (ADB) and Neural Networks. 
# 
#    #####  __**Dimensionality Reduction**__: Principal Component Analysis (PCA) 
# 

# # Answer 1) 
# ## For a better comprehension, I choose __***BREAST CANCER***__ dataset

# In[1]:


#1) Importing libraries from dataset. Also, we charge the dataset
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().magic('matplotlib inline')
import time


# # Answer 2)
# 
# 2) That´s a dataset with **569 cases (X) of Observations** (row) with **30 attributes** (columns)
#  Each attribute is composed by **numeric discret** metrics of every one of the 30 features.
# 
# #**'feature_names'**: array(['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension','radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error','concave points error', 'symmetry error', 'fractal dimension error','worst radius', 'worst texture', 'worst perimeter', 'worst area','worst smoothness', 'worst compactness', 'worst concavity','worst concave points', 'worst symmetry', 'worst fractal dimension'],
# 
#  On other hand, **labels (y)** for each one are also 569 with a binary outpout: 
#  meaning 0 or 1 for indicates malignant or benign study case result.
#  
# Due to discrete values, we will consider it is suitable for **classification** problems.
# 
# We will parse our objetc to dataframe just for a better data exploration as follow:

# In[2]:


import pandas as pd

data = load_breast_cancer()


# In[4]:


df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)

df.head()


# In[31]:


df.info()


# In[33]:


df.describe()


# # Answer 3) "Playing" with experiments in classrom with other dataset and giving additional value

# ### First, we will create variables for X and y in order to apply diferent methods:

# In[3]:


x= data.data
y= data.target


# # **KNN**

# ###  STRATIFIED VS SHUFFLE

# 5) Choosing a stratified process, despite is not big sample, just 569 but with 30 features...so ensuring avoid repetitions of random shuffleSplit method.

# In[6]:


from sklearn.model_selection import StratifiedShuffleSplit
misss = StratifiedShuffleSplit(20, 0.2) #reducing test for ¿better train? Lets see

from sklearn.neighbors import KNeighborsClassifier
misKvecinos = KNeighborsClassifier(n_neighbors=3)

fallos= []# Failures count
index = 0 #Index initializated to 0

for train_index, test_index in misss.split(x,y):
    print(train_index)
    print(test_index)

    Xtrain = x[train_index, :]
    Xtest = x[test_index,:]
   
    Ytrain = y[train_index]
    Ytest = y[test_index]
   
    misKvecinos.fit(Xtrain,Ytrain)
    Ypred = misKvecinos.predict(Xtest)
    fallos.append(sum(Ypred != Ytest))
    index = index+1
    
print(fallos)
print("Error Coeficient is: "+ str(sum(Ypred != Ytest) / len(Ytest)))


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
misss = StratifiedShuffleSplit(20, 0.2) #reducing test for ¿better train? Lets see

from sklearn.neighbors import KNeighborsClassifier
misKvecinos = KNeighborsClassifier(n_neighbors=3)

fallos= []# Failures count
index = 0 #Index initializated to 0

for train_index, test_index in misss.split(x,y):
    Xtrain = x[train_index, :]
    Xtest = x[test_index,:]
    Ytrain = y[train_index]
    Ytest = y[test_index]
    misKvecinos.fit(Xtrain,Ytrain)
    Ypred = misKvecinos.predict(Xtest)
    fallos.append(sum(Ypred != Ytest))
    index = index+1
    
print(fallos)
print("Error Coeficient is: "+ str(sum(Ypred != Ytest) / len(Ytest)))


# # METRICS DIAGRAM

# ![Metrics.png](attachment:Metrics.png)

# ### Metrics for Stratified ShuffleSplit

# In[36]:



print(classification_report(Ytest,Ypred))


# 5) Let´s try with Shuffle AND compare with stratified

# In[13]:



from sklearn.model_selection import ShuffleSplit

miss= ShuffleSplit(20, 0.2)

for train_index, test_index in miss.split(x,y):
    Xtrain = x[train_index, :]
    Xtest = x[test_index,:]
    Ytrain = y[train_index]
    Ytest = y[test_index]
    misKvecinos.fit(Xtrain,Ytrain)
    Ypred = misKvecinos.predict(Xtest)
    fallos.append(sum(Ypred != Ytest))
    index = index+1

print("Error Coeficient is: "+ str(sum(Ypred != Ytest)/ (len(Ytest))))


# ### Metrics for ShuffleSplit

# In[6]:



print(classification_report(Ytest,Ypred))


# In[7]:


print( " Numero medio de errores  " + str(100*np.mean(fallos)/len(Ytest)))
print( " Desviacion Standar de errores " + str(100*np.std(fallos)/len(Ytest)))


# # Answer 5)
# 
# ##### According to our classification report in Stratified or just Shuffle (NO STRATIFIED), we can say that SHUFFLE as more precision for this dataset. Also, we can compare other indicators in metrics use. 
# 

# # Cross Validation Score using a Kfold model (misKvecinos)

# In[8]:


from sklearn.model_selection import cross_val_score
micvs = cross_val_score(misKvecinos,x,y, cv=20)

print("mean " +str(np.mean(micvs)))
print( "Std " +str(np.std(micvs)))


# I get a .9247 score using Kfolds cross validation, better than last time. Otherhand, **Standar Desviation has increased... 0.061**

# # Leave Out One (LOO), 
# 
# Just for comparing with before Iterators:

# In[9]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

loocv =model_selection.LeaveOneOut()
model= LogisticRegression()
results = model_selection.cross_val_score(model, x, y, cv=loocv)

print("mean "+str(np.mean(results)))
print(" Std "+str(np.std(results)))


# According to Library predictions, LOO has improve Bias (mean has increased) BUT Variance (Std) has increased too... Otherwise, LOO take more time; from we can infer a longer time calculating...We´ll do it in GridSearch using **start time function.**

# # Answer 4)
# 
# # Exhaustive search over specified parameter values for an estimator: GridSearchCV

# ### What we should do is, just for comparing, execute also Kfolds and LOO for comparing the CV parameter:
# 
# >Redefine misKvecinos with a no fix number of K.// Do as miKvecinos.
# 
# >According instructions, we will validate with leave-one-out despite Kfolds

#    ## 'e.g: kfolds'

# In[10]:


miKvecinos = KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV

mi_param_grid = {'n_neighbors' :[3,5,7,9,11,13,15],'weights':['uniform','distance']}
migscv = GridSearchCV(miKvecinos,mi_param_grid,cv=10,verbose=2)
start=time.time()
migscv.fit(x,y)
print("execution time for : " + str((time.time()-start)))
#We will select the best for accuracy.
miMejorKvecino = migscv.best_estimator_
miMejorKvecino.fit(x,y)
miMejorKvecino.score(x,y)


# That,s a score for our best KVecino combination with an execution time of 1.09588seconds...  Now, we will try with LOO

# ## 'e.g: LOO'

# In[11]:


loocv =model_selection.LeaveOneOut() # Introducing LOO as object
mi_param_grid = {'n_neighbors' :[3,5,7,9,11,13,15],'weights':['uniform','distance']}
migscv = GridSearchCV(miKvecinos,mi_param_grid,cv=loocv,verbose=2)
start=time.time()
migscv.fit(x,y)
print("Execution time with LOO is: "+ str((time.time()-start)))
#...LOO cross Validation, a Never Ended Compiling History...
miMejorKvecino = migscv.best_estimator_
miMejorKvecino.fit(x,y)
migscv.best_score_
miMejorKvecino.score(x,y)


# Despite the time in compiling is highly more, (60,62852 seconds) PROBABLY DIFFERENT when you RUN (;P),  the best scoring in diff:
#  n_neighbors=5, Weights='uniform0, best score=,9384
#  
# Now we will check what could be the best score (MiMejorVecino) and 
# the score for the best (MiMejorKvecino) is 0.94727
# 

# # Answer 6) optimized K neighbors and fit 'distance' in weights parametric

# In[12]:


miMejorKvecino = migscv.best_estimator_
print(miMejorKvecino)


# In[13]:


mi_param_grid = {'n_neighbors' : [5],'weights': ['distance']}
migscv = GridSearchCV(miKvecinos,mi_param_grid,cv=loocv,verbose=2)
migscv.fit(x,y)
miMejorKvecino = migscv.best_estimator_
miMejorKvecino.fit(x,y)
migscv.best_score_
miMejorKvecino.score(x,y)

print(miMejorKvecino)


# Conclusions: Using the method best_estimator_ for the best aproximation and just applying for distance (uniform weights), we achive the score 1 for miMejorKvecino.(Overfitted¿?)

# # Decomposition library, and visualizing the dataset

# In[6]:



from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


# In[7]:


miPCA = PCA(n_components = 2)
X_PCA = miPCA.fit_transform(x)


# In[8]:


print(miPCA.explained_variance_ratio_) # Obviously Axis=0 the 'x', is a better approximtaion to model with a STD of 98.2% over model


# In[17]:


plt.scatter(X_PCA[:,0],X_PCA[:,1],s=200,c=y)
plt.show()


# In[18]:


miKNN = KNeighborsClassifier(n_neighbors = 30)
miKNN.fit(X_PCA,y)
# And now we will create a "Grill" for visualize with a Z axis


xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),100),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),100))

Z = miKNN.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
plt.axis('tight')
plt.show()


# # Answer 7) Following the class explanation, we will change the "metric" parameter in KNeighborsClassifier trying to get still a better prediction:     'E.g: Euclidean'

# In[19]:


miKNN = KNeighborsClassifier(n_neighbors = 30, metric='euclidean')
miKNN.fit(X_PCA,y)

xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),100),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),100))

Z = miKNN.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# # Naives Bayes Gaussian 

# In[20]:


from sklearn.naive_bayes import GaussianNB

miGNB = GaussianNB()

micvs = cross_val_score(miGNB,x,y,cv=10)

print( "Mean: " + str(np.mean(micvs)))
print( "Std: " + str(np.std(micvs)))



# #### Gaussian Model is popular in Medical Dominio. Let´s compare the Gausan Naives Bayes model with this data refering to others...

# In[21]:


miGNB = GaussianNB()
miGNB.fit(x,y)

for train_index, test_index in misss.split(x,y):
    Xtrain = x[train_index, :]
    Xtest = x[test_index,:]
    Ytrain = y[train_index]
    Ytest = y[test_index]
    miGNB.fit(Xtrain,Ytrain)
    YpredNB = miGNB.predict(Xtest)
    

print(" That case we just have the next failures :"+ str (sum(YpredNB != Ytest)))


# In[22]:


tn, fp, fn, tp = confusion_matrix(y_true=Ytest, y_pred=YpredNB).ravel()
#print("true negatives" +tn, "false positives: "+fp, " false negatives: " +fn, "true positives: "+tp )

print("True Negatives: Predict Non Cancer which are no cancer" + str(tn))
print("False Positives: predict  Cancer that ARE NOT CANCER!" + str(fp))
print("False Negatives: Predict NON cancer that ARE CANCER!" + str(fn))
print("True Positives: Predict Cancer that, unfortunnately, are Cancer" + str(tp))


# ### Metrics for Gaussian Naives Bayes 

# In[23]:


print(classification_report(Ytest,YpredNB))


# #### After repeat the NB experiment, we can conclude that not big differences with KNN, but average of failures is lower for Naives Bays Gaussian and more stable behaivour repeating the experiment.  

# # SVM (Support Vector Machine)

# In[25]:


from sklearn.svm import SVC

miSVC = SVC(C=1000, gamma=100,kernel='rbf')
miSVC.fit(X_PCA,y)

xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),500),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),500))

Z = miSVC.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# In[26]:


for train_index, test_index in misss.split(x,y):
    Xtrain = x[train_index, :]
    Xtest = x[test_index,:]
    Ytrain = y[train_index]
    Ytest = y[test_index]
    miSVC.fit(Xtrain,Ytrain)
    YpredSVC = miSVC.predict(Xtest)
    
print(" That case we just have the next failures in % :"+ str ((sum(YpredSVC != Ytest)/ len(Ytest))*100))


# ### Sinceriously, it´s a very BAD fitting. So, let´s see in this case if possible playing with 2 parametres: C from 1 to 1000 and with kernel (Linear, Poli and RBF) for find our best.

# In[17]:


from sklearn.model_selection import GridSearchCV

#SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, 
#probability=False, tol=0.001, cache_size=200, class_weight=None, 
#verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)

param_grid = {'C': [1, 10, 100], 'kernel': ['linear']} #, 'poly', 'rbf'
miSVC = SVC()

migscv = GridSearchCV(miSVC, param_grid,cv=10,verbose=2) 
start=time.time()
migscv.fit(x,y)
print("execution time for : " + str((time.time()-start)))
#We will select the best for accuracy.
miMejorEstimador= migscv.best_estimator_
miMejorEstimador.fit(x,y)
miMejorEstimador.score(x,y)


# In[19]:


print(miMejorEstimador)


# In[21]:


print(miMejorEstimador.score(x,y))


# ### Let´s check if we can Show a new plot with last results:

# In[22]:


from sklearn.svm import SVC

miSVC = SVC(C=1, kernel='linear')
miSVC.fit(X_PCA,y)

xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),500),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),500))

Z = miSVC.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# # IMPORTANT NOTE: Obviously I should check now: 
# 
# > #### PARAMETERS: Kernel: POLY AND RBF and C from 0.001 to 1
# > #### CROSS VALIDATION ITERATORS: At least, repeat all done in KFolds with LOO...
# 
# ### FOR TIME AND _"JUST-INTRODUCTION-COURSE-PURPOUSES REASONS"_, WE WILL DO IT IN A FUTURE REVISION TO THIS EXPERIMENTS. 
# 
# ### OF COURSE IT´S ALSO VALID FOR THE OTHER MODELS.

#  

#  

# # Decission Tree

# In[26]:


from sklearn.tree import DecisionTreeClassifier

miDT = DecisionTreeClassifier()

from sklearn.model_selection import cross_val_score
micvs = cross_val_score(miDT,x,y,cv=10)

print("Mean: " + str(np.mean(micvs)))
print( "Std: " + str(np.std(micvs)))


# DecissionTree Clasifier has a low bias (mean) but a better variance ration. So, we can apply to take decissions in medical paradigma. How could we know how good is Decission Tree for a Cancer Breast Prediction? Let´s see a Confussion matrix.
# 
# ### Confussion Matrix in DT###

# In[27]:


from sklearn.metrics import confusion_matrix

miDT.fit(Xtrain, Ytrain)
YpredDt=miDT.predict(Xtest)


tn, fp, fn, tp = confusion_matrix(y_pred=YpredDt,y_true=Ytest).ravel()
#print("true negatives" +tn, "false positives: "+fp, " false negatives: " +fn, "true positives: "+tp )

print("True Negatives: Predict Non Cancer which are no cancer" + str(tn))
print("False Positives: predict  Cancer that ARE NOT CANCER!" + str(fp))
print("False Negatives: Predict NON cancer that ARE CANCER!" + str(fn))
print("True Positives: Predict Cancer that, unfortunnately, are Cancer" + str(tp))


# # So now, we will apply a model of DT combined with Decomposition Library, just for check the plotting and compare decission frontier with Kfolds model and KNN.

# In[28]:


miPCA = PCA(n_components = 2)
X_PCA = miPCA.fit_transform(x)

miDT = DecisionTreeClassifier()
miDT.fit(X_PCA,y)

xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),500),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),500))

Z = miDT.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# Obviously we can check a more "linear" decission frontier than in other models.. But visually overfitted.
# 
# CONCLUSIONS: Despite tried to "play" with max_depth parametre, not definetively conclusions about the best fitting. I tried with a range from 2 to 10 max_depth and several times with not a great variance in results...
# 
# How did I do? Sinceriously?? __PURE HEURISTIC WAY!__ and cross validation scores.
# 
# 1) I tried use the graphics PDF conversor with export_graphviz, but really was not able to...
# 
# 2) Despite parametric regulation, n_size input was fix...and I should consider look for optimal fitting value to in a future.
# 
# 3) Really had not more time and alone with the  this project...sorry 

# In[29]:



miDT = DecisionTreeClassifier(max_depth=5)

from sklearn.model_selection import cross_val_score
micvs = cross_val_score(miDT,x,y,cv=10)

print("Mean: " + str(np.mean(micvs)))
print( "Std: " + str(np.std(micvs)))


# # **ENSEMBLE METHODS IN CLASSIFICATION**

# ## Random Forest

# In[40]:


from sklearn.ensemble import RandomForestClassifier

miPCA = PCA(n_components = 2)
X_PCA = miPCA.fit_transform(x)

miRF = RandomForestClassifier(n_estimators=10)
miRF.fit(X_PCA,y)

xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),500),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),500))

Z = miRF.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# In[41]:


from sklearn.model_selection import cross_val_score
micvs = cross_val_score(miRF,x,y,cv=10)

print("Mean: " + str(np.mean(micvs)))
print( "Std: " + str(np.std(micvs)))


# #### Just exploring how max_depth for a better fitting model avoiding the overfitting (reducing depth) None value for max_depth make the model maximum depth of the tree. Then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples

# In[32]:


from sklearn.ensemble import RandomForestClassifier

miPCA = PCA(n_components = 2)
X_PCA = miPCA.fit_transform(x)

miRF = RandomForestClassifier(n_estimators=10,max_depth=3)
miRF.fit(X_PCA,y)

xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),500),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),500))

Z = miRF.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# In[33]:


from sklearn.model_selection import cross_val_score
micvs = cross_val_score(miRF,x,y,cv=10)

print("Mean: " + str(np.mean(micvs)))
print( "Std: " + str(np.std(micvs)))


# # BOOSTING

# #### The core principle of AdaBoost is to fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction

# In[34]:


from sklearn.ensemble import AdaBoostClassifier

miADB = AdaBoostClassifier(n_estimators=100)
start=time.time()
miADB.fit(X_PCA,y)
print("execution time for fitting in secs : " + str((time.time()-start)))


xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),500),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),500))

Z = miADB.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# In[35]:


for train_index, test_index in misss.split(x,y):
    Xtrain = x[train_index, :]
    Xtest = x[test_index,:]
    Ytrain = y[train_index]
    Ytest = y[test_index]
    miADB.fit(Xtrain,Ytrain)
    YpredADB = miADB.predict(Xtest)
    
print("Failures for this model are JUST! " + str (sum(YpredADB != Ytest)))


# In[36]:


print(classification_report(Ytest,YpredADB))


# #### Sumarizing, we make learn the model with weaks learners but in not just an error reinforce way. That makes that fitting n_estimator, we reach a really good prediction model. Otherwise, time for fitting is highly delayed than other models. 

# # NEURAL NETWORKS

# ##### Adding multiple layers, despite is still supervised, the model could predict new labels for new samples. Usually the parameter for fitting (predicting) has relation with layers 

# In[27]:


from sklearn.neural_network import MLPClassifier

miMLP = MLPClassifier(hidden_layer_sizes=(100,10),max_iter=1000)
miMLP.fit(X_PCA,y)

xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),500),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),500))

Z = miMLP.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha= 0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# In[30]:


for train_index, test_index in misss.split(x,y):
    Xtrain = x[train_index, :]
    Xtest = x[test_index,:]
    Ytrain = y[train_index]
    Ytest = y[test_index]
    miMLP.fit(Xtrain,Ytrain)
    YpredADB = miMLP.predict(Xtest)
    
print(" That case we just have the next failures in % :"+ str ((sum(YpredADB != Ytest)/ len(Ytest))*100))


# ## Let´s try to Fit it better searching for a best estimator...

# In[35]:


from sklearn.model_selection import GridSearchCV

# MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, 
# batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, 
# shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
# nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
# epsilon=1e-08)

# Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples 
# or more) in terms of both training time and validation score. 
# For small datasets LIKE THIS, however, ‘lbfgs’ can converge faster and perform better.

param_grid = {'hidden_layer_sizes': [(100,10), (200,50)], 'solver': ['lbfgs']} #, 'adam', 'sgd'
miMLP = MLPClassifier()

migscv = GridSearchCV(miMLP ,param_grid, cv=10) 
start=time.time()
migscv.fit(x,y)
print("execution time for : " + str((time.time()-start)))
#We will select the best for accuracy.
miMejorEstimador= migscv.best_estimator_
miMejorEstimador.fit(x,y)
miMejorEstimador.score(x,y)


# In[36]:


print(miMejorEstimador)


# In[38]:


from sklearn.neural_network import MLPClassifier

miMLP = MLPClassifier(hidden_layer_sizes=(200,50), solver='lbfgs')
miMLP.fit(X_PCA,y)

xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),500),np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),500))

Z = miMLP.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha= 0.4)
plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=y)
#plt.scatter(xx1,xx2,s=50,c=Z)
plt.axis('tight')
plt.show()


# In[39]:


for train_index, test_index in misss.split(x,y):
    Xtrain = x[train_index, :]
    Xtest = x[test_index,:]
    Ytrain = y[train_index]
    Ytest = y[test_index]
    miMLP.fit(Xtrain,Ytrain)
    YpredADB = miMLP.predict(Xtest)
    
print(" That case we just have the next failures in % :"+ str ((sum(YpredADB != Ytest)/ len(Ytest))*100))


# # Next revision, I should check now: 
# 
# > #### PARAMETERS: : hidden_layer_sizes, max_iter AND study other as solver: 'adam' and 'sgd'...etc
# > #### CROSS VALIDATION ITERATORS: At least, repeat all done in KFolds with LOO...
# 
# ### FOR TIME AND _"JUST-INTRODUCTION-COURSE-PURPOUSES REASONS"_, WE WILL DO IT IN A FUTURE REVISION TO THIS EXPERIMENTS. 
# 
# ### OF COURSE IT´S ALSO VALID FOR THE OTHER MODELS, BUT AT LEAST I REDUCE IN MIDDLE THE FAILURES (DESPITE OVERFITTED PLOT)
