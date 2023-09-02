# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 09:22:06 2023

@author: domingosdeeulariadumba
"""




""" IMPORTING LIBRARIES """


# EDA and Plotting

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Machine Learning Modules

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE, ADASYN as AD
from imblearn.combine import SMOTETomek as ST, SMOTEENN as SE
from sklearn.metrics import confusion_matrix as cm, classification_report as cr
from sklearn.metrics import roc_auc_score as ras, roc_curve as rcv


# To save the ML model

import joblib as jb


# For statistical summary

import statsmodels.api as smd


# For ignoring warnings

import warnings
warnings.filterwarnings('ignore')



"""" EXPLORATORY DATA ANALYSIS """

  
    '''
    Importing the dataset...
    '''
df_bnk = pd.read_csv("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction2/BankDataset_full.csv",
                 sep = ';')

    '''
    Checking the main info of the dataset...
    '''
df_bnk.info()

    '''
    Statistical summary...
    '''
df_bnk.describe()

    '''
    Presenting the first and last ten entries...
    '''
df_bnk.head(10)
df_bnk.tail(10)

   '''
   Distribution plot.
   For this step we split the attributes into two different lists, one for 
   numerical ('NumList') and other for categorical values ('ObjList').
   '''
ObjList = []
NumList = []

for i in list(df_bnk.columns):
    if df_bnk[i].dtype == 'O':
        ObjList.append(i)
    else:
        NumList.append(i)
 
for j in ObjList[:-1]:
    plt.figure()
    sb.set(rc = {'figure.figsize':(16,12)}, font_scale = 1)
    sb.displot(x = df_bnk[j])
    plt.xticks(rotation = 15)
    plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction2/{0}_Cat.barplot.png'.format(j))
    plt.close()

for j in NumList:
    plt.figure()
    sb.set(rc = {'figure.figsize':(16,12)}, font_scale = 1)
    sb.displot(x = df_bnk[j])
    plt.xticks(rotation = 15)
    plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction2/{0}_Num.displot.png'.format(j))
    plt.close()

   '''
   Concentration of different attributes on the outcome.
   '''
for q in list(df_bnk.drop('y', axis = 1).columns):
    if np.unique(df_bnk[q]).size < 10:
        plt.figure()
        sb.set(rc = {'figure.figsize':(16,12)}, font_scale = 1)
        pd.crosstab(df_bnk[q], df_bnk.y).plot(kind = 'bar')
        plt.xticks(rotation = 15)
        plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction2/{0}_Num.compbarplot.png'.format(q))
        plt.close()
        
   '''
   Conversion Rate Pie Chart.
   '''
conversion = df_bnk['y'].value_counts().reset_index(name = 'conversion')

conv_colors = []

for k in conversion['index']:
    if k == 'no':
        conv_colors.append('red')
    else:
        conv_colors.append('green')
      
plt.figure()
sb.set(rc = {'figure.figsize':(6,6)}, font_scale = 1.5)
conversion.groupby('index').sum().plot(kind = 'pie', colors = conv_colors,
                                     y= 'conversion', autopct='%1.0f%%',
                                     explode = (0.00, 0.15))
plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction2/coversion.piechart.png')
plt.close()


    '''
    Mean values of each feature to the campaign result.
    '''
df_bnk.groupby('y').mean()




""" Prediction Model"""



    '''
    Hot encoding and splitting the dataset.
    '''
X = pd.get_dummies(df_bnk.drop('y', axis = 1), drop_first = True)

y = df_bnk['y']

for i in range(len(y)):
    if y[i] == 'yes':
        y[i] = 1
    else:
        y[i] = 0

Y_Final = pd.DataFrame(np.asarray(y, dtype = np.uint8))
   
X_train, X_test, y_train, y_test = tts(X,Y_Final, test_size = 0.2,
                                       random_state = 7)


    '''
    As the pie chart illustrates, the dataset is severely imbalanced. Making
    predictions in this conditions can easily generate biased results (in this
    case, for every 1000 prospects, the model can easily predict that about 120
    of them will subscribe the campaign). To address this issue we'll use four
    broadly recommended resampling methods to deal with imbalanced data. Two of
    them are oversampling techiques, namely: Synthetic Minority Oversampling 
    Technique (SMOTE) and Adaptive Synthetic (ADASYN). The last two are hybrid
    techniques (involving simultaneouly undersampling and oversampling 
    approaches), namely: SMOTE+Tomek Links and SMOTE+ENN. In the end will be 
    compared the prediction accuracy of the model considering each of these 
    four approaches.
    '''
    
    '''
    USING SMOTE
    '''
    
    '''
    Step 1: 
        Resampling
    '''

X_trainSM, y_trainSM = SMOTE(random_state = 0).fit_resample(X_train, y_train)
print ('Before oversampling:', y_train.value_counts())
print ('After oversampling with SMOTE:', y_trainSM.value_counts())

    '''
    Step 2:
        Next is used Recursive Feature Elimination, RFE, to capture the most
        influential attributes so the ML can make more accurate prediction.
    '''
rfeSM = RFE(LogReg(), n_features_to_select = None)
rfeSM_fit = rfeSM.fit(X_trainSM, y_trainSM) 
print(rfeSM.support_)
print(rfeSM.ranking_)

    '''
    Step 3:
        Passing the most influential variables to a new set, and printing the
        final attributes...
    '''
X_trainSM_RFE = X_trainSM[X_trainSM.columns[rfeSM.support_.tolist()]]
X_testSM_RFE = X_test[X_trainSM.columns[rfeSM.support_.tolist()]]

print ('Number of final attributes:', len(X_trainSM.columns[rfeSM.support_]))
print ('Final attributes:', X_trainSM.columns[rfeSM.support_])

    '''
    Step 4:
        Implementing the model...
    '''
lgregSM = LogReg()
lgregSM.fit(X_trainSM_RFE, y_trainSM)

    '''
    USING ADASYN
    '''
    
    '''
    SMOTE is based on a KNN approach to generate data for the minority class,
    and tends to create a huge number of noisy data points. As an alternative,
    below is presented the ADASYN technique. This technique is a generalized
    form of the former and, as the other one, aims too at oversampling the minority
    class, but by taking into consideration the density distribution which 
    decides the amount of synthetic data points for samples that are difficult
    to learn.
    '''
    
    '''
    Step 1 for ADASYN
    '''

X_trainAD, y_trainAD = AD(random_state = 0).fit_resample(X_train, y_train)
print ('Before oversampling:', y_train.value_counts())
print ('After oversampling with ADASYN:', y_trainAD.value_counts())

    '''
    Step 2 for ADASYN
    '''
rfeAD = RFE(LogReg(), n_features_to_select = None)
rfeAD_fit = rfeAD.fit(X_trainAD, y_trainAD) 
print(rfeAD.support_)
print(rfeAD.ranking_)

    '''
    Step 3 for ADASYN
    '''
X_trainAD_RFE = X_trainAD[X_trainAD.columns[rfeAD.support_.tolist()]]
X_testAD_RFE = X_test[X_trainAD.columns[rfeAD.support_.tolist()]]

print ('Number of final attributes:', len(X_trainAD.columns[rfeAD.support_]))
print ('Final attributes:', X_trainAD.columns[rfeAD.support_])

    '''
    Step 4 for ADASYN
    '''
lgregAD = LogReg()
lgregAD.fit(X_trainAD_RFE, y_trainAD)

    '''
    USING Hybridization (SMOTE+Tomek Links, STL)
    '''
    
    '''
    STL is an hybrid technique that aims at cleaning overlapping data points
    for each of the classes distributed in sample space so to avoid any 
    possibility of overfitting.
    '''
    '''
    Step 1 for STL
    '''
X_trainST, y_trainST = ST(random_state = 0).fit_resample(X_train, y_train)
print ('Before oversampling:', y_train.value_counts())
print ('After oversampling with STL:', y_trainST.value_counts())

    '''
    Step 2 for STL
    '''
rfeST = RFE(LogReg(), n_features_to_select = None)
rfeST_fit = rfeST.fit(X_trainST, y_trainST) 
print(rfeST.support_)
print(rfeST.ranking_)

    '''
    Step 3 for STL
    '''
X_trainST_RFE = X_trainST[X_trainST.columns[rfeST.support_.tolist()]]
X_testST_RFE = X_test[X_trainST.columns[rfeST.support_.tolist()]]

print ('Number of final attributes:', len(X_trainST.columns[rfeST.support_]))
print ('Final attributes:', X_trainST.columns[rfeST.support_])

    '''
    Step 4 for STL
    '''
lgregST = LogReg()
lgregST.fit(X_trainAD_RFE, y_trainAD)

    '''
    USING Hybridization (SMOTE+ENN, STE)
    '''
    '''
    STE is another hybrid technique by which relatively more observations
    are deleted.
    '''
    '''
    Step 1 for STE
    '''
X_trainSE, y_trainSE = SE(random_state = 0).fit_resample(X_train, y_train)
print ('Before oversampling:', y_train.value_counts())
print ('After oversampling with STE:', y_trainSE.value_counts())

    '''
    Step 2 for STE
    '''
rfeSE = RFE(LogReg(), n_features_to_select = None)
rfeSE_fit = rfeSE.fit(X_trainSE, y_trainSE) 
print(rfeSE.support_)
print(rfeSE.ranking_)

    '''
    Step 3 for STE
    '''
X_trainSE_RFE = X_trainSE[X_trainSE.columns[rfeSE.support_.tolist()]]
X_testSE_RFE = X_test[X_trainSE.columns[rfeSE.support_.tolist()]]

print ('Number of final attributes:', len(X_trainSE.columns[rfeSE.support_]))
print ('Final attributes:', X_trainSE.columns[rfeSE.support_])

    '''
    Step 4 for STE
    '''
lgregSE = LogReg()
lgregSE.fit(X_trainSE_RFE, y_trainSE)

    '''
    ACCURACY COMPARISON BETWEEN THE RESAMPLING TECHNIQUES
    '''

print('Accuracy of the model (SMOTE):', lgregSM.fit(X_trainSM_RFE, y_trainSM).
                           score(X_testSM_RFE, y_test))
print('Accuracy of the model (ADASYN):', lgregAD.fit(X_trainAD_RFE, y_trainAD).
                           score(X_testAD_RFE, y_test))
print('Accuracy of the model (STL):', lgregST.fit(X_trainAD_RFE, y_trainAD).
                           score(X_testST_RFE, y_test))
print('Accuracy of the model (STE):', lgregSE.fit(X_trainSE_RFE, y_trainSE).
                           score(X_testSE_RFE, y_test))

    '''
    MODEL IMPLEMENTATION AND STATISTICAL SUMMARY
    '''
X_final = X_trainSM_RFE
Y_final = y_trainSM

model = smd.Logit(Y_final, X_final).fit()

print ('Statistical summary:', model.summary2())

    '''
    Two variables have p-values greater than 0.05 ('job_student' and
    'month_oct'). We'll drop them, and then fit the model with the update input
    values so we can check its accuracy. In the following step will be presented
    the confusion Matrix.
    '''

x_final = X_final.drop(['job_student', 'month_oct'], axis = 1)

model_final = smd.Logit(Y_final, x_final).fit()

print ('Statistical summary:', model_final.summary2())

LG_final = lgregSM.fit(x_final, Y_final)

x_test_final = X_testSM_RFE.drop(['job_student', 'month_oct'], axis = 1)

y_pred = LG_final.predict(x_test_final)

print('Accuracy of the final model:', LG_final.score(x_test_final, y_test))

print ('Confusion Matrix:', cm(y_test, y_pred))

    '''
    From the Confusion Matrix it is noticed that the model has 7174+374
    correct predictions. The former being True Positives and the latter True
    Negatives. Therefore, there is 853+642 (False Positives or Type Error 1
    and False Negatives or Type Error 2, respectively)
    '''

  '''
  PRECISION, RECALL, F1-SCORE, SUPPORT AND AUC-ROC CURVE
  '''
print ('Classification Report:', cr(y_test, y_pred))

    '''
    As we have an imbalanced data, it will be considered 'weighted avg' metric
    to evaluate the model accuracy. From this, it is noticed that the model has
    predicted correctly 85% of the test set as a conversion and, from all
    converted clients, in the same set, 83% of them were labelled accordingly.
    '''

auc_roc = ras(y_test, y_pred)
y_pred_prb = LG_final.predict_proba(x_test_final)[:,1]
fpr, tpr, tresholds = rcv(y_test, y_pred_prb)

plt.figure()
plt.plot(fpr, tpr, 'b-.', label = 'AUC = '+str(round(auc_roc,2)))
plt.plot([0,1], [0,1], 'k--')
plt.title('Receiver Operating Characterictic')
plt.ylabel('True Positive Rate')
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.xlim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction2/AUC-ROC_CURVE.png')
plt.close()

    '''
    From the AUC-ROC CURVE it is noticed that there is a chance of 63% that the 
    model will be able to distinguish between converted and non-converted 
    clients. Despite of not being considerably far from 50% (no discrimination
    ability) and the cost of a  wrong prediction not posing a life-altering
    case, this performance could still be useful.
    '''

    '''
    Saving the model...
    '''
jb.dump(LG_final,"C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction2/LogModelBank.sav")
_______________________________________________________________________end________________________________________________________________________