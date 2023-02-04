# flexible-losses-for-binary-classification-with-GBDT
Gradient statistics for implementing a generalized hybrid loss function in gradient-boosted decision trees (GBDT) for binary classification tasks. 

import FXGBoost as fxgb #Import flexible Gradient statistics module.
import numpy as np  #for mathematical operations
from hmeasure import h_score #hmeasure

###########################################
#   standard GEV distibution based link   #
###########################################
def gevlink(x,tau=-0.25):
    base=1 + (tau*x)
    power=-1/tau        
    return np.exp(-np.power(base,power)) 

###########################################
#   standard EEL distibution based link   #
###########################################
def eel(x,beta=0.75,lamda=0.75):
    p=np.power(1-np.power((1+np.exp(x)),-lamda),beta)
    return p

###########################################
#      Dice Score                         #
###########################################
def DSC(y_true, y_pred):
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1-y_pred))
    fp = np.sum((1-y_true) * y_pred)
    # Calculate Dice score
    dice_class = (2*tp)/((2*tp) + fp + fn)
    return dice_class  

###########################################
#      MCC                                #
###########################################
def MCC(y_true, y_pred):
    # Calculate true positives (tp),true megstivess(tn), false negatives (fn) and false positives (fp)
    tp = np.sum(y_true * y_pred)
    tn=  np.sum((1-y_true) * (1-y_pred))
    fn = np.sum(y_true * (1-y_pred))
    fp = np.sum((1-y_true) * y_pred)
    # Calculate Dice score
    num = (tp*tn)-(fp*fn)
    den =(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return np.divide(num,np.power(den,0.5))

###########################################################################################################
# Example : Implementing the symmetric hybrid loss function with an EEL distribution based link function  #
###########################################################################################################

#datasets
dmatrix_train = xgb.DMatrix(data=x_train, label=y_train) #training sample in DMatrix form
dmatrix_val = xgb.DMatrix(data=x_validate, label=y_validate) #validation sample in DMatrix form

#model fitting
model = xgb.train(
                  params, #Hyperparameter set
                  dmatrix_train, 
                  num_round, #number of iterations
                  obj=fxgb.symmetric_unified_focal_eel(alpha=1,delta=0.6,gamma1=1.25, gamma2=2,lamda=0.25, beta=0.75,pi=0.2) #custom hybrid loss function    
                 )
#making predictions on a validation sample using a standard EEL distribution-based link function  
#note the raw XGBoost predictions are passed to the eel link
y_pred = eel(model.predict(dmatrix_val,output_margin=True)+0.5,lamda=0.25, beta=0.75)

#estimate H measure, Dice score and MCC for validation sample with target outcome, y_true as a 0/1 target outcome. 
H_measure = h_score(y_validate, y_pred)
Dice=DSC(y_validate, y_pred)    
MCC=MCC(y_validate, y_pred)    

#Print performance
print("H measure: {:.4%}".format(H_measure))
print("Dice score: {:.4%}".format(Dice))
print("MCC: {:.4%}".format(MCC))

