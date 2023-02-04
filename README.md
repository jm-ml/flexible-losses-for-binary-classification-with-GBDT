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



# Importing the training sample
training = pd.read_csv('FM12.csv')   #Importing the primary modeling dataset, i.e. FM12.csv in this case
# Spling the training sample into equal halves randomly
train, validate = \
              np.split(training.sample(frac=1), [int(.5*len(training))])  
              
x_train = train.iloc[:, 2:80].values  #set of predictors by selecting all rows;excluding last column
y_train = np.float32(np.array(train.iloc[:, 0].values, float))   # first row of data frame
cases=round(len(train)*1) #cases to use for modelling

x_validate = validate.iloc[:, 2:80].values  #set of predictors by selecting all rows;excluding last column
y_validate = np.float32(np.array(validate.iloc[:, 0].values, float))   # first row of data frame
                  
dmatrix_train = xgb.DMatrix(data=x_train, label=y_train)
dmatrix_val = xgb.DMatrix(data=x_validate, label=y_validate)


# implementing the symmetric hybrid loss function with an EEL distribution based link function
num_round=10
model = xgb.train(params,
                  dmatrix_train,
                  num_round,
                  obj=fxgb.symmetric_unified_focal_eel(alpha=1,delta=0.6,gamma1=1.25, gamma2=2,lamda=0.25, beta=0.75,pi=0.2)     
                 )
#making predictions using a standard EEL distribution-based link function                 
y_pred = eel(model.predict(dmatrix_val,output_margin=True)+0.5,lamda=0.25, beta=0.75)

#estimate H measure, Dice score and MMC for validation sample
h_val = h_score(y_validate, y_pred_val)
h_test_oos = h_score(y_test_oos, y_pred_test_oos)
h_test_oot = h_score(y_test_oot, y_pred_test_oot)

auc_test_oos = roc_auc_score(y_test_oos, y_pred_test_oos)
auc_test_oot = roc_auc_score(y_test_oot, y_pred_test_oot)

DSC_test_oos=fxgb.DSC(y_test_oos, y_pred_test_oos)    
DSC_test_oot=fxgb.DSC(y_test_oot, y_pred_test_oot)

MCC_test_oos=fxgb.MCC(y_test_oos, y_pred_test_oos)    
MCC_test_oot=fxgb.MCC(y_test_oot, y_pred_test_oot)
#Print statistics
print("AUC oos E12: {:.4%}".format(auc_test_oos))
print("AUC oot E12: {:.4%}".format(auc_test_oot))

print("DSC oos E12: {:.4%}".format(DSC_test_oos))
print("DSC oot E12: {:.4%}".format(DSC_test_oot))

print("MCC oos E12: {:.4%}".format(MCC_test_oos))
print("MCC oot E12: {:.4%}".format(MCC_test_oot))


print("H val E12: {:.4%}".format(h_val))
print("H measure oos E12: {:.4%}".format(h_test_oos))
print("H measure oot E12: {:.4%}".format(h_test_oot))

print("nrounds E12: {:}".format(best_iter))
print ("Runtime E12: ",end_time-start_time)
