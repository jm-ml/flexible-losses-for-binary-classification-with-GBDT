# Importing the libraries as in R
import FXGBoost as fxgb
import numpy as np  
import pandas as pd 

from hmeasure import h_score
from sklearn.metrics import roc_auc_score 
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier 
from sklearn.ensemble import AdaBoostClassifier
from imblearn.combine import SMOTEENN
import Functions as fn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


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


col=["variable1", "variable2"]  #insert list of covariates


########Links

def logit(x):
    return 1.0/(1.0+np.exp(-x)) 


# Importing the dataset
training = pd.read_csv('modelDevData.csv')
oos = pd.read_csv('oosTestData.csv')
oot = pd.read_csv('ootTestData.csv')

scaler = MinMaxScaler(feature_range=(0, 1))

with open("SavedREsults.txt", "a") as f:
    for x in range(5):   

        print("Run number : {0}".format(x+1),file=f) 
        
        training_sample = training.sample(n=100000)
        
        train, validate = \
                      np.split(training_sample.sample(frac=1), [int(.5*len(training_sample))])
                      
        oos_sample = oos.sample(n=50000)   
        oot_sample  = oot.sample(n=50000)    
    
        train_f=pd.DataFrame(train,columns=col)
        validate_f=pd.DataFrame(validate,columns=col)
        test_oos=pd.DataFrame(oos_sample,columns=col)
        test_oot=pd.DataFrame(oot_sample,columns=col)
        
        
        #standardize data
        train_f = pd.DataFrame(scaler.fit_transform(train_f))
        validate_f = pd.DataFrame(scaler.fit_transform(validate_f))
        test_oos = pd.DataFrame(scaler.fit_transform(test_oos))
        test_oot = pd.DataFrame(scaler.fit_transform(test_oot))
        
        invars=len(train_f.columns)-1
        
        x_train = train_f.iloc[:, 1:len(col)].values  #set of predictors by selecting all rows;excluding last column
        y_train = np.float32(np.array(train_f.iloc[:, 0].values, float))   # first row of data frame
        cases=round(len(train_f)*1) #cases to use for modelling
        
        x_validate = validate_f.iloc[:, 1:len(col)].values  #set of predictors by selecting all rows;excluding last column
        y_validate = np.float32(np.array(validate_f.iloc[:, 0].values, float))   # first row of data frame
        
        x_test_oos = test_oos.iloc[:, 1:len(col)].values  #set of predictors by selecting all rows;excluding last column
        x_test_oot = test_oot.iloc[:, 1:len(col)].values  #set of predictors by selecting all rows;excluding last column
        
        y_test_oos = np.float32(np.array(test_oos.iloc[:, 0].values, float))   # first row of data frame
        y_test_oot = np.float32(np.array(test_oot.iloc[:, 0].values, float))   # first row of data frame
        
        print ("Train DR: ", sum(y_train)/len(train),file=f)
        print ("Validate DR: ", sum(y_validate)/len(validate),file=f)
        print ("Test oos DR: ", sum(y_test_oos)/len(oos),file=f)
        print ("Test oot DR: ", sum(y_test_oot)/len(oot),file=f)
        
        pen=round((len(train)-sum(y_train))/sum(y_train),0)
        print ("IR: ", pen,file=f)
        
        dmatrix_train = xgb.DMatrix(data=x_train, label=y_train)
        dmatrix_val = xgb.DMatrix(data=x_validate, label=y_validate)
        dmatrix_oos = xgb.DMatrix(data=x_test_oos, label=y_test_oos)
        dmatrix_oot = xgb.DMatrix(data=x_test_oot, label=y_test_oot)
        
        watchlist = [(dmatrix_train, 'train'),(dmatrix_val, 'eval') ]
        num_round = 1000
        
        params={  
             'max_depth': 2,
              'min_child_weight': 6,    
              'subsample': 0.85,
              'learning_rate': 0.01,
              'colsample_bytree': 0.9,
              'gamma':0,
              'scale_pos_weight':0.5*pen,
              'eval_metric':'auc',
              'tree_method':'approx'
               }
        
        print("======bstacking diverse classfiers (meta - classifier)=======")
    
        def bstacking(T=1, verbose=False, num_round=1000, params=params):
                   
            meta_pred_oos = []
            meta_pred_oot = []
    
            for t in range(T):
                print(f"Iteration: {t+1}")
                OOB_l2_data = []
                OOS_l2_data = []
                OOT_l2_data = []
                np.random.seed(t+1)
                positions = np.random.choice(train_f.shape[0], size=train_f.shape[0], replace=True)
                training = train_f.iloc[positions, :]
                #train, validate = np.split(training.sample(frac=1), [int(split*len(training))])
                OOB = train_f.iloc[~np.unique(positions), :]
                
               # Set up the Train, OOB, validate test and data
                x_train_bs = training.iloc[:, 1:len(col)].values
                y_train_bs = np.float32(np.array(training.iloc[:, 0].values, float)) 
                dmatrix_train_bs = xgb.DMatrix(data=x_train_bs, label=y_train_bs)
          
                x_oob = OOB.iloc[:, 1:len(col)].values
                y_oob = np.float32(np.array(OOB.iloc[:, 0].values, float)) 
                dmatrix_OOB = xgb.DMatrix(data=x_oob, label=y_oob)   
    
                # Watchlist for XGBvalidate early stopping
                watchlist = [(dmatrix_train_bs, 'train'), (dmatrix_val, 'eval')]
               # Train predefined custom XGBoost models
                print("======Base Learner 2.1=======")
                smenn = SMOTEENN()
                x_train_smenn, y_train_smenn = smenn.fit_resample(x_train_bs, y_train_bs)
                rf = RandomForestClassifier(n_estimators=100, max_features=11)
                rf.fit(x_train_smenn, y_train_smenn)
               # Save predictions for OOB and test sample
                OOB_l2_data.append(rf.predict_proba(x_oob)[:, 1]) 
                OOS_l2_data.append(rf.predict_proba(x_test_oos)[:, 1]) 
                OOT_l2_data.append(rf.predict_proba(x_test_oot)[:, 1])
                print("======Base Learner 2.2=======")
                n_estimators = 91
                n_samples = 31
                eec = EasyEnsembleClassifier(
                    n_estimators=n_samples, base_estimator=AdaBoostClassifier(n_estimators=n_estimators))
                eec.fit(x_train_bs, y_train_bs)
               # Save predictions for OOB and test sample
                OOB_l2_data.append(eec.predict_proba(x_oob)[:, 1])  
                OOS_l2_data.append(eec.predict_proba(x_test_oos)[:, 1])  
                OOT_l2_data.append(eec.predict_proba(x_test_oot)[:, 1])  
                print("======Base Learner 2.3=======")
                opt = tf.keras.optimizers.Adam(learning_rate=0.03, clipnorm=1.0)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.0001, patience=10, restore_best_weights=True)
                model = Sequential()
                model.add(Dense(54, input_dim=invars, activation='gelu'))
                model.add(Dropout(rate=0.4))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss=fn.focal_loss(alpha=14),optimizer=opt, metrics=['accuracy'])
                model.fit(x_train, y_train, validation_data=(x_validate, y_validate), callbacks=[es], epochs=1000, batch_size=5718, verbose=0)
                # Save predictions for OOB and test sample
                OOB_l2_data.append(np.nan_to_num(model.predict(x_oob))) 
                OOS_l2_data.append(np.nan_to_num(model.predict(x_test_oos))) 
                OOT_l2_data.append(np.nan_to_num(model.predict(x_test_oot))) 
                print("======Base Learner 2.4=======")
                tent = xgb.train(params, dmatrix_train_bs, num_round, watchlist, early_stopping_rounds=10,
                                 obj=fxgb.symmetric_unified_focal_gl (alpha=0.8*pen,delta=0.8,gamma1=0.5, gamma2=4/3,beta=4.0,lamda=3.0,pi=0.5), 
                                 verbose_eval=verbose
                                 )
                best_iter = tent.best_iteration
                bst = xgb.train(params, dmatrix_train_bs, best_iter,
                                obj=fxgb.symmetric_unified_focal_gl (alpha=0.8*pen,delta=0.8,gamma1=0.5, gamma2=4/3,beta=4.0,lamda=3.0,pi=0.5),
                                verbose_eval=verbose
                                )
               # Save predictions for OOB and test sample 
                OOB_l2_data.append(eel(bst.predict(dmatrix_OOB,output_margin=True)+0.5,beta=4.0,lamda=3.0))
                OOS_l2_data.append(eel(bst.predict(dmatrix_oos,output_margin=True)+0.5,beta=4.0,lamda=3.0)) 
                OOT_l2_data.append(eel(bst.predict(dmatrix_oot,output_margin=True)+0.5,beta=4.0,lamda=3.0))  
              # Train Meta Model on OOB sample
                OOB_l2_data.append(y_oob)  
               
                #Create level-data frames
                OOB_l2_df = pd.DataFrame(OOB_l2_data).transpose()   
                OOS_l2_df = pd.DataFrame(OOS_l2_data).transpose()
                OOT_l2_df = pd.DataFrame(OOT_l2_data).transpose()
                y_meta=np.float32(np.array(OOB_l2_df.iloc[:, 4].values, float))
                x_meta=OOB_l2_df.iloc[:, 0:4].values
                
                
                print("======Meta Learner=======")
                meta = BalancedRandomForestClassifier()
                meta.fit(x_meta, y_meta)   
               # Score test data
                meta_pred_oos.append(meta.predict_proba(OOS_l2_df)[:, 1])
                meta_pred_oot.append(meta.predict_proba(OOT_l2_df)[:, 1])
               # calculate average predictions
            meta_pred_oos_avg = np.mean(np.array(meta_pred_oos), axis=0)
            meta_pred_oot_avg = np.mean(np.array(meta_pred_oot), axis=0)
            return meta_pred_oos_avg, meta_pred_oot_avg
        # Usage
        start_time = pd.Timestamp.now()
        bs_pred = bstacking(40)
        end_time = pd.Timestamp.now()
        execution_time = end_time - start_time        
        
        #estimate AUC and H measure
        #h_val = h_score(y_validate, y_pred_val)
        h_test_oos = h_score(oos_sample['Target'].values, bs_pred[0])
        h_test_oot = h_score(oot_sample['Target'].values, bs_pred[1])
        
        auc_test_oos = roc_auc_score(oos_sample['Target'].values, bs_pred[0])
        auc_test_oot = roc_auc_score(oot_sample['Target'].values, bs_pred[1])
        
        DSC_test_oos=DSC(oos_sample['Target'].values, bs_pred[0])    
        DSC_test_oot=DSC(oot_sample['Target'].values, bs_pred[1])
        
        MCC_test_oos=MCC(oos_sample['Target'].values, bs_pred[0])    
        MCC_test_oot=MCC(oot_sample['Target'].values, bs_pred[1])        

        print(' Classifier  >>>  Bstacking ',file=f)
    
        print("AUC_oos : {:.4%}".format(auc_test_oos),file=f)
        print("AUC_oot : {:.4%}".format(auc_test_oot),file=f)
        
        print("DSC_oos : {:.4%}".format(DSC_test_oos),file=f)
        print("DSC_oot : {:.4%}".format(DSC_test_oot),file=f)
        
        print("MCC_oos : {:.4%}".format(MCC_test_oos),file=f)
        print("MCC_oot : {:.4%}".format(MCC_test_oot),file=f)        
        
        #print("H val M7: {:.4%}".format(h_val),file=f)
        print("H_oos : {:.4%}".format(h_test_oos),file=f)
        print("H_oot : {:.4%}".format(h_test_oot),file=f)
        
        print ("Runtime : ",(execution_time).total_seconds(),file=f)




