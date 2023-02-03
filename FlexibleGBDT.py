import xgboost as xgb
import numpy as np 

################################################################
#                 Cost sensing XGBoost (CSXGBoost)             #
################################################################
def gradient_csxgb(preds: np.ndarray, dtrain: xgb.DMatrix,c0=1.0,c1=1.0):           
    labels = dtrain.get_label()
    preds=1/(1+np.exp(-2*preds))
    preds2=1-preds    
    fy=(1/(c1+c0))*np.log((preds*c1)/(preds2*c0)) 
    eta=0.5*np.log(c0/c1)
    delta=(c1+c0)/2   
    py=1/(1+np.exp((-2*delta*fy)-(2*eta)))
    grad=-2*delta*(labels-py) 
    return grad

def hessian_csxgb(preds: np.ndarray, dtrain: xgb.DMatrix,c0=1,c1=1):      
    preds=1/(1+np.exp(-2*preds))
    preds2=1-preds    
    fy=(1/(c1+c0))*np.log((preds*c1)/(preds2*c0)) 
    eta=0.5*np.log(c0/c1)
    delta=(c1+c0)/2   
    py=1/(1+np.exp((-2*delta*fy)-(2*eta)))
    hess= 4*np.power(delta,2)*py*(1-py)
    return hess
''' CSXGBoost.'''
def cost_sensing_xgb (c0=1.0,c1=1.0):    
    def csxgb(preds: np.ndarray, dtrain: xgb.DMatrix):
        """Xia, Y., Liu, C., & Liu, N. (2017). Cost-sensitive boosted tree for loan evaluation in peer-to-peer lending.
        Electronic Commerce Research and Applications, 24, 30-49. 
        Parameters
        ----------
        c0 : float, default value is 1.0
            denote the cost of incorrectly classifying a negative case (y=0) as a positive case (y=1).
        c1 : float, default value is 1.0
            denote the cost of incorrectly classifying a positive case (y=1) as a negative case (y=0)."""
        grad = gradient_csxgb(preds, dtrain,c0,c1)
        hess = hessian_csxgb(preds, dtrain,c0,c1)
        return grad, hess
    return csxgb


   
################################################################
#     Symmetric  Modified Focal loss    with GEV link          #
################################################################
def gradient_mf_gev(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0,tau=-0.25):           
    labels = dtrain.get_label()
    preds=np.exp(-np.power(1 + (tau*preds),-1/tau))
    n1=-(gamma1*preds*((1-preds)**(gamma1-1))*np.log(preds))+((1-preds)**gamma1)
    n2s=(gamma1*(1-preds)*(preds**(gamma1-1))*np.log(1-preds))-(preds**gamma1)          
    L1_mF=-((alpha*labels*n1*(1-preds))+((1-labels)*preds*n2s))/(preds*(1-preds))
    P1_gev=preds*(np.log(1/preds))**(tau+1)
    grad=L1_mF*P1_gev  
    return grad

def hessian_mf_gev(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0,tau=-0.25):      
    labels = dtrain.get_label()
    preds=np.exp(-np.power(1 + (tau*preds),-1/tau))
    n1=-(gamma1*preds*((1-preds)**(gamma1-1))*np.log(preds))+((1-preds)**gamma1)
    n2s=(gamma1*(1-preds)*(preds**(gamma1-1))*np.log(1-preds))-(preds**gamma1)
    n3=-(2*gamma1*preds*((1-preds)**(gamma1-1)))-((1-preds)**gamma1)+(((gamma1*preds)**2)*((1-preds)**(gamma1-2))*np.log(preds))-(gamma1*preds*preds*((1-preds)**(gamma1-2))*np.log(preds))
    n4s=((gamma1*(1-preds)*((1-preds)*(gamma1-1)*(preds**(gamma1-1))-(preds**(gamma1-1))))-(gamma1*(preds**(gamma1-1))*(1-preds))-(preds**gamma1))          
    L1_mF=-((alpha*labels*n1*(1-preds))+((1-labels)*preds*n2s))/(preds*(1-preds))
    L2_mF=-((alpha*labels*n3*(1-preds)**2)+((1-labels)*(preds**2)*n4s))/(preds*(1-preds))**2
    P1_gev=preds*(np.log(1/preds))**(tau+1)
    P2_gev=-preds*((tau+1)*((np.log(1/preds))**-1)-1)*(np.log(1/preds))**(2*(tau+1))
    hess=(L1_mF*P2_gev)+((P1_gev**2)*L2_mF)
    return hess
'''symmetric  Modified Focal loss    with GEV link .'''
def modified_focal_loss_gev (alpha=1.0, gamma1=0.0,tau=-0.25):    
    def mF_gev(preds: np.ndarray, dtrain: xgb.DMatrix):
        """Mushava, J., & Murray, M. (2022). A novel XGBoost extension for credit scoring class-imbalanced data combining a generalized extreme value link and a modified focal loss function.
        Expert Systems with Applications, 202, 117233. 
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        gamma1 : float, default value is 0.0
            The focal parameter controls the degree of down-weighting of easy to classify cases in both classes.
        tau : float, default value is -0.25
            controls the skewness of the standard GEV-distribution based link"""
        grad = gradient_mf_gev(preds, dtrain,alpha, gamma1,tau)
        hess = hessian_mf_gev(preds, dtrain,alpha, gamma1,tau)
        return grad, hess
    return mF_gev

####################################################################################
#     Asymmetric Positive class (y=1) Modified Focal loss    with GEV link         #
####################################################################################
def gradient_maf_gev1(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0,tau=-0.25):           
    labels = dtrain.get_label()
    preds=np.exp(-np.power(1 + (tau*preds),-1/tau))
    n1s=-(gamma1*preds*((1-preds)**(gamma1-1))*np.log(preds))+((1-preds)**gamma1)
    n2a=((1-preds)*np.log(1-preds))-preds       
    L1_maF=-((alpha*labels*n1s*(1-preds))+((1-labels)*preds*n2a))/(preds*(1-preds))
    P1_gev=preds*(np.log(1/preds))**(tau+1)
    grad=L1_maF*P1_gev  
    return grad

def hessian_maf_gev1(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0,tau=-0.25):      
    labels = dtrain.get_label()
    preds=np.exp(-np.power(1 + (tau*preds),-1/tau))
    n1s=-(gamma1*preds*((1-preds)**(gamma1-1))*np.log(preds))+((1-preds)**gamma1)
    n2a=((1-preds)*np.log(1-preds))-preds
    n3s=-(2*gamma1*preds*((1-preds)**(gamma1-1)))-((1-preds)**gamma1)+(((gamma1*preds)**2)*((1-preds)**(gamma1-2))*np.log(preds))-(gamma1*preds*preds*((1-preds)**(gamma1-2))*np.log(preds))
    n4a=-(1-preds)      
    L1_maF=-((alpha*labels*n1s*(1-preds))+((1-labels)*preds*n2a))/(preds*(1-preds))
    L2_maF=-((alpha*labels*n3s*(1-preds)**2)+((1-labels)*(preds**2)*n4a))/(preds*(1-preds))**2
    P1_gev=preds*(np.log(1/preds))**(tau+1)
    P2_gev=-preds*((tau+1)*((np.log(1/preds))**-1)-1)*(np.log(1/preds))**(2*(tau+1))
    hess=(L1_maF*P2_gev)+((P1_gev**2)*L2_maF)
    return hess
'''asymmetric modified Focal loss function for positive class  with GEV link.'''
def modified_asy_focal_loss_gev1 (alpha=1.0, gamma1=0.0,tau=-0.25):    
    def mF_gev(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y=1).
        gamma1 : float, default value is 0.0
            The focal parameter controls the degree of down-weighting of easy to classify  positive cases (y=1).
        tau : float, default value is -0.25
            controls the skewness of the standard GEV-distribution based link"""
        grad = gradient_maf_gev1(preds, dtrain,alpha, gamma1,tau)
        hess = hessian_maf_gev1(preds, dtrain,alpha, gamma1,tau)
        return grad, hess
    return mF_gev


####################################################################################
#     Asymmetric Negative class (y=0) Modified Focal loss    with GEV link         #
####################################################################################
def gradient_maf_gev0(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0,tau=-0.25):           
    labels = dtrain.get_label()
    preds=np.exp(-np.power(1 + (tau*preds),-1/tau))
    n1a=-preds*np.log(preds)+(1-preds)
    n2s=(gamma1*(1-preds)*(preds**(gamma1-1))*np.log(1-preds))-(preds**gamma1)       
    L1_maF=-((alpha*labels*n1a*(1-preds))+((1-labels)*preds*n2s))/(preds*(1-preds))
    P1_gev=preds*(np.log(1/preds))**(tau+1)
    grad=L1_maF*P1_gev  
    return grad

def hessian_maf_gev0(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0,tau=-0.25):      
    labels = dtrain.get_label()
    preds=np.exp(-np.power(1 + (tau*preds),-1/tau))
    n1a=-preds*np.log(preds)+(1-preds)
    n2s=(gamma1*(1-preds)*(preds**(gamma1-1))*np.log(1-preds))-(preds**gamma1)  
    n3a=-(preds+1)
    n4s=((gamma1*(1-preds)*((1-preds)*(gamma1-1)*(preds**(gamma1-1))-(preds**(gamma1-1))))-(gamma1*(preds**(gamma1-1))*(1-preds))-(preds**gamma1))     
    L1_maF=-((alpha*labels*n1a*(1-preds))+((1-labels)*preds*n2s))/(preds*(1-preds))
    L2_maF=-((alpha*labels*n3a*(1-preds)**2)+((1-labels)*(preds**2)*n4s))/(preds*(1-preds))**2
    P1_gev=preds*(np.log(1/preds))**(tau+1)
    P2_gev=-preds*((tau+1)*((np.log(1/preds))**-1)-1)*(np.log(1/preds))**(2*(tau+1))
    hess=(L1_maF*P2_gev)+((P1_gev**2)*L2_maF)
    return hess
'''asymmetric modified Focal loss function for negative class with GEV link.'''
def modified_asy_focal_loss_gev0 (alpha=1.0, gamma1=0.0,tau=-0.25):    
    def mF_gev(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        gamma1 : float, default value is 0.0
            The focal parameter controls the degree of down-weighting of easy to classify  negative cases (y=0).
        tau : float, default value is -0.25
            controls the skewness of the standard GEV-distribution based link"""
        grad = gradient_maf_gev1(preds, dtrain,alpha, gamma1,tau)
        hess = hessian_maf_gev1(preds, dtrain,alpha, gamma1,tau)
        return grad, hess
    return mF_gev



################################################################
#     Symmetric  Modified Focal loss   with EEL link          #
################################################################
def gradient_mf_eel(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):           
    labels = dtrain.get_label()
    preds=np.power(1-np.power((1+np.exp(preds)),-lamda),beta)
    psi=1-np.power(preds,1/beta)    
    n1=-(gamma1*preds*((1-preds)**(gamma1-1))*np.log(preds))+((1-preds)**gamma1)
    n2s=(gamma1*(1-preds)*(preds**(gamma1-1))*np.log(1-preds))-(preds**gamma1)          
    L1_mF=-((alpha*labels*n1*(1-preds))+((1-labels)*preds*n2s))/(preds*(1-preds))
    P1_eel=beta*lamda*psi*(1-np.power(psi,1/lamda))*np.power(1-psi,beta-1) 
    grad=L1_mF*P1_eel  
    return grad

def hessian_mf_eel(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):      
    labels = dtrain.get_label()
    preds=np.power(1-np.power((1+np.exp(preds)),-lamda),beta)
    psi=1-np.power(preds,1/beta)
    n1=-(gamma1*preds*((1-preds)**(gamma1-1))*np.log(preds))+((1-preds)**gamma1)
    n2s=(gamma1*(1-preds)*(preds**(gamma1-1))*np.log(1-preds))-(preds**gamma1)
    n3=-(2*gamma1*preds*((1-preds)**(gamma1-1)))-((1-preds)**gamma1)+(((gamma1*preds)**2)*((1-preds)**(gamma1-2))*np.log(preds))-(gamma1*preds*preds*((1-preds)**(gamma1-2))*np.log(preds))
    n4s=((gamma1*(1-preds)*((1-preds)*(gamma1-1)*(preds**(gamma1-1))-(preds**(gamma1-1))))-(gamma1*(preds**(gamma1-1))*(1-preds))-(preds**gamma1))          
    L1_mF=-((alpha*labels*n1*(1-preds))+((1-labels)*preds*n2s))/(preds*(1-preds))
    L2_mF=-((alpha*labels*n3*(1-preds)**2)+((1-labels)*(preds**2)*n4s))/(preds*(1-preds))**2
    P1_eel=beta*lamda*psi*(1-np.power(psi,1/lamda))*np.power(1-psi,beta-1) 
    P2_eel=-beta*lamda*np.power(psi,(beta+(1/lamda)))*(1-np.power(psi,1/lamda))*np.power((1/psi)-1,beta-2)*((1/psi)*(lamda*np.power(psi,-1/lamda)-lamda-1)-beta*lamda*(np.power(psi,-1/lamda)-1)+1)
    hess=(L1_mF*P2_eel)+((P1_eel**2)*L2_mF)
    return hess
'''symmetric modified Focal loss function with EEL link.'''
def modified_focal_loss_eel (alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):    
    def mF_eel(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        gamma1 : float, default value is 0.0
            The focal parameter controls the degree of down-weighting of easy to classify cases in both classes.
        lamda,beta : float, default value is 1.0
            shape parameters for the standard EEL-distribution based link"""
        grad = gradient_mf_eel(preds, dtrain,alpha, gamma1,beta)
        hess = hessian_mf_eel(preds, dtrain,alpha, gamma1,beta)
        return grad, hess
    return mF_eel


####################################################################################
#     Asymmetric Positive class (y=1) Modified Focal loss    with EEL link          #
####################################################################################
def gradient_maf_eel1(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):           
    labels = dtrain.get_label()
    preds=np.power(1-np.power((1+np.exp(preds)),-lamda),beta)
    psi=1-np.power(preds,1/beta)    
    n1=-(gamma1*preds*((1-preds)**(gamma1-1))*np.log(preds))+((1-preds)**gamma1)
    n2a=((1-preds)*np.log(1-preds))-preds          
    L1_maF=-((alpha*labels*n1*(1-preds))+((1-labels)*preds*n2a))/(preds*(1-preds))
    P1_eel=beta*lamda*psi*(1-np.power(psi,1/lamda))*np.power(1-psi,beta-1) 
    grad=L1_maF*P1_eel  
    return grad

def hessian_maf_eel1(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):      
    labels = dtrain.get_label()
    preds=np.power(1-np.power((1+np.exp(preds)),-lamda),beta)
    psi=1-np.power(preds,1/beta)
    n1=-(gamma1*preds*((1-preds)**(gamma1-1))*np.log(preds))+((1-preds)**gamma1)
    n2a=((1-preds)*np.log(1-preds))-preds
    n3=-(2*gamma1*preds*((1-preds)**(gamma1-1)))-((1-preds)**gamma1)+(((gamma1*preds)**2)*((1-preds)**(gamma1-2))*np.log(preds))-(gamma1*preds*preds*((1-preds)**(gamma1-2))*np.log(preds))
    n4a=-preds-2*(1-preds)          
    L1_maF=-((alpha*labels*n1*(1-preds))+((1-labels)*preds*n2a))/(preds*(1-preds))
    L2_maF=-((alpha*labels*n3*(1-preds)**2)+((1-labels)*(preds**2)*n4a))/(preds*(1-preds))**2
    P1_eel=beta*lamda*psi*(1-np.power(psi,1/lamda))*np.power(1-psi,beta-1) 
    P2_eel=-beta*lamda*np.power(psi,(beta+(1/lamda)))*(1-np.power(psi,1/lamda))*np.power((1/psi)-1,beta-2)*((1/psi)*(lamda*np.power(psi,-1/lamda)-lamda-1)-beta*lamda*(np.power(psi,-1/lamda)-1)+1)
    hess=(L1_maF*P2_eel)+((P1_eel**2)*L2_maF)
    return hess
''' asymmetric modified Focal loss function for positive class with EEL link.'''
def modified_asy_focal_loss_eel1 (alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):    
    def mF_eel(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        gamma1 : float, default value is 0.0
            The focal parameter controls the degree of down-weighting of easy to classify  positive cases (y=1).
        lamda,beta : float, default value is 1.0
            shape parameters for the standard EEL-distribution based link"""
        grad = gradient_mf_eel(preds, dtrain,alpha,lamda, gamma1,beta)
        hess = hessian_mf_eel(preds, dtrain,alpha,lamda, gamma1,beta)
        return grad, hess
    return mF_eel


####################################################################################
#     Asymmetric Positive class (y=0) Modified Focal loss    with EEL link          #
####################################################################################
def gradient_maf_eel0(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):           
    labels = dtrain.get_label()
    preds=np.power(1-np.power((1+np.exp(preds)),-lamda),beta)
    psi=1-np.power(preds,1/beta)    
    n1a=-preds*np.log(preds)+(1-preds)
    n2s=(gamma1*(1-preds)*(preds**(gamma1-1))*np.log(1-preds))-(preds**gamma1)          
    L1_maF=-((alpha*labels*n1a*(1-preds))+((1-labels)*preds*n2s))/(preds*(1-preds))
    P1_eel=beta*lamda*psi*(1-np.power(psi,1/lamda))*np.power(1-psi,beta-1) 
    grad=L1_maF*P1_eel  
    return grad

def hessian_maf_eel0(preds: np.ndarray, dtrain: xgb.DMatrix,alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):      
    labels = dtrain.get_label()
    preds=np.power(1-np.power((1+np.exp(preds)),-lamda),beta)
    psi=1-np.power(preds,1/beta)
    n1a=-preds*np.log(preds)+(1-preds)
    n2s=(gamma1*(1-preds)*(preds**(gamma1-1))*np.log(1-preds))-(preds**gamma1)  
    n3a=-(preds+1)
    n4s=((gamma1*(1-preds)*((1-preds)*(gamma1-1)*(preds**(gamma1-1))-(preds**(gamma1-1))))-(gamma1*(preds**(gamma1-1))*(1-preds))-(preds**gamma1))           
    L1_maF=-((alpha*labels*n1a*(1-preds))+((1-labels)*preds*n2s))/(preds*(1-preds))
    L2_maF=-((alpha*labels*n3a*(1-preds)**2)+((1-labels)*(preds**2)*n4s))/(preds*(1-preds))**2
    P1_eel=beta*lamda*psi*(1-np.power(psi,1/lamda))*np.power(1-psi,beta-1) 
    P2_eel=-beta*lamda*np.power(psi,(beta+(1/lamda)))*(1-np.power(psi,1/lamda))*np.power((1/psi)-1,beta-2)*((1/psi)*(lamda*np.power(psi,-1/lamda)-lamda-1)-beta*lamda*(np.power(psi,-1/lamda)-1)+1)
    hess=(L1_maF*P2_eel)+((P1_eel**2)*L2_maF)
    return hess
''' asymmetric modified Focal loss function for negative class with EEL link.'''
def modified_asy_focal_loss_eel0 (alpha=1.0, gamma1=0.0, lamda=1.0,beta=1.0):    
    def mF_eel(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        gamma1 : float, default value is 0.0
            The focal parameter controls the degree of down-weighting of easy to classify  negative cases (y=0).
        lamda,beta : float, default value is 1.0
            shape parameters for the standard EEL-distribution based link"""
        grad = gradient_mf_eel(preds, dtrain,alpha, gamma1, lamda,beta)
        hess = hessian_mf_eel(preds, dtrain,alpha, gamma1, lamda,beta)
        return grad, hess
    return mF_eel



################################################################
#     Modified Focal Tversky loss  with GEV link             #
################################################################
def gradient_mft_gev(preds: np.ndarray, dtrain: xgb.DMatrix,delta=0.7, gamma2=1.0,tau=-0.25):           
    labels = dtrain.get_label()
    preds=np.exp(-np.power(1 + (tau*preds),-1/tau))
    phi1=preds*(labels+delta-1)-(delta*labels)
    phi2=preds*(delta-1)-(delta*labels)        
    L1_mF=-np.divide((delta*labels)*np.power((phi1/phi2),1/gamma2),gamma2*phi1*phi2)
    P1_gev=preds*(np.log(1/preds))**(tau+1)
    grad=L1_mF*P1_gev  
    return grad

def hessian_mft_gev(preds: np.ndarray, dtrain: xgb.DMatrix,delta=0.7, gamma2=1.0,tau=-0.25):      
    labels = dtrain.get_label()
    preds=np.exp(-np.power(1 + (tau*preds),-1/tau))
    phi1=preds*(labels+delta-1)-(delta*labels)
    phi2=preds*(delta-1)-(delta*labels)            
    L1_mF=-np.divide((delta*labels)*np.power((phi1/phi2),1/gamma2),gamma2*phi1*phi2)
    L2_mF=np.divide((delta*labels)*np.power((phi1/phi2),1/gamma2)*((2*(delta-1)*gamma2*phi1)-(labels*delta*(gamma2-1))),np.power(gamma2*phi1*phi2,2))
    P1_gev=preds*(np.log(1/preds))**(tau+1)
    P2_gev=-preds*((tau+1)*((np.log(1/preds))**-1)-1)*(np.log(1/preds))**(2*(tau+1))
    hess=(L1_mF*P2_gev)+((P1_gev**2)*L2_mF)
    return hess
'''Focal Tversky loss  with GEV link.'''
def modified_focal_tversky_gev (delta=0.7, gamma2=1.0,tau=-0.25):    
    def mFT_gev(preds: np.ndarray, dtrain: xgb.DMatrix):
        """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
        Link: https://arxiv.org/abs/1810.07842
        Parameters
        ----------
        gamma2 : float and non-zero, default value is 1.0
            The focal parameter controls the degree of enhancement or suppression of the positive  class (y=1).
        delta : float, default value is 0.7
            controls weight given to false positive and false negatives
        tau : float, default value is -0.25
            controls the skewness of the standard GEV-distribution based link"""
        grad = gradient_mft_gev(preds, dtrain,delta, gamma2,tau)
        hess = hessian_mft_gev(preds, dtrain,delta, gamma2,tau)
        return grad, hess
    return mFT_gev


################################################################
#     Modified Focal Tversky loss  with EEL link             #
################################################################
def gradient_mft_eel(preds: np.ndarray, dtrain: xgb.DMatrix,delta=0.7,gamma2=1.0,lamda=1.0, beta=1.0):           
    labels = dtrain.get_label()
    preds=np.power(1-np.power((1+np.exp(preds)),-lamda),beta)
    psi=1-np.power(preds,1/beta)
    phi1=preds*(labels+delta-1)-(delta*labels)
    phi2=preds*(delta-1)-(delta*labels)        
    L1_mF=-np.divide((delta*labels)*np.power((phi1/phi2),1/gamma2),gamma2*phi1*phi2)
    P1_eel=beta*lamda*psi*(1-np.power(psi,1/lamda))*np.power(1-psi,beta-1)
    grad=L1_mF*P1_eel  
    return grad

def hessian_mft_eel(preds: np.ndarray, dtrain: xgb.DMatrix,delta=0.7,gamma2=1.0,lamda=1.0, beta=1.0):      
    labels = dtrain.get_label()
    preds=np.power(1-np.power((1+np.exp(preds)),-lamda),beta)
    psi=1-np.power(preds,1/beta)
    phi1=preds*(labels+delta-1)-(delta*labels)
    phi2=preds*(delta-1)-(delta*labels)            
    L1_mF=-np.divide((delta*labels)*np.power((phi1/phi2),1/gamma2),gamma2*phi1*phi2)
    L2_mF=np.divide((delta*labels)*np.power((phi1/phi2),1/gamma2)*((2*(delta-1)*gamma2*phi1)-(labels*delta*(gamma2-1))),np.power(gamma2*phi1*phi2,2))
    P1_eel=beta*lamda*psi*(1-np.power(psi,1/lamda))*np.power(1-psi,beta-1) 
    P2_eel=-beta*lamda*np.power(psi,(beta+(1/lamda)))*(1-np.power(psi,1/lamda))*np.power((1/psi)-1,beta-2)*((1/psi)*(lamda*np.power(psi,-1/lamda)-lamda-1)-beta*lamda*(np.power(psi,-1/lamda)-1)+1)
    hess=(L1_mF*P2_eel)+((P1_eel**2)*L2_mF)
    return hess
'''Focal Tversky loss  with EEL link.'''
def modified_focal_tversky_eel (delta=0.7,gamma2=1.0,lamda=1.0, beta=1.0):    
    def mFT_eel(preds: np.ndarray, dtrain: xgb.DMatrix):
        """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
        Link: https://arxiv.org/abs/1810.07842
        Parameters
        ----------
        gamma2 : float and non-zero, default value is 1.0
            The focal parameter controls the degree of enhancement or suppression of the positive  class (y=1).
        delta : float, default value is 0.7
            controls weight given to false positive and false negatives
        lamda,beta : float, default value is 1.0
            shape parameters for the standard EEL-distribution based link"""
        grad = gradient_mft_eel(preds, dtrain,delta, gamma2,lamda,beta)
        hess = hessian_mft_eel(preds, dtrain,delta, gamma2,lamda,beta)
        return grad, hess
    return mFT_eel


################################################################
#    Symetric Unified Focal loss with EEL link                         #
################################################################
'''symetric Unified Focal loss with EEL link'''
def symmetric_unified_focal_eel (alpha=1.0,delta=0.7,gamma1=0.0, gamma2=1.0,lamda=1.0, beta=1.0,pi=0.5):    
    def sUF_eel(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        delta : float, default value is 0.7
            controls weight given to false positive and false negatives
        gamma1 : float, default value is 0.0
            The focal parameter for the cross-enthropy based loss.    
        gamma2 : float and non-zero, default value is 1.0
            The focal parameter for the Tversky loss.    
        lamda,beta : float, default value is 1.0
            shape parameters for the standard EEL-distribution based link
        pi : float, default value is 1.0. Takes values between 0 and 1.
            controls the component weights of the cross-enthropy-based and Dice-based losses.
        
        """
        grad_mf_eel = gradient_mf_eel(preds, dtrain,alpha, gamma1,lamda,beta)
        grad_mft_eel = gradient_mft_eel(preds, dtrain,delta, gamma2,lamda,beta)
        
        hess_mf_eel = hessian_mf_eel(preds, dtrain,alpha, gamma1,lamda,beta)
        hess_mft_eel = hessian_mft_eel(preds, dtrain,delta, gamma2,lamda,beta)
        
        grad_suf_eel=(pi*grad_mf_eel)+((1-pi)*grad_mft_eel)
        hess_suf_eel=(pi*hess_mf_eel)+((1-pi)*hess_mft_eel)
        
        return grad_suf_eel, hess_suf_eel
    return sUF_eel


#############################################################################################
#    Asymetric Positive class (y=1) Unified Focal loss with EEL link                         #
#############################################################################################
'''asymetric Positive class (y=1) Unified Focal loss with EEL link  '''
def asymmetric_unified_focal_eel1 (alpha=1.0,delta=0.7,gamma1=0.0, gamma2=1.0,lamda=1.0, beta=1.0,pi=0.5):    
    def aUF_eel(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        delta : float, default value is 0.7
            controls weight given to false positive and false negatives
        gamma1 : float, default value is 0.0
            The focal parameter for the positive class in the asymmetric cross-enthropy based loss.    
        gamma2 : float and non-zero, default value is 1.0
            The focal parameter for the Tversky loss.    
        lamda,beta : float, default value is 1.0
            shape parameters for the standard EEL-distribution based link
        pi : float, default value is 1.0. Takes values between 0 and 1.
            controls the component weights of the cross-enthropy-based and Dice-based losses.
        
        """
        grad_maf_eel = gradient_maf_eel1(preds, dtrain,alpha,lamda, gamma1,beta)
        grad_mft_eel = gradient_mft_eel(preds, dtrain,delta, gamma2,lamda,beta)
        
        hess_maf_eel = hessian_maf_eel1(preds, dtrain,alpha,lamda, gamma1,beta)
        hess_mft_eel = hessian_mft_eel(preds, dtrain,delta, gamma2,lamda,beta)
        
        grad_auf_eel=(pi*grad_maf_eel)+((1-pi)*grad_mft_eel)
        hess_auf_eel=(pi*hess_maf_eel)+((1-pi)*hess_mft_eel)
        
        return grad_auf_eel, hess_auf_eel
    return aUF_eel


#############################################################################################
#    Asymetric Positive class (y=0) Unified Focal loss with EEL link                         #
#############################################################################################
'''asymetric Unified Focal loss function.'''
def asymmetric_unified_focal_eel0 (alpha=1.0,delta=0.7,gamma1=0.0, gamma2=1.0,lamda=1.0, beta=1.0,pi=0.5):    
    def aUF_eel(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        delta : float, default value is 0.7
            controls weight given to false positive and false negatives
        gamma1 : float, default value is 0.0
            The focal parameter for the negative class in the asymmetric cross-enthropy based loss.    
        gamma2 : float and non-zero, default value is 1.0
            The focal parameter for the Tversky loss.    
        lamda,beta : float, default value is 1.0
            shape parameters for the standard EEL-distribution based link
        pi : float, default value is 1.0. Takes values between 0 and 1.
            controls the component weights of the cross-enthropy-based and Dice-based losses.
        
        """
        grad_maf_eel = gradient_maf_eel0(preds, dtrain,alpha,lamda, gamma1,beta)
        grad_mft_eel = gradient_mft_eel(preds, dtrain,delta, gamma2,lamda,beta)
        
        hess_maf_eel = hessian_maf_eel0(preds, dtrain,alpha,lamda, gamma1,beta)
        hess_mft_eel = hessian_mft_eel(preds, dtrain,delta, gamma2,lamda,beta)
        
        grad_auf_eel=(pi*grad_maf_eel)+((1-pi)*grad_mft_eel)
        hess_auf_eel=(pi*hess_maf_eel)+((1-pi)*hess_mft_eel)
        
        return grad_auf_eel, hess_auf_eel
    return aUF_eel


################################################################
#    Symetric Unified Focal loss with GEV link                         #
################################################################
'''symetric Unified Focal loss function.'''
def symmetric_unified_focal_gev (alpha=1.0,delta=0.7,gamma1=0.0, gamma2=1.0,tau=-0.25,pi=0.5):    
    def sUF_gev(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        delta : float, default value is 0.7
            controls weight given to false positive and false negatives
        gamma1 : float, default value is 0.0
            The focal parameter for the cross-enthropy based loss.    
        gamma2 : float and non-zero, default value is 1.0
            The focal parameter for the Tversky loss.    
        tau : float, default value is -0.25
            controls the skewness of the standard GEV-distribution based link
        pi : float, default value is 1.0. Takes values between 0 and 1.
            controls the component weights of the cross-enthropy-based and Dice-based losses.
        
        """
        grad_mf_gev = gradient_mf_gev(preds, dtrain,alpha, gamma1,tau)
        grad_mft_gev = gradient_mft_gev(preds, dtrain,delta, gamma2,tau)
        
        hess_mf_gev = hessian_mf_gev(preds, dtrain,alpha, gamma1,tau)
        hess_mft_gev = hessian_mft_gev(preds, dtrain,delta, gamma2,tau)
        
        grad_suf_gev=(pi*grad_mf_gev)+((1-pi)*grad_mft_gev)
        hess_suf_gev=(pi*hess_mf_gev)+((1-pi)*hess_mft_gev)
        
        return grad_suf_gev, hess_suf_gev
    return sUF_gev


##############################################################################################
#    Asymetric Positive CLass (y=1) Unified Focal loss with GEV link                         #
##############################################################################################
'''asymetric Unified Focal loss function.'''
def asymmetric_unified_focal_gev1 (alpha=1.0,delta=0.7,gamma1=0.0, gamma2=1.0,tau=-0.25,pi=0.5):    
    def aUF_gev(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        delta : float, default value is 0.7
            controls weight given to false positive and false negatives
        gamma1 : float, default value is 0.0
            The focal parameter for the positive class in the asymmetric cross-enthropy based loss.    
        gamma2 : float and non-zero, default value is 1.0
            The focal parameter for the Tversky loss.    
        tau : float, default value is -0.25
            controls the skewness of the standard GEV-distribution based link
        pi : float, default value is 1.0. Takes values between 0 and 1.
            controls the component weights of the cross-enthropy-based and Dice-based losses.
        
        """           
        grad_maf_gev = gradient_maf_gev1(preds, dtrain,alpha, gamma1,tau)
        grad_mft_gev = gradient_mft_gev(preds, dtrain,delta, gamma2,tau)
        
        hess_maf_gev = hessian_maf_gev1(preds, dtrain,alpha, gamma1,tau)
        hess_mft_gev = hessian_mft_gev(preds, dtrain,delta, gamma2,tau)
        
        grad_auf_gev=(pi*grad_maf_gev)+((1-pi)*grad_mft_gev)
        hess_auf_gev=(pi*hess_maf_gev)+((1-pi)*hess_mft_gev)
        
        return grad_auf_gev, hess_auf_gev
    return aUF_gev


##############################################################################################
#    Asymetric Positive CLass (y=0) Unified Focal loss with GEV link                         #
##############################################################################################
'''asymetric Unified Focal loss function.'''
def asymmetric_unified_focal_gev0 (alpha=1.0,delta=0.7,gamma1=0.0, gamma2=1.0,tau=-0.25,pi=0.5):    
    def aUF_gev(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Parameters
        ----------
        alpha : float, default value is 1.0.
            Penalty parameter control the degree of weight assigned to the misclassifcation of the positive class (y = 1).
        delta : float, default value is 0.7
            controls weight given to false positive and false negatives
        gamma1 : float, default value is 0.0
            The focal parameter for the negative class in the asymmetric cross-enthropy based loss.    
        gamma2 : float and non-zero, default value is 1.0
            The focal parameter for the Tversky loss.    
        tau : float, default value is -0.25
            controls the skewness of the standard GEV-distribution based link
        pi : float, default value is 1.0. Takes values between 0 and 1.
            controls the component weights of the cross-enthropy-based and Dice-based losses.
        
        """
        grad_maf_gev = gradient_maf_gev0(preds, dtrain,alpha, gamma1,tau)
        grad_mft_gev = gradient_mft_gev(preds, dtrain,delta, gamma2,tau)
        
        hess_maf_gev = hessian_maf_gev0(preds, dtrain,alpha, gamma1,tau)
        hess_mft_gev = hessian_mft_gev(preds, dtrain,delta, gamma2,tau)
        
        grad_auf_gev=(pi*grad_maf_gev)+((1-pi)*grad_mft_gev)
        hess_auf_gev=(pi*hess_maf_gev)+((1-pi)*hess_mft_gev)
        
        return grad_auf_gev, hess_auf_gev
    return aUF_gev


