import numpy as np
import pandas as pd

def pair_error(v, spin=True):
    if spin:
        v=(v+1)/2
        
    np.random.shuffle(v) #shuffle the data
    list_aa=[]
    
    def aa_type(aa):
        if np.sum(aa*[1,1,0,0]>0):
            return "P"
        else:
            return "A"
    
    #variables for the error
    err_count=0
    
    #calculation of the polarity and the error
    for i in range(len(v)):
        prec_aa='N'
        for j in range(5):
            aa=v[i, 4*j : 4*j+4]
            aa=aa_type(aa)
            list_aa.append(aa)
            if(aa==prec_aa):
                err_count+=1
            prec_aa=aa
            
    return err_count/(4*len(v))*100

def sequence_error(v1, spin=True, vmin=-1):
    if spin:
        v1=np.array((v1-vmin)/(1-vmin), dtype=int)
    df=pd.DataFrame(v1).astype('int')

    error=np.zeros(len(df))
    for i in range(len(df)):
        amminoacids = np.reshape(np.array(df.loc[i]), (5,4))
        for a in range(4):
            if list(np.where(amminoacids[a,:]))[0][0] <2 and list(np.where(amminoacids[a+1,:]))[0][0] <2 :
                error[i]=1
            elif list(np.where(amminoacids[a,:]))[0][0] >1 and list(np.where(amminoacids[a+1,:]))[0][0] >1:
                error[i]=1
    return sum(error)*100/len(df)

def KL_divergence(p, q):
    return np.sum(p*np.log(p/q))

def JS_divergence(p,q):
    m = 0.5*(p+q)
    js = 0.5*(KL_divergence(p, m) + KL_divergence(q, m))
    return js

def p(v, spin=True):
    
    if spin:
        vp = (v+1)/2 
        
    p = np.sum(vp, axis=0)/v.shape[0]
    return p