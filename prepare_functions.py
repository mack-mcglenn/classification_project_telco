#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import numpy as np
import env
import pandas as pd
import acquire_copy as acq
import matplotlib as plt
import seaborn as  sb
import scipy.stats as stats
from pydataset import data
from sklearn.model_selection import train_test_split


# In[33]:


iris = acq.get_iris_data()
titanic = acq.get_titanic_data()
telco = acq.get_telco_data()


# In[37]:


def prep_iris(df):
    """This function preps data in the iris csv (via the get_iris_data() function
    in acquire_copy) for future use"""
    df = df.drop(columns=['species_id', 'measurement_id'])
    df = df.rename(columns={'species_name':'species'})
    df=  pd.concat([df, (pd.get_dummies(df['species']))], axis = 1)
    
    return df

def prep_titanic(titanic):
    titanic=titanic.drop(columns=['embarked', 'class', 'passenger_id', 'deck'])
    titanic['age'] = ['age'].fillna(titanic['age'].mean())
    titanic['embark_town'] = titanic['embark_town'].fillna(titanic['age'].mean())
    titanic=  pd.concat([titanic, (pd.get_dummies(titanic['sex', 'embark_town']))], axis = 1)
    titanic = titanic.drop(columns=['embark_town', 'sex'])
    
    return titanic

def prep_telco(telco):
    """This function presp data from the telco csv (acquired via the get_telco_data() function in 
    acquire_copy and preps it for future use."""
    
    # drop unnecessary/redundant columns
    telco=telco.drop(columns=['internet_service_type_id', 'payment_type_id', 'contract_type_id'],    inplace=True)
    
    # convert monthly charges column from str to float
    telco['total_charges']= telco['total_charges'].str.replace(' ', '0')
    telco['total_charges']= telco['total_charges'].astype('float')
    
    # convert binary cat variables to numeric
    telco['churn_bin'] = telco['churn'].map({'Yes': 1, 'No': 0})
    telco['gender_bin'] = telco['gender'].map({'Female': 1, 'Male': 0})
    telco['partner_bin'] = telco['partner'].map({'Yes': 1, 'No': 0})
    telco['dependents_bin'] = telco['dependents'].map({'Yes': 1, 'No': 0})
    telco['paperless_billing_bin'] = telco['paperless_billing'].map({'Yes': 1, 'No': 0})
    telco['phone_service_bin'] = telco['phone_service'].map({'Yes': 1, 'No': 0})

    
    # Dummy variables for enby cat variables
    radioshack= pd.get_dummies( telco[['multiple_lines', \
                                       'online_security', \
                                       'online_backup', \
                                       'device_protection', \
                                       'tech_suport', \
                                       'payment_type', \
                                       'streaming_tv', \
                                       'streaming_movies',\
                                       'internet_Sercive_type',\
                                       'contract_type' 
                                      ]], drop_first= True)
    telco= pd.concat([telco, radioshack], axis=1)
    
    return telco   
                                       
    





# In[30]:


def tvt_split(df, target):
     # 20% test, 80% train_validate
# then of the 80% train_validate: 25% validate, 75% train.
# Final split will be 20/20/60 between datasets
  
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, val = train_test_split(train, test_size=.30, random_state=123, stratify=train[target])

    return train,val,test




