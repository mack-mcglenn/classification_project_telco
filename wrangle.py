




from sklearn.model_selection import train_test_split


def connect(db):
    
    """This function will pull the information from my env file (username, password, host,
    database) to connect to Codeup's MySQL database"""
    
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'



def get_telco_data():
    """This function will check whether or not there is a csv for the telco data saved 
      locally. If no such csv exists locally, this function will pull the telco data from
    Codeup's MySQL database and return as a dataframe based on the credentials provided in 
    the env.py file in use"""

    if os.path.isfile('telco.csv'):
        df = pd.read_csv('telco.csv', index_col=0)
    else:
        query = """select * from customers
        join contract_types using (contract_type_id)
        join internet_service_types using (internet_service_type_id)
        join payment_types using (payment_type_id)"""
        connection = connect('telco_churn')
        df = pd.read_sql(query, connection)
        df.to_csv('telco.csv')
    return df




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


# For charts and such
def plot_histogram(x, y, z):
    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist([x[y==0], x[y==1]], bins=30, alpha=0.5, label=['Churn = No', 'Churn = Yes'])
    y_vals, _ = np.histogram(x, bins=30)
    ax.plot([x.mean(), x.mean()], [0, np.max(y_vals)], 'r--', linewidth=2)
    for i, group in enumerate(['Low', 'Medium', 'High']):
        ax.plot([x[z==group].mean(), x[z==group].mean()], [0, np.max(y_vals)], '--', linewidth=2, label=f'{group} monthly charge')
    ax.legend(loc='upper right')
    ax.set_xlabel('Tenure (months)')
    ax.set_ylabel('Count')
    ax.set_title('Churn by Monthly Charges and Tenure')

    plt.show()

# More cleaning

monthlycats = pd.cut(train['monthly_charges'],3)
monthlycats



# Color palletes for data visualization
my_pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}


