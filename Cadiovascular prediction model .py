#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas-profiling')

import pandas as pd
import numpy as np 
import seaborn as sns
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LogisticRegression


# In[11]:


dataset = pd.read_csv("Heart_Disease_Prediction.csv")
display(dataset.head(10))


# In[12]:


dataset.describe()


# In[13]:


dataset.info()


# In[14]:


dataset.isnull().sum()


# In[15]:


total_missing_rows = dataset.isna().any(axis=1).sum()
dataset.dropna(axis=0,inplace=True)


# In[16]:


dataset.drop_duplicates(inplace=True)
total_missing_rows = dataset.isna().sum().sum()


# In[17]:


total_missing_rows = dataset.isna().any(axis=1).sum()
dataset.dropna(axis=0,inplace=True)
duplicate_sum = dataset.duplicated().sum()
if duplicate_sum:
    print('Duplicates Rows in Dataset are : {}'.format(duplicate_sum))
else:
    print('Dataset contains no Duplicate Values')


# In[18]:


dataset.dropna(axis=0,inplace=True)


# In[19]:


dataset.describe()


# In[20]:


def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataset[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    plt.show()
draw_histograms(dataset,dataset.columns,6,3)


# In[21]:


from statsmodels.tools import add_constant as add_constant
dataset_constant = add_constant(dataset)
dataset_constant.head()


# In[22]:


sns.pairplot(data=dataset)


# In[46]:


target_name = 'Heart Disease'
data_target = dataset[target_name]
data = dataset.drop([target_name], axis=1)
data.head(10)






# In[55]:


data_2 = dataset.drop(['ST depression', 'Heart Disease'], axis=1)
data_2.head(10)


# In[56]:


num_samples_features = len(data_2)
num_samples_target = len(data_target)

if num_samples_features == num_samples_target:
    print("Number of samples in feature data and target variable are consistent.")
else:
    print("Number of samples in feature data and target variable are inconsistent.")


# In[58]:


from statsmodels.tools import add_constant as add_constant
dataset_constant = add_constant(data_2)
dataset_constant.head()


# In[88]:


train, test, target, target_test = train_test_split(data_2, data_target, test_size=0.3, random_state=50)
from sklearn.model_selection import train_test_split



#data_2_train, data_2_test, data_target_train, data_target_test = train_test_split(data_2, data_target, test_size=0.2, random_state=42)


# In[89]:


train.head(10)


# In[90]:


test.head(10)


# In[91]:


train.info()


# In[92]:


test.info()


# In[93]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




# In[94]:


# Standardize the feature data (optional but recommended)
scaler = StandardScaler()
train = scaler.fit_transform(data_2_train)
test = scaler.transform(data_2_test)


# In[95]:


logreg = LogisticRegression()
logreg.fit(train,target)
acc_log = round(logreg.score(train,target) * 100, 2)
acc_log

