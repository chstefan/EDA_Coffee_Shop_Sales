#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data_path = '/Users/chantalstefan/Documents/GitProjects/Coffee_Shop_Sales_EDA/Coffee_Sales_Data.csv'
sales_data = pd.read_csv(data_path)

sales_data_head = sales_data.head()

sales_data_info = sales_data.info()


# In[2]:


# Check for missing values
print(sales_data.isnull().sum())


# In[3]:


import pandas as pd

# Convert to datetime
sales_data['transaction_date'] = sales_data['transaction_date'].astype(str)
sales_data['transaction_time'] = sales_data['transaction_time'].astype(str)

sales_data['transaction_datetime'] = pd.to_datetime(
    sales_data['transaction_date'] + ' ' + sales_data['transaction_time']
)

sales_data.drop(columns=['transaction_date', 'transaction_time'], inplace=True)


# In[4]:


sales_data = sales_data.drop_duplicates()


# In[5]:


#standardize numerical values

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
sales_data[['transaction_qty', 'unit_price', 'Total_Bill']] = scaler.fit_transform(sales_data[['transaction_qty', 'unit_price', 'Total_Bill']])

print("Mean after standardization:\n", sales_data.mean(axis=0))
print("Standard Deviation after standardization:\n", sales_data.std(axis=0))


# In[6]:

#get dummies for data
sales_data_with_dummies = pd.get_dummies(sales_data, columns=['store_location', 'product_category', 'product_type', 'Size', 'Month Name', 'Day Name'], drop_first=True)
sales_data_with_dummies.head()


# In[7]:


# Display summary statistics for numerical columns
summary_stats = sales_data_with_dummies.describe()
print(summary_stats)


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for numerical attributes
numerical_cols = ['transaction_qty', 'unit_price', 'Total_Bill']  # Update with relevant numerical columns
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(sales_data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[9]:


# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
corr_matrix = sales_data_with_dummies[numerical_cols].corr()  # Update if you have more numerical columns
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[10]:


# 'product_category' distribution - categorical Analysis
plt.figure(figsize=(10, 6))
sales_data_with_dummies['product_category_Tea'] = sales_data_with_dummies.filter(regex='product_category_').sum(axis=1)  # Adjust based on actual encoding
sns.countplot(x='product_category_Tea', data=sales_data_with_dummies)
plt.title('Distribution of Product Categories')
plt.show()


# In[11]:


#Time Series Analysis
# Assuming 'transaction_datetime' is your datetime column
sales_data_with_dummies.set_index('transaction_datetime', inplace=True)

# Monthly Total Bill trend
monthly_sales = sales_data_with_dummies['Total_Bill'].resample('M').sum()
plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title('Monthly Sales Trend')
plt.ylabel('Total Sales')
plt.xlabel('Month')
plt.show()


# In[12]:


# Box plots for numerical variables
numerical_cols = ['transaction_qty', 'unit_price', 'Total_Bill']  # Example numerical columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=sales_data[col])
    plt.show()


# In[13]:

sns.pairplot(sales_data_with_dummies[numerical_cols])
plt.show()


# In[14]:


#Relationship between 'unit_price' and 'Total_Bill'
plt.scatter(sales_data_with_dummies['unit_price'], sales_data_with_dummies['Total_Bill'])
plt.xlabel('Unit Price')
plt.ylabel('Total Bill')
plt.title('Unit Price vs Total Bill')
plt.show()


# In[15]:


#Sales Forcasting Model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Feature selection and Encoding
features = ['store_id', 'product_id', 'Hour', 'Month', 'Day of Week', 'Size', 'product_category']
X = sales_data[features]
y = sales_data['Total_Bill']


categorical_features = ['store_id', 'product_id', 'Size', 'product_category']
numeric_features = ['Hour', 'Month', 'Day of Week']

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a modeling pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# model initialisation and rmse
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('RMSE:', rmse)




