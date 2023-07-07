#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Lab Session

# ## Learning outcomes:
# - Exploratory data analysis and data preparation for model building.
# - PCA for dimensionality reduction.
# - K-means and Agglomerative Clustering

# ## Problem Statement
# Based on the given marketing campigan dataset, segment the similar customers into suitable clusters. Analyze the clusters and provide your insights to help the organization promote their business.

# ## Context:
# - Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
# - Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.

# ## About dataset
# - Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?datasetId=1546318&sortBy=voteCount
# 
# ### Attribute Information:
# - ID: Customer's unique identifier
# - Year_Birth: Customer's birth year
# - Education: Customer's education level
# - Marital_Status: Customer's marital status
# - Income: Customer's yearly household income
# - Kidhome: Number of children in customer's household
# - Teenhome: Number of teenagers in customer's household
# - Dt_Customer: Date of customer's enrollment with the company
# - Recency: Number of days since customer's last purchase
# - Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# - MntWines: Amount spent on wine in last 2 years
# - MntFruits: Amount spent on fruits in last 2 years
# - MntMeatProducts: Amount spent on meat in last 2 years
# - MntFishProducts: Amount spent on fish in last 2 years
# - MntSweetProducts: Amount spent on sweets in last 2 years
# - MntGoldProds: Amount spent on gold in last 2 years
# - NumDealsPurchases: Number of purchases made with a discount
# - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# - NumWebPurchases: Number of purchases made through the company’s website
# - NumCatalogPurchases: Number of purchases made using a catalogue
# - NumStorePurchases: Number of purchases made directly in stores
# - NumWebVisitsMonth: Number of visits to company’s website in the last month

# ### 1. Import required libraries

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


# ### 2. Load the CSV file (i.e marketing.csv) and display the first 5 rows of the dataframe. Check the shape and info of the dataset.

# In[56]:


df = pd.read_csv('marketing.csv')
df.head()


# ### 3. Check the percentage of missing values? If there is presence of missing values, treat them accordingly.

# In[57]:


missing_percentage = df.isnull().sum() / len(df) * 100
print(missing_percentage)


# ### 4. Check if there are any duplicate records in the dataset? If any drop them.

# In[58]:


duplicates = df.duplicated()
print(duplicates.sum())

df = df.drop_duplicates()


# ### 5. Drop the columns which you think redundant for the analysis 

# In[59]:


columns_to_drop = ['ID', 'Dt_Customer']
df = df.drop(columns_to_drop, axis=1)


# ### 6. Check the unique categories in the column 'Marital_Status'
# - i) Group categories 'Married', 'Together' as 'relationship'
# - ii) Group categories 'Divorced', 'Widow', 'Alone', 'YOLO', and 'Absurd' as 'Single'.

# In[60]:


print(df['Marital_Status'].unique())

relationship_categories = ['Married', 'Together']
single_categories = ['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd']

df['Marital_Status'] = df['Marital_Status'].replace(relationship_categories, 'relationship')
df['Marital_Status'] = df['Marital_Status'].replace(single_categories, 'Single')


# In[61]:


df['Marital_Status'].value_counts()


# ### 7. Group the columns 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', and 'MntGoldProds' as 'Total_Expenses'

# In[62]:


expense_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Expenses'] = df[expense_columns].sum(axis=1)


# In[63]:


df['Total_Expenses']


# ### 8. Group the columns 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', and 'NumDealsPurchases' as 'Num_Total_Purchases'

# In[64]:


purchase_columns = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases']
df['Num_Total_Purchases'] = df[purchase_columns].sum(axis=1)


# In[65]:


df['Num_Total_Purchases']


# ### 9. Group the columns 'Kidhome' and 'Teenhome' as 'Kids'

# In[66]:


df['Kids'] = df['Kidhome'] + df['Teenhome']


# In[67]:


df['Kids']


# ### 10. Group columns 'AcceptedCmp1 , 2 , 3 , 4, 5' and 'Response' as 'TotalAcceptedCmp'

# In[68]:


response_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df['TotalAcceptedCmp'] = df[response_columns].sum(axis=1)


# In[69]:


df['TotalAcceptedCmp'] 


# ### 11. Drop those columns which we have used above for obtaining new features

# In[70]:


columns_to_drop = expense_columns + purchase_columns + response_columns + ['Kidhome', 'Teenhome']
df = df.drop(columns_to_drop, axis=1)


# ### 12. Extract 'age' using the column 'Year_Birth' and then drop the column 'Year_birth'

# In[71]:


df['Age']= 2022 - df['Year_Birth']


# In[72]:


df['Age']


# ### 13. Encode the categorical variables in the dataset

# In[73]:


df_encoded = pd.get_dummies(df)
df_encoded.fillna(df_encoded.mean(), inplace=True)


# ### 14. Standardize the columns, so that values are in a particular range

# In[74]:



scaled_features = StandardScaler().fit_transform(df_encoded.values)  # Standardize the data
scaled_features_df = pd.DataFrame(scaled_features, index=df_encoded.index, columns=df_encoded.columns)


# In[75]:


scaled_features_df.head(3)


# # 15. Apply PCA on the above dataset and determine the number of PCA components to be used so that 90-95% of the variance in data is explained by the same.

# In[76]:


cov_matrix = np.cov(scaled_features.T)
cov_matrix


# In[77]:


eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
print('eigin vals:','\n',eig_vals)
print('\n')
print('eigin vectors','\n',eig_vectors)


# In[78]:


total = sum(eig_vals)
var_exp = [(i/total)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('Explained Variance:', var_exp)
print('Cummulative Variance Explained:', cum_var_exp)


# In[79]:


plt.bar(range(len(var_exp)), var_exp, align='center', color='lightgreen', edgecolor='black', label='Explained Variance')
plt.step(range(len(var_exp)), cum_var_exp, where='mid', color='red', label='Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.legend()
plt.show()


# ### 16. Apply K-means clustering and segment the data (Use PCA transformed data for clustering)

# In[80]:


pca = PCA(n_components=8)

pca_df = pd.DataFrame(pca.fit_transform(scaled_features_df),columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'])
pca_df.head()


# In[81]:


cluster_errors =[]
cluster_range = range(2,15)
for num_cluster in cluster_range:
    cluster = KMeans(num_cluster,random_state=100)
    cluster.fit(pca_df)
    cluster_errors.append(cluster.inertia_)


# In[82]:


cluster_df = pd.DataFrame({'num_clusters':cluster_range,'cluster_errors':cluster_errors})
plt.figure(figsize=[15,5])
plt.plot(cluster_df['num_clusters'],cluster_df['cluster_errors'],marker='o',color='b')
plt.show()


# In[83]:


KMeans = KMeans(n_clusters=3, random_state=100)
KMeans.fit(pca_df)


# In[84]:


label=pd.DataFrame(KMeans.labels_,columns=['Label'])


# In[85]:


KMeans_df =pca_df.join(label)
KMeans_df.head()


# In[86]:


KMeans_df['Label'].value_counts()


# In[87]:


sns.scatterplot(KMeans_df['PC1'],KMeans_df['PC2'],hue='Label',data=KMeans_df)
plt.show()


# ### 17. Apply Agglomerative clustering and segment the data (Use Original data for clustering), and perform cluster analysis by doing bivariate analysis between the cluster label and different features and write your observations.

# In[88]:


plt.figure(figsize=[18,5])
merg = linkage(scaled_features, method='ward')
dendrogram(merg, leaf_rotation=90,)
plt.xlabel('Datapoints')
plt.ylabel('Euclidean distance')
plt.show()


# In[89]:


from sklearn.metrics import silhouette_score


# In[90]:


for i in range (2,15):
    hier = AgglomerativeClustering(n_clusters=i)
    hier = hier.fit(scaled_features_df)
    labels = hier.fit_predict(scaled_features_df)
    print(i,silhouette_score(scaled_features_df,labels))


# In[91]:


hie_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
hie_cluster_model = hie_cluster.fit(scaled_features_df)


# In[92]:


df_label1 = pd.DataFrame(hie_cluster_model.labels_,columns=['Labels'])
df_label1.head(5)


# In[93]:


df_hier = df.join(df_label1)
df_hier.head()


# ### Visualization and Interpretation of results

# In[94]:


sns.barplot(df_hier['Labels'],df_hier['Total_Expenses'])
plt.show()


# In[95]:


sns.barplot(df_hier['Labels'],df_hier['Income'])
plt.show()


# In[99]:


sns.barplot(df_hier['Marital_Status'],df_hier['Labels'])
plt.show()


# In[104]:


sns.barplot(df_hier['Labels'],df_hier['Num_Total_Purchases'])
plt.show()


# In[ ]:





# -----
# ## Happy Learning
# -----
