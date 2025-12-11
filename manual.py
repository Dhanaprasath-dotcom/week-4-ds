import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Creat  clear data groups

data = {
    'Annual Income':[33,44,57,87,65,45,34,76,54,34],
    'Spending Score':[90,98,87,80,90,70,60,50,50,40], 

}
#convert into data frame
df= pd.DataFrame(data)
print(df)

# select columns
X = df[['Annual Income','Spending Score']]

# scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add cluster colum 
df['Cluster'] = kmeans.labels_
cluster1=df[df['Cluster']==0]
print(df)
print(cluster1)
print("Cluster 1 Data:")
cluster2=df[df['Cluster']==1]
print(df)
print(cluster1)
print("Cluster 1 Data:")
cluster3=df[df['Cluster']==2]
print(df)






