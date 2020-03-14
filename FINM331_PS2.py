import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

### Problem 1 ###

columns = ["country","100m","200m","400m","800m","1500m","Marathon"]
data = pd.read_csv("p1.txt",delimiter="\t",header=None, names = columns)

### Problem 1 (a) ###

sample_means=data.mean(axis=0)
#sample_means
round(sample_means,2) # round the numbers to two decimal places

### Problem 1 (b) ###

# Get covariance matrix
covm = np.cov(data.T) # compute covariance matrix
covm_rounded=np.around(covm,decimals=2) # round the numbers to two decimal places
pd.DataFrame(covm_rounded)

# Get correlation matrix
dev_sqrd=pd.DataFrame((data-sample_means)**2) # subtract sample mean from each data and then square the result
sam_var=dev_sqrd.sum(axis=0)/(data.shape[0]-1) # perform column-wise sum and divide by the number of samples-1(53)
D=np.diag(1/np.sqrt(sam_var)) # This is D^(-1/2) in the lecture note
R=np.dot(np.dot(D,covm),D) # correlation matrix R = D^(-1/2)SD^(-1/2)
R=np.around(R,decimals=2)
pd.DataFrame(R) # show the correlation matrix

### Problem 1 (c) ###

# Get correlation matrix of the logarithm of the data
log10data=np.log10(data)
#Get mean of the log10data
means_log10=log10data.mean(axis=0)
covm = np.cov(log10data.T) # compute covariance matrix

# Get correlation matrix
dev_sqrd=pd.DataFrame((log10data-means_log10)**2) # subtract sample mean from each data and then square the result
sam_var=dev_sqrd.sum(axis=0)/(data.shape[0]-1) # perform column-wise sum and divide by the number of samples-1(53)
D=np.diag(1/np.sqrt(sam_var)) # This is D^(-1/2) in the lecture note
R=np.dot(np.dot(D,covm),D) # correlation matrix R = D^(-1/2)SD^(-1/2)
R=np.around(R,decimals=2)
pd.DataFrame(R) # show the correlation matrix

### Problem 3 (a) ###

cov = np.array([[2,-1,1],[-1,5,0],[1,0,3]]) # Define a covariance matrix
w, v = np.linalg.eig(cov) # store eigenvalues in the variable w
# Print out eigenvalues of the covariance matrix
w
# Print out eigenvectors
v

### Problem 4 ###

columns = ["country","100m","200m","400m","800m","1500m","Marathon"]
data = pd.read_csv("p1.txt",delimiter="\t",header=None, names = columns)
data = data.drop(['country'], axis=1) # drop "country column"

### Problem 4 (a) ###

std_data = (data - data.mean(axis=0))/data.std(axis=0) # get standardized data
std_dataT = std_data.T # Transposed standardized data
std_dataT = std_data.T # Transposed standardized data
R = std_dataT.dot(std_data)/(std_data.shape[0]-1) # compute the sample correlation matrix R
w, v = np.linalg.eig(R) # get eigenvalues of R
print("eigenvalues of R is: ", w)
print("eigenvectors of R is: ",v[:,0],v[:,1],v[:,2],v[:,3],v[:,4],v[:,5])
print("sum of the eigenvalues of the sample coorelation matrix R is: ", np.sum(w))

### Problem 4 (b) ###

# Part(i)
svd_data = np.linalg.svd(std_data,full_matrices=False) # Perform svd on the standardized data and store principal components in vh
print("PC1 from SVD is: ", svd_data[2][0]," and the corresponding eigenvector from the correlation matrix is: ", v[:,0])
print("PC2 from SVD is: ", svd_data[2][1]," and the corresponding eigenvector from the correlation matrix is: ", v[:,1])

# Part(ii)
print("The percentage of tatal sample variation explained by the first two PCs is: ",(w[0]+w[1])/sum(w)*100,"%.")
w[1]/sum(w)*100

### Problem 4 (c) ###

#coord = pd.DataFrame(np.zeros((54,2))) # Initialize coordinates
# to get coordinates of the datapoints projected onto the PC1-PC2 plane, we simply need to multiply 1st and 2nd
# singular values with the first and second columns of U in SVD.
r_sv = pd.DataFrame(svd_data[2]).iloc[0:2] # Take the first two singular vectors
s_values = pd.DataFrame(svd_data[1]).iloc[0:2] # Take the first two singular values
l_sv = pd.DataFrame(svd_data[0]).iloc[:,0:2] # Take the first two columns of U
coordinates = pd.DataFrame(np.multiply(l_sv,s_values.T)) # Take element-wise product of l_sv and s_values
coordinates.columns =['PC1','PC2']
coordinates.index=data.index
# plot datapoints on PC1-PC2 plane
fig, ax = plt.subplots()
ax.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1],label = data.index)
for i, txt in enumerate(data.index):
    ax.annotate(txt, (coordinates.iloc[:,0][i], coordinates.iloc[:,1][i]))
ax.set_xlabel('PC1') # Label x and y axes
ax.set_ylabel('PC2')
plt.savefig('countries.png')
# Rank the nations based on their scores on the first PC
rank = coordinates.sort_values(by=['PC1'], ascending=False)
# Top six countries
print("The top six countries along PC1 is: ", rank.iloc[0:6].index.tolist())
# Last three countries
print("The last three countries along PC1 is: ", rank.iloc[51:54].index.tolist())
