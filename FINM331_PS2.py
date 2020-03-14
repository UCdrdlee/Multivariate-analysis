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

### Problem 5 ###

columns5 = ["Wind","Radiation","CO","NO","NO2","O3","HC"]
data5=pd.read_csv("p5.txt",delim_whitespace=True,header=None, names = columns5) # read and store p5.txt file

### Problem 5 (a) ###

# Compute eigenvalues from EVD of the sample covariance matrix
covm5=np.cov(data5.T) # compute covariance matrix
w, v = np.linalg.eig(covm5) # get eigenvalues of R and store them in variable w
print(v[:,0])
print("In non-standardized data the first two PCs explain ",(w[0]+w[1])/sum(w)*100,
      "% of the variability in data while the first PC explains only", w[0]/sum(w)*100,"% of the variability.")
print("Therefore we just need first two principal components.")

# Compute eigenvalues from EVD of the sample covariance matrix from the standardized data
std_data5 = (data5 - data5.mean(axis=0))/data5.std(axis=0) # get standardized data
std_covm5=np.cov(std_data5.T) # compute covariance matrix
y, z = np.linalg.eig(std_covm5)
y=-np.sort(-y) # Sort eigenvalues in descending order
print("In the standardized data the first six PCs explain ",(sum(y[0:6]))/sum(y)*100,"% of the variability in data",
      "while the first five PCs explain ",(sum(y[0:5]))/sum(y)*100,"% of the variability")
print("Therefore we need first six principal components.")

### Problem 5 (b) ###

# Plot a screeplot using the original (non-standardized) data.
fig = plt.figure(figsize=(8,5))
eigvals = np.arange(w.size)
plt.plot(eigvals, w, 'ro-', linewidth=2, color='r')
plt.title('Scree Plot using the original data')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from EVD'], loc='best', borderpad=0.3,
                 shadow=False,
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
plt.show()

# Plot a screeplot using the standardized data.
fig2 = plt.figure(figsize=(8,5))
eigvals2 = np.arange(y.size)
plt.plot(eigvals2, y, 'ro-', linewidth=2,color='b')
plt.title('Scree Plot using the standardized data')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from EVD'], loc='best', borderpad=0.3,
                 shadow=False,
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
plt.show()

### Problem 6 ###

# Load the data as a dataframe. Data was saved in an excel sheet in a xlsx format.
data6 = pd.read_excel("p6-data.xlsx",header=1,index_col=0)

### Problem 6(a) ###

# Standardizing

# step 1: mean-center the data
mean_centered_data6 = (data6 - data6.mean(axis=0))
#step 2: divide the mean-centered data by the respective standard deviation
std_data6 = mean_centered_data6/data6.std(axis=0)
#Denote the standardized data matrix by A_hat
A_hat = std_data6

### Problem 6(b) ###

# Singular Value Decomposition

#Find the first two singular vectors of A_hat
svd_data = np.linalg.svd(A_hat,full_matrices=False) # This returns svd decomposition of data with U, Sigma and V
r_sv = pd.DataFrame(svd_data[2]).iloc[0:2] # Take the first two singular vectors
r_sv_original = r_sv
#print("The first two right singular vectors: ", r_sv)
s_values = pd.DataFrame(svd_data[1]).iloc[0:2] # Take the first two singular values
#print("s_values are: ", s_values)
l_sv = pd.DataFrame(svd_data[0]).iloc[:,0:2] # Take the first two columns of U
#print("l_sv are: ", l_sv)
coordinates = np.multiply(l_sv,s_values.T) # Take element-wise product of l_sv and s_values
#print(coordinates)

#Plot the coordinates
fig, ax = plt.subplots()
ax.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1],label = A_hat.index)
for i, txt in enumerate(A_hat.index):
    ax.annotate(txt, (coordinates.iloc[:,0][i], coordinates.iloc[:,1][i]))
ax.set_xlabel('PC1') # Label x and y axes
ax.set_ylabel('PC2')

### Problem 6(c) ###

coordinates2 = np.multiply(r_sv.T,s_values.T)
fig2, ax2 = plt.subplots()
ax2.scatter(coordinates2.iloc[:,0], coordinates2.iloc[:,1],label = A_hat.columns)
for i, txt in enumerate(A_hat.columns):
    ax2.annotate(txt, (coordinates2.iloc[:,0][i], coordinates2.iloc[:,1][i]))
ax2.set_xlabel('PC1') # Label x and y axes
ax2.set_ylabel('PC2')

### Problem 6(d) ###

# Overlaying the two scatterplots. Red dots are the countries and green dots are the demographic variables.
plt.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1],label = A_hat.index,color='r')
plt.scatter(coordinates2.iloc[:,0], coordinates2.iloc[:,1],label = A_hat.columns,color='g')
plt.show()

### Problem 6(e) ###

# Outlier removed

# Remove Singapore and Hong Kong from the original data
new_data = data6.drop(index=['Hong Kong','Singapore'])
#step 1: mean-center the data
mc_new_data = (new_data - new_data.mean(axis=0))
#step 2: divide the mean-centered data by the respective standard deviation
std_new_data = mc_new_data/new_data.std(axis=0)
#Denote the standardized data matrix by A_hat
A_hat = std_new_data

svd_data2 = np.linalg.svd(A_hat,full_matrices=False) # This returns svd decomposition of data with U, Sigma and V
r_sv2 = pd.DataFrame(svd_data2[2]).iloc[0:2] # Take the first two eight singular vectors
s_values2 = pd.DataFrame(svd_data2[1]).iloc[0:2] # Take the first two singular values
l_sv2 = pd.DataFrame(svd_data2[0]).iloc[:,0:2] # Take the first two columns of U
coordinates3 = np.multiply(l_sv2,s_values2.T) # Take element-wise product of l_sv and s_values

# Plot scatterplot from the data excluding Hong Kong and Singapore
plt.figure(figsize=(40,40))
fig3, ax3 = plt.subplots()
ax3.scatter(coordinates3.iloc[:,0], coordinates3.iloc[:,1],label = new_data.index)
for i, txt in enumerate(new_data.index):
    ax3.annotate(txt, (coordinates3.iloc[:,0][i], coordinates3.iloc[:,1][i]))
plt.savefig('scatter_overview.png')
ax3.set_xlabel('PC1') # Label x and y axes
ax3.set_ylabel('PC2')

### Problem 6(f) ###

# Standardizing

# step 1: mean-center the data
mean_centered_data6 = (data6 - data6.mean(axis=0))

# Denote the standardized data matrix by A_hat
A_hat = mean_centered_data6

# Compute the standard deviation of the data
data6_std = data6.std(axis=0)

# Singular Value Decomposition

#Find the first two singular vectors of A_hat
svd_data = np.linalg.svd(A_hat,full_matrices=False) # This returns svd decomposition of data with U, Sigma and V
r_sv = pd.DataFrame(svd_data[2]).iloc[0:2] # Take the first two singular vectors
s_values = pd.DataFrame(svd_data[1]).iloc[0:2] # Take the first two singular values
l_sv = pd.DataFrame(svd_data[0]).iloc[:,0:2] # Take the first two columns of U
coordinates = np.multiply(l_sv,s_values.T) # Take element-wise product of l_sv and s_values

# Compute the weighted standard deviation of the data
weighted_std = np.dot(r_sv,data6_std)

# Divide the coordinates by the weighted standard deviation. Explained below
coordinates = np.divide(coordinates,-weight_std)

#Plot the coordinates
fig, ax = plt.subplots()
ax.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1],label = A_hat.index)
for i, txt in enumerate(A_hat.index):
    ax.annotate(txt, (coordinates.iloc[:,0][i], coordinates.iloc[:,1][i]))
ax.set_xlabel('PC1') # Label x and y axes
ax.set_ylabel('PC2')

coordinates2 = np.multiply(r_sv.T,s_values.T)
data6_std=data6.std(axis=0)

coordinates2 = np.divide(coordinates2.T,data6_std).T # divide the coordinates by the standard deviations
fig2, ax2 = plt.subplots()
ax2.scatter(coordinates2.iloc[:,0], coordinates2.iloc[:,1],label = A_hat.columns)
for i, txt in enumerate(A_hat.columns):
    ax2.annotate(txt, (coordinates2.iloc[:,0][i], coordinates2.iloc[:,1][i]))
ax2.set_xlabel('PC1') # Label x and y axes
ax2.set_ylabel('PC2')

# Overlaying the two scatterplots. Red dots are the countries and green dots are the demographic variables.
plt.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1],label = A_hat.index,color='r')
plt.scatter(coordinates2.iloc[:,0], coordinates2.iloc[:,1],label = A_hat.columns,color='g')
plt.show()

### Problem 6(g) ###

# Effect of mean centering

# Singular Value Decomposition

#Find the first two singular vectors of the original data: 'data6'
svd_data = np.linalg.svd(data6,full_matrices=False) # This returns svd decomposition of data with U, Sigma and V
r_sv = pd.DataFrame(svd_data[2]).iloc[0:2] # Take the first two singular vectorsprint("r_sv is: ", r_sv)
r_sv_original = r_sv
s_values = pd.DataFrame(svd_data[1]).iloc[0:2] # Take the first two singular values
l_sv = pd.DataFrame(svd_data[0]).iloc[:,0:2] # Take the first two columns of U
coordinates = np.multiply(l_sv,s_values.T) # Take element-wise product of l_sv and s_values

#Plot the coordinates
fig, ax = plt.subplots()
ax.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1],label = data6.index)
for i, txt in enumerate(data6.index):
    ax.annotate(txt, (coordinates.iloc[:,0][i], coordinates.iloc[:,1][i]))
ax.set_xlabel('PC1') # Label x and y axes
ax.set_ylabel('PC2')

coordinates2 = np.multiply(r_sv.T,s_values.T) # Compute coordinates
fig2, ax2 = plt.subplots()
ax2.scatter(coordinates2.iloc[:,0], coordinates2.iloc[:,1],label = data6.columns)
for i, txt in enumerate(A_hat.columns):
    ax2.annotate(txt, (coordinates2.iloc[:,0][i], coordinates2.iloc[:,1][i]))
ax2.set_xlabel('PC1') # Label x and y axes
ax2.set_ylabel('PC2')

# Overlaying the two scatterplots. Red dots are the countries and green dots are the demographic variables.
plt.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1],label = A_hat.index,color='r')
plt.scatter(coordinates2.iloc[:,0], coordinates2.iloc[:,1],label = A_hat.columns,color='g')
plt.show()


## The end of code ## 
