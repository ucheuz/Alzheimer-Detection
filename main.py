from google.colab import files
files.upload()
-----
import pandas as pd

# Load csv file
df = pd.read_csv("DARWIN_dataset.csv")
data = df.to_numpy()

# Print number of samples
print('Number of samples: ',data.shape[0])
------
from sklearn.preprocessing import StandardScaler
import numpy as np

X = StandardScaler().fit_transform(data[:,:450])
y = data[:,450]

print('Features dim: ', X.shape)
print('Labels dim: ', y.shape)
print('We have {} samples and {} features.'.format(X.shape[0],X.shape[1]))
------
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot reduced data
plt.figure(figsize=(8,6))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.title('PCA of DARWIN Dataset', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.colorbar(label='Labels')
plt.show()
------
# Perform Lapplacian Eigenmaps
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt

model = SpectralEmbedding(n_components=2, n_neighbors=10)

# Apply the model to the data to reduce to 2 dimensions
X_lle = model.fit_transform(X)

# Plot reduced data
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xticks([])
ax.set_yticks([])

scatter_plot = ax.scatter(X_lle[:, 0], X_lle[:, 1], c=y, cmap='viridis', alpha=0.5)

ax.set_title('Laplacian Eigenmaps of DARWIN Dataset', fontsize=16)
ax.set_xlabel('Component 1', fontsize=12)
ax.set_ylabel('Component 2', fontsize=12)

plt.colorbar(scatter_plot, label='Labels')

# Show plot
plt.show()
------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Fit and evaluate a random forest classifier
clf = RandomForestClassifier(max_depth=5,n_estimators=50, random_state=0)
clf.fit(X, y)

# Perform 5-fold cross-validation
accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

# Print the cross-validated accuracy
print("Cross-validated accuracy: {:.2f}%".format(accuracy.mean() * 100))
------
# Identify and print the two most important features
importances = clf.feature_importances_

# Get indices of the top 2 most important features
indices = np.argsort(importances)[::-1]

# Print feature names and importance values
print("Top 2 most important features:")
for i in range(2):  # Loop through the top 2 features
    print(f"{i + 1}. Feature: {df.columns.values[indices[i]]} (Importance: {importances[indices[i]]:.4f})")
------
import numpy as np
from sklearn.decomposition import PCA

# Transform features using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

clf.fit(X_pca,y)

# Define the 2D feature range
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

# Generate feature space
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
X_mesh = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

# Predict and plot labels for the features space
grid = np.c_[xx.ravel(), yy.ravel()]
grid_original = pca.inverse_transform(grid)
Z = clf.predict(pca.transform(grid_original))
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot reduced data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
plt.title("Random Forest Classification in 2D Feature Space", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.colorbar(label="Labels")
plt.show()
------
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=2)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)


selected_features = np.where(rfe.support_)[0]
print("Selected feature indices:", selected_features)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_rfe, y_train)

y_pred = clf.predict(X_test_rfe)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with {len(selected_features)} features: {accuracy:.2f}")

if X_train_rfe.shape[1] == 2:
    x_min, x_max = X_train_rfe[:, 0].min() - 1, X_train_rfe[:, 0].max() + 1
    y_min, y_max = X_train_rfe[:, 1].min() - 1, X_train_rfe[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_train_rfe[:, 0], X_train_rfe[:, 1], c=y_train, cmap='viridis', edgecolor='k')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Labels')
    plt.show()

print("Selected features:", [df.columns.values[i] for i in selected_features])
------
import torch
import torch.nn as nn

# Define network architecture
class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(450, 150)  # First linear layer (450 input features, 150 units)
        self.layer2 = nn.Linear(150, 50)   # Second linear layer (150 input features, 50 units)
        self.layer3 = nn.Linear(50, 1)     # Third linear layer (50 input features, 1 output)

        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the network
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

# Loss Function
loss_function = nn.BCELoss()

model = NNClassifier()
------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to Pytorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train.reshape(-1,1)).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test.reshape(-1,1)).float()

# Create the model
net = NNClassifier()

# Create the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# Loss function
loss_fucntion = nn.BCELoss()  # Binary Cross Entropy Loss

# Training
epochs = 100
for epoch in range(epochs):
  net.train()
  # Zero the gradients before each step
  optimizer.zero_grad()

  # Forward pass
  outputs = net(X_train)

  # Calculate the loss
  loss = loss_function(outputs.squeeze(), y_train.squeeze())

  # Backpropagation
  loss.backward()

  # Update the weights
  optimizer.step()

  # Print the loss every 10 epochs
  if (epoch + 1) % 10 == 0:
      print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# accuracy on test set
net.eval()
with torch.no_grad():
    y_pred = net(X_test).squeeze()
    y_pred = (y_pred >= 0.5).float()

acc = accuracy_score(y_test, y_pred)
print('Test accuracy: ', np.round(acc, 2))

pca = PCA(n_components=2)
features_PCA = pca.fit_transform(X)

x_min, x_max = features_PCA[:, 0].min() - 0.1, features_PCA[:, 0].max() + 0.1
y_min, y_max = features_PCA[:, 1].min() - 0.1, features_PCA[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.stack([xx.flatten(), yy.flatten()]).T


Z = net(torch.from_numpy(pca.inverse_transform(grid)).float()).detach().numpy()
Z = (Z >= 0.5).astype(int)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot
plt.scatter(features_PCA[:, 0], features_PCA[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)

plt.title("Neural Network Classification in 2D Feature Space", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.colorbar(label="Labels")
plt.show()
