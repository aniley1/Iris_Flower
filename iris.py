import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create an SVM classifier with a linear kernel
clf = SVC(kernel='linear')

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report and confusion matrix
report = classification_report(y_test, y_pred, target_names=iris.target_names)
confusion = confusion_matrix(y_test, y_pred)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Use the model for predictions
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your own measurements
predicted_species = clf.predict(new_data)
print(f"Predicted Species: {iris.target_names[predicted_species[0]]}")

# Visualize the actual vs. predicted labels
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c=y_pred, cmap='viridis')
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")
plt.title("Actual vs. Predicted Labels")
plt.colorbar(label='Predicted Species')
plt.show()
