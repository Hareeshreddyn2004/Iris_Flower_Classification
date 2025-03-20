#importing useful packages
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Models to train
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


#Identifying the most significant features influencing flower species classification.

# Load Iris data from CSV
iris_data = pd.read_csv("/home/reddy/iris/IRIS.csv")

# Separate features (X) and target (y)
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']

# Encode target labels (species) into numeric values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y_encoded)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display
print(feature_importance_df)
print("\nPetal_length and petal_width seems to have high and importance of 0.436\n")
# Plot
feature_importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False, color='skyblue')
plt.title('Feature Importance in Iris Classification')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.show()




# Features and target
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(solver='lbfgs', max_iter=200),
    'K-NN': KNeighborsClassifier(),
    'SVM': SVC(kernel='rbf', probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB()
}

# Train, evaluate, and cross-validate models
for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y_encoded, cv=5)
    print(f"Cross-Validation Accuracy: ", cv_scores)
    print(f"Cross-Validation Accuracy: Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
print("\nBest Models: Logistics Regression and KNN based on Cross Validated Score: 97%\n")
