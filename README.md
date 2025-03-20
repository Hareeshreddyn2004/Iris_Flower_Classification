**Iris Species Classification – ML Model Comparison**

**Project Objective:**

The goal of this project is to: Classify iris flower species (Setosa, Versicolor, Virginica) based on four features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- Train and compare multiple machine learning models on the Iris dataset.
- Evaluate models using accuracy, classification reports, confusion matrices, and cross-validation scores.
- Identify important features influencing classification using Random Forest feature importance.


**Models Implemented**
- Logistic Regression
- K-Nearest Neighbors (K-NN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes Classifier


**Steps to Run the Project**

1. **Clone or Download** the project files.

2. Ensure **Python 3.12** is installed.

3. Install required libraries:
   
   pip install pandas scikit-learn matplotlib seaborn

4. Update the path to **`IRIS.csv`** file in main.py

5. Run the project script:

   python main.py
  
  

**Project Structure**
```
iris_classification_project/
    ├── main.py
    ├── main.ipnynb
    ├── README.md
    ├── main.html
    ├── LICENSE
```

**Outputs Generated**
- **Model Accuracy** and **Classification Reports**
- **Confusion Matrices** for visualizing model predictions
- **Cross-Validation Scores** (Mean and Standard Deviation)
- **Feature Importance Plot** (Random Forest)


**Code Structure Highlights**
- Clean and modular code with comments.
- Uses `LabelEncoder` for target Column encoding.
- Implements **train-test split** and **cross-validation** for robust evaluation.
- Visualizations using **Seaborn** and **Matplotlib**.


**Example Results**
All models achieved **100% test accuracy** on the dataset. Cross-validation scores revealed minor differences:

| Model               | CV Accuracy (Mean ± Std) |
|---------------------|--------------------------|
| Logistic Regression | 0.9733 ± 0.0249          |
| K-NN                | 0.9733 ± 0.0249          |
| SVM                 | 0.9667 ± 0.0211          |
| Decision Tree       | 0.9600 ± 0.0327          |
| Random Forest       | 0.9600 ± 0.0249          |
| Naive Bayes         | 0.9533 ± 0.0267          |


**Conclusion**
- All models performed excellently.
- Random Forest helped **identify Petal Length & Petal Width** as key features.
- Logistic Regression and K-NN showed slightly better cross-validation consistency.
