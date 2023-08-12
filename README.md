# BreastCancer-DetectionModel

### Description
This repository is dedicated to the exploration and prediction of breast cancer diagnoses using Machine Learning techniques. While the primary goal is to build and validate predictive models, there is also a strong emphasis on exploratory data analysis (EDA) to understand the underlying patterns in the data.

### Libraries Used
- Pandas: For data manipulation and preprocessing.
- NumPy: For numerical computations and support in handling arrays.
- Matplotlib & Seaborn: For data visualization, essential in EDA and result interpretation.
- Scikit-learn: For building, training, and validating machine learning models.

### Key Features of the Notebook
- Data Loading and Initial Exploration: A peek into the dataset (data.csv) to understand its structure and contents.
  
- Data Cleaning & Preprocessing:
  - Removal of redundant columns to streamline the dataset.
  - Transformation of categorical data, like the 'diagnosis' column, into a numerical format suitable for machine learning.
  - Selection of relevant features for prediction.
  - Splitting the dataset into training and testing sets to validate model performance.
 
- Machine Learning:
  - Scikit-learn Models:
      - Random Forest: An ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) of the individual trees.
      - K-Nearest Neighbors (KNN): A non-parametric method used for classification and regression.
      - Support Vector Machine (SVM): A set of supervised learning methods used for classification, regression, and outliers detection.
      - MLP (Multi-layer Perceptron): A subset of linear classifiers that utilizes multiple layers of a perceptron (or neuron) for data transformation.
  
  - Custom Implementations:
      - Random Forest: A custom implementation of the Random Forest algorithm from scratch.
      - Support Vector Machine (SVM): A custom linear SVM model implemented from scratch.
- Model Evaluation: For each model, a variety of metrics, including precision, recall, and accuracy, are calculated to gauge their performance.
Visualization of results using confusion matrices to provide a more intuitive understanding of the model's strengths and weaknesses.
  
### How to Use
- Clone the repository.
- Install the necessary Python libraries.
- Launch main.ipynb in a Jupyter environment.
- Execute the cells in sequence to experience the full data science and machine learning workflow.
