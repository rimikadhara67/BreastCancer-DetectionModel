import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')
df = df.drop(columns='id', axis=1)
df = df.drop(columns='Unnamed: 32', axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

mean_features = list(df.columns[1:11])
se_features = list(df.columns[11:21])
worst_features = list(df.columns[21:31])
mean_features.append('diagnosis')
se_features.append('diagnosis')
worst_features.append('diagnosis')

# FEATURE EXTRACTION
corr_mean = df[mean_features].corr()
corr_se = df[se_features].corr()
corr_worst = df[worst_features].corr()
corr_mean.to_csv('mean-features.csv', sep='\t')
corr_se.to_csv('se-features.csv', sep='\t')
corr_worst.to_csv('worst-features.csv', sep='\t')
prediction_vars = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 
                'concavity_mean', 'concave points_mean', 
                'radius_se', 'area_se', 
                'radius_worst', 'perimeter_worst', 'compactness_worst']

# TRAINING 
train,test = train_test_split(df, test_size=0.15, random_state=1)
train_x = train[prediction_vars]
train_y = train['diagnosis']
test_x = test[prediction_vars]
test_y = test['diagnosis']

# MODEL 1: RANDOM FOREST 
model_best = RandomForestClassifier()
model_best.fit(train_x, train_y)
predictions = model_best.predict(test_x)

# ANALYSIS
print("Model 1: Random Forest Classifier:")
# print(confusion_matrix(test_y, predictions))
precision = precision_score(test_y, predictions)
print(f"----Precision: {precision}")
recall = recall_score(test_y, predictions)
print(f"----Recall Score: {recall}")
accuracy = accuracy_score(test_y, predictions)
print(f"----Accuracy Score: {accuracy}")

# MODEL 2: K-NEAREST NEIGHBORS
model = KNeighborsClassifier()
model.fit(train_x, train_y)
predictions = model.predict(test_x)

# ANALYSIS
print("Model 2: K-Nearest Neighbors:")
# print(confusion_matrix(test_y, predictions))
precision = precision_score(test_y, predictions)
print(f"----Precision: {precision}")
recall = recall_score(test_y, predictions)
print(f"----Recall Score: {recall}")
accuracy = accuracy_score(test_y, predictions)
print(f"----Accuracy Score: {accuracy}")

# MODEL 3: Support Vector Machine
model = SVC()
model.fit(train_x, train_y)
predictions = model.predict(test_x)

# ANALYSIS
print("Model 3: Support Vector Machine:")
# print(confusion_matrix(test_y, predictions))
precision = precision_score(test_y, predictions)
print(f"----Precision: {precision}")
recall = recall_score(test_y, predictions)
print(f"----Recall Score: {recall}")
accuracy = accuracy_score(test_y, predictions)
print(f"----Accuracy Score: {accuracy}")

# MODEL 3: MLP
model = MLPClassifier()
model.fit(train_x, train_y)
predictions = model.predict(test_x)

# ANALYSIS
print("Model 4: MLP:")
# print(confusion_matrix(test_y, predictions))
precision = precision_score(test_y, predictions)
print(f"----Precision: {precision}")
recall = recall_score(test_y, predictions)
print(f"----Recall Score: {recall}")
accuracy = accuracy_score(test_y, predictions)
print(f"----Accuracy Score: {accuracy}")

# Because Random Forest Classifier was our best model, 
# we will be doing Hyper-parameter Optimizations

from sklearn.model_selection import GridSearchCV
parameters = {'max_depth' :(1,2,3,4), 'n_estimators':(10,50,100,500)}
best_model = GridSearchCV(model_best, parameters)
best_model.fit(train_x, train_y)
print(best_model.best_params_)