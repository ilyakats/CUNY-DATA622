import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer

train_df = pd.read_csv('D:/CUNY/DATA622/train.csv')

train_df.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)

train_df = pd.get_dummies(train_df, drop_first=True)

train_df.count()
train_df.isnull().sum()

train_df.info()
train_df.describe()

train_df.boxplot('Age', rot=60)

train_df
train_df['Age']
train_df['Age'].mean()

# Impute
imp = Imputer(strategy='mean', axis=0)
train_df2 = imp.fit_transform(train_df)
train_df2 = pd.DataFrame(data=train_df2, columns=train_df.columns.values)
for column in train_df2:
    train_df2[column] = train_df2[column].astype(train_df[column].dtype.name)

train_df = train_df2
train_df.drop(['PassengerId'], axis=1, inplace=True)


# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

y = train_df['Survived'].values
X = train_df.drop('Survived', axis=1).values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=38)

# Create the classifier: logreg
logreg = LogisticRegression(C=0.1931, penalty='l2')

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# Setup the hyperparameter grid
c_space = np.logspace(-5, 5)
param_grid = {'C': c_space, 'penalty': ['l1','l2']}

# Instantiate a logistic regression classifier: logreg
#logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))
