


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

'''

                        Data Preparation

'''

# Import the data from directory using pandas
data = pd.read_csv("H:/Uni/Year 2/ML Coursework/EcoliData.csv", header = 0)
# Encode the string predictor to produce categorical 
labelencoder = LabelEncoder()
data.loc[:,'localization site'] = labelencoder.fit_transform(data.loc[:,'localization site'])

x = data.loc[:, 'mcg':'alm2']

y = data.loc[:,'localization site']

X_train, X_test, y_train, y_test = train_test_split(x,y,random_state = 0)



'''
                        Decision Tree Method

'''



# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 

def accuracy(matrix):
    return matrix.trace() / (matrix.sum())

# test depth vs accuracy for simply decision tree
c = []
d = []
for i in range(1,10):
    c.append(i)
    dtree_model = DecisionTreeClassifier(max_depth = i).fit(X_train, y_train) 
    dtree_predictions = dtree_model.predict(X_test)   
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, dtree_predictions) 
    d.append(accuracy(cm))
    
plt.plot(c,d)
plt.show()    
    

dtree_model = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions) 

print('The accuracy of the optimised decision tree method is ' + str(accuracy(cm)))



'''

                        Random Forest Method

'''
from sklearn.ensemble import RandomForestClassifier

# Set up the random forest classifier
a = []
b = []
def randomforest_param_plot():
    for i in range(10,180):
        a.append(i)
        rfor_model = RandomForestClassifier(n_jobs = 4, random_state = 0, n_estimators = i).fit(X_train, y_train)
        rfor_pred = rfor_model.predict(X_test)
        b.append(accuracy(confusion_matrix(y_test, rfor_pred)))

        plt.plot(a, b)
        plt.show()   
 




clf = RandomForestClassifier(n_jobs = 2, random_state = 0, max_depth = 5)

rfor_model = clf.fit(X_train, y_train)
rfor_pred = rfor_model.predict(X_test)

print('The accuracy of the optimised random forest method is ' + str(accuracy(confusion_matrix(y_test, rfor_pred))))


    

feature_importances = pd.DataFrame(clf.feature_importances_, index = X_train.columns, columns = ['importance'] ).sort_values('importance', ascending = False)




# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print('\n\n')
print(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}



rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


rf_random.fit(X_train, y_train)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    acc = accuracy(confusion_matrix(predictions, test_labels))
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(acc))
    return accuracy
    
    
evaluate(clf, X_test, y_test)








