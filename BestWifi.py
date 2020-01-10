
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objects as go
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from math import sqrt

warnings.filterwarnings("ignore")

data = pd.read_csv("trainingData.csv")
data2 = pd.read_csv("validationData.csv")
data3 = pd.read_csv("trainingData1.csv")
data4 = pd.read_csv("validationData1.csv")
data5 = pd.concat([data3,data4]).sample(frac=1)
#data = data5.iloc[:16838 :,]
#data2 = data5.iloc[:-16838 :,]
finaldata = pd.read_csv("TEST_DATA.csv")
datadani = pd.read_csv("cascade_xgb_hyperp_knn.csv")
david = pd.read_csv("david.csv")
sergi = pd.read_csv("RandomForestandknnsearchgridCV.csv")
xenia = pd.read_csv("xenia.csv")
david = pd.read_csv("predictions_1.csv")
bestresults = []


# def visualizations():

plotdata = [go.Scatter3d(x=data['LONGITUDE'], y=data['LATITUDE'], z=data['FLOOR'], mode='markers')]
fig = go.Figure(data=plotdata)
pyo.plot(fig, filename='wifispots.html')


#def predictions(fastmode):

# Following code removes all the duplicates
data.drop_duplicates()
data2.drop_duplicates()

# Selecting features of training and testing and replacing value 100 to -105
features_training = data.filter(regex='WAP').replace(100,-105)
features_testing = data2.filter(regex='WAP').replace(100,-105)

finaldata_features_testing = finaldata.filter(regex='WAP').replace(100,-105)


# Selecting the dependent variables
depVar = data['LONGITUDE']
depVar2 = data2['LONGITUDE']
latVar = data['LATITUDE']
latVar2 = data2['LATITUDE']
floorVar = data['FLOOR']
floorVar2 = data2['FLOOR']
buildingVar = data['BUILDINGID']
buildingVar2 = data2['BUILDINGID']

depVar3 = finaldata['LONGITUDE']
latVar3 = finaldata['LATITUDE']
floorVar3 = finaldata['FLOOR']
buildingVar3 = finaldata['BUILDINGID']



# The following code is to make the loop easier
dependentVars_training_regression = [depVar, latVar]
dependentVars_testing_regression = [depVar2, latVar2]
dependentVars_final_regression = [depVar3, latVar3]
dependentVars_training_classification = [floorVar, buildingVar]
dependentVars_testing_classification = [floorVar2, buildingVar2]
dependentVars_final_classification = [floorVar3, buildingVar3]
dependentVars_name_regression = ["Longitude", "Latitude"]
dependentVars_name_classification = ["Floor", "Building"]
''' print(depVar) '''

print('----Starting the models----')
# Normalizing the data although I see little improvement in knn
preprocessing.normalize(data)
preprocessing.normalize(data2)



# Classification models

# Random Hyper-parameter Grid

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
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
print(random_grid)

# knn hyper parameters

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]

#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

# Need some tuning for the random forest?
modelRF_Regression = RandomForestRegressor()
modelRFClassification = RandomForestClassifier()
modelRFClassification_bestparam = RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf=1, max_features='auto', max_depth=80, bootstrap= False)
modelRFClassification_random = RandomizedSearchCV(estimator = modelRFClassification, param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1)


# Need some tuning for the knn?
classifier = KNeighborsClassifier(n_neighbors=5)
regressor = KNeighborsRegressor(n_neighbors=5, algorithm= 'auto')
# Need some tuning for the xgboost
xg_classifier = xgb.XGBClassifier(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                                      max_depth=5, alpha=10, n_estimators=10)


for x in range(len(dependentVars_training_classification)):
    fastmode = True
    # Classifications predictions Random Forest
    print('Predicting ' + str(dependentVars_name_classification[x]) + ' in Random forest "Classifier"...')
    if dependentVars_name_classification[x]== 'Floor':
        modelRFClassification_bestparam.fit(features_training, dependentVars_training_classification[x])
    else:
        modelRFClassification.fit(features_training, dependentVars_training_classification[x])
    if dependentVars_name_classification[x]== 'Floor':
        predictions = modelRFClassification_bestparam.predict(finaldata_features_testing)
    else:
        predictions = modelRFClassification.predict(finaldata_features_testing)
    #accuracy = accuracy_score(dependentVars_testing_classification[x], predictions)
    #print('Accuracy: %.3f' % accuracy)
    #print(' ')
    bestresults.extend([predictions])


    # Cross-validation in classification knn
    if not fastmode:
        print('Cross-validating...')
        print('Mean Cross-validated accuracy in training: %.3f' % cross_val_score(classifier, features_training, dependentVars_training_classification[x], cv=10, scoring='accuracy').mean())
        print('')

    # Classifications predictions Knn
    print('Predicting ' + str(dependentVars_name_classification[x]) + ' in knn "Classifier"...')
    classifier.fit(features_training, dependentVars_training_classification[x])
    y_pred = classifier.predict(features_testing)
    #accuracy = accuracy_score(dependentVars_testing_classification[x], y_pred)
    #print('Accuracy: %.3f' % accuracy)
    print('potat')

    # Classifications predictions in xgboost
    print('Predicting ' + str(dependentVars_name_classification[x]) + ' in xgboost "Classifier"...')
    xg_classifier.fit(features_training, dependentVars_training_classification[x])
    preds = xg_classifier.predict(features_testing)
    #accuracy = accuracy_score(dependentVars_testing_classification[x], preds)
    #print('Accuracy: %.3f' % accuracy)
    print('----')
# Regression models
for i in range(len(dependentVars_training_regression)):
    modelRF_Regression.fit(features_training, dependentVars_training_regression[i])
    ''' print(cross_val_score(modelRF, features, depVar))
    print(modelRF.score(features, depVar)) '''

    # Regressive predictions RF
    print('Predicting ' + str(dependentVars_name_regression[i]) + ' in Random forest "Regressor"...')
    predictions = modelRF_Regression.predict(features_testing)
    #predRsquared = r2_score(dependentVars_testing_regression[i], predictions)
    #rmse = sqrt(mean_squared_error(dependentVars_testing_regression[i], predictions))
    #mae = mean_absolute_error(dependentVars_testing_regression[i], predictions)
    #print('R Squared: %.3f' % predRsquared)
    #print('RMSE: %.3f' % rmse)
    #print('MAE: %.3f' % mae)
    #print(' ')
    # Regressive predictions knn
    print('Predicting ' + str(dependentVars_name_regression[i]) + ' in knn "Regressor"...')
    regressor.fit(features_training, dependentVars_training_regression[i])
    predictions = regressor.predict(finaldata_features_testing)
    #predRsquared2 = r2_score(dependentVars_testing_regression[i], predictions)
    #rmse2 = sqrt(mean_squared_error(dependentVars_testing_regression[i], predictions))
    #mae2 = mean_absolute_error(dependentVars_testing_regression[i], predictions)
    #print('R Squared: %.3f' % predRsquared2)
    #print('RMSE: %.3f' % rmse2)
    #print('MAE: %.3f' % mae2)
    print('----')
    print(i)
    bestresults.extend([predictions])


# def predictedvisualizations():
plotdata = [go.Scatter3d(x=bestresults[2], y=bestresults[3], z=bestresults[0], mode='markers')]
fig = go.Figure(data=plotdata)
pyo.plot(fig, filename='wifispotspredicted.html')


# Crafting table


# visualizations()
# predictions(fastmode=True)
# predictedvisualizations()





