import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
import matplotlib.pyplot as plt


ks = pandas.read_excel("/Users/Christine/Documents/INSY 652/Kickstarter.xlsx")
ks = ks.drop(['project_id','name','pledged','currency','deadline','state_changed_at', 'created_at',
              'launched_at','name_len','name_len_clean','blurb_len','blurb_len_clean','state_changed_at_weekday',
             'created_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr',
              'state_changed_at_hr','created_at_month','created_at_day',
              'created_at_yr','created_at_hr','launch_to_state_change_days'], axis=1)
ks.shape #(18568,22)
ks.isna().sum()
ks['category'].value_counts()
ks = ks.dropna(axis=0, how='any') # dropping 9.07% of total data
# final shape: (16884, 22)

ks_dummied = ks.copy()
for elem in ks_dummied['country'].unique():
    ks_dummied['country_'+str(elem)] = ks_dummied['country'] == elem
for elem in ks_dummied['category'].unique():
    ks_dummied['category_'+str(elem)] = ks_dummied['category'] == elem
for elem in ks_dummied['deadline_weekday'].unique():
    ks_dummied['deadline_weekday_'+str(elem)] = ks_dummied['deadline_weekday'] == elem
for elem in ks_dummied['launched_at_weekday'].unique():
    ks_dummied['launched_at_weekday_'+str(elem)] = ks_dummied['launched_at_weekday'] == elem

ks_dummied = ks_dummied.drop(["country","category","deadline_weekday","launched_at_weekday"], axis=1)



###################################### Question 1 ######################################
# Provide summary statistics for the variables that are interesting/relevant to your analyses.
og_description = ks.describe(include='all')
print(og_description.transpose())
og_description.transpose().to_excel("og_description.xlsx")


###################################### Question 2 ######################################
# Develop a regression model (i.e., a supervised-learning model where the target variable is a continuous variable)
# to predict the value of the variable “usd_pledged.” After you obtain the final model,
# explain the model and justify the predictors you include/exclude.

#### Preparing data for regression
ks_regression = ks_dummied.copy()
for elem in ks_regression['state'].unique():
    ks_regression['state_'+str(elem)] = ks_regression['state'] == elem
ks_regression = ks_regression.drop(["state"], axis=1)

#### Feature selection for regression tasks
X = ks_regression.drop(["usd_pledged"], axis=1)
y = ks_regression["usd_pledged"]
lasso_scaler = StandardScaler()
X_std = lasso_scaler.fit_transform(X)

model = Lasso(alpha=0.01, positive=True) # alpha here is the penalty term ?????
model.fit(X_std,y)
temp = pandas.DataFrame(list(zip(X.columns,model.coef_)), columns=['predictor','coefficient'])
temp = temp.sort_values(by='coefficient',ascending=False)
# backers_count,//4D//category_Hardware//category_Web,category_Gadgets,staff_pick,category_Wearables,category_Software


################### Linear Regression
from sklearn.linear_model import LinearRegression
X = ks_regression[["backers_count","category_Hardware","category_Web","category_Gadgets","staff_pick","category_Wearables","category_Software"]]
y = ks_regression["usd_pledged"]
lir_scaler = StandardScaler()
X_std = lir_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

lir = LinearRegression()
lirmodel = lir.fit(X_train, y_train)
y_test_pred = lirmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
#3500362781.9228187

################### KNN regressor
from sklearn.neighbors import KNeighborsRegressor

X = ks_regression[["backers_count","category_Hardware","category_Web","category_Gadgets","staff_pick","category_Wearables","category_Software"]]
y = ks_regression["usd_pledged"]
knnr_scaler = StandardScaler()
X_std = knnr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

for i in range(2,20):
    knnr = KNeighborsRegressor(n_neighbors=i)
    knnrmodel = knnr.fit(X_train, y_train)
    y_test_pred = knnrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# The best number for n_neighbors is 6

knnr = KNeighborsRegressor(n_neighbors=6)
knnrmodel = knnr.fit(X_train, y_train)
y_test_pred = knnrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# Average mse: 2495078860.480824 (iteration:10)


################### Random forest regressor
from sklearn.ensemble import RandomForestRegressor
X = ks_regression[["backers_count","category_Hardware","category_Web","category_Gadgets","staff_pick","category_Wearables","category_Software"]]
y = ks_regression["usd_pledged"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

for i in range(2,8):
    rfr = RandomForestRegressor(max_features=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_test_pred = rfrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# Best max_features = 2

for i in range(2,20):
    rfr = RandomForestRegressor(max_features=2, max_depth=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_test_pred = rfrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# Best max_depth = 6

for i in range(2,10):
    rfr = RandomForestRegressor(max_features=2, max_depth=6, min_samples_split=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_test_pred = rfrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# Best min_samples_split = 3

rfr = RandomForestRegressor(max_features=2, max_depth=6, min_samples_split = 3)
rfrmodel = rfr.fit(X_train, y_train)
y_test_pred = rfrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# Average mse: 2774198879.7817373 (iteration:10)


################### SVM regressor
from sklearn.svm import SVR
X = ks_regression[["backers_count","category_Hardware","category_Web","category_Gadgets","staff_pick","category_Wearables","category_Software"]]
y = ks_regression["usd_pledged"]
svmr_scaler = StandardScaler()
X_std = svmr_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

svr = SVR(kernel="linear", epsilon=0.1)
svrmodel = svr.fit(X_train, y_train)
y_test_pred = svrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# 9372579465.64207
# Some for-loops were used to find the best parameters, but there wasn't any dramatic change to mse


################### ANN regressor
from sklearn.neural_network import MLPRegressor
X = ks_regression[["backers_count","category_Hardware","category_Web","category_Gadgets","staff_pick","category_Wearables","category_Software"]]
y = ks_regression["usd_pledged"]
mlpr_scaler = StandardScaler()
X_std = mlpr_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

for i in range(2,11):
    annr = MLPRegressor(hidden_layer_sizes=(i), max_iter=1000)
    annrmodel = annr.fit(X_train, y_train)
    y_test_pred = annrmodel.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)

annr = MLPRegressor(hidden_layer_sizes=(9), max_iter=1000)
annrmodel = annr.fit(X_train, y_train)
y_test_pred = annrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# 7504426858.044228
# Some for-loops were used to find the best parameters, but there wasn't any dramatic change to mse

#### As a result, a model made with KNN Regressor has been chosen. ####




###################################### Question 3 ######################################
# Develop a classification model (i.e., a supervised-learning model where the target variable is a categorical variable)
# to predict whether the variable “state” will take the value “successful” or “failure.”
# After you obtain the final model, explain the model and justify the predictors you include/exclude.

#### Preparing data for classification
ks_classification = ks_dummied.loc[(ks_dummied["state"]=="successful")|(ks_dummied["state"]=="failed")]

#### Feature selection for regression tasks
### Feature selection with Random Forest SFM
from sklearn.ensemble import RandomForestClassifier
# "Spotlight" had to be dropped as it was causing data leakage (had the exact same pattern as "state"
# "Goal" and "usd_pledged" were dropped in the same manner; if "usd_pledged" was higher than "goal", the state would be automatically successful
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]

randomforest = RandomForestClassifier()
rfmodel = randomforest.fit(X, y)

sfm = SelectFromModel(rfmodel, threshold=0.039) # only get the variables above the threshold
sfm.fit(X,y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
# staff_pick, backers_count, create_to_launch_days, and maybe launch_to_deadline_days


################### Logistic Regression
from sklearn.linear_model import LogisticRegression
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]
logr_scaler = StandardScaler()
X_std = logr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)

logr = LogisticRegression()
logrmodel = logr.fit(X_train, y_train)

y_test_pred = logrmodel.predict(X_test)
accuracy_score(y_test,y_test_pred)
# Average accuracy: 84.38 (iteration:10)


################### kNN
from sklearn.neighbors import KNeighborsClassifier
X = ks_classification[["staff_pick","backers_count","create_to_launch_days"]]
y = ks_classification["state"]
knn_scaler = StandardScaler()
X_std = knn_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)

for i in range(24,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knnmodel = knn.fit(X_train, y_train)
    y_test_pred = knnmodel.predict(X_test)
    print(accuracy_score(y_test,y_test_pred))
# best n_neighbors is 24 and accuracy is 84.17%

knn = KNeighborsClassifier(n_neighbors=24)
knnmodel = knn.fit(X_train, y_train)
y_test_pred = knnmodel.predict(X_test)

accuracy_score(y_test,y_test_pred)
# Average accuracy: 84.17 (iteration:10)


################### Random Forest
from sklearn.ensemble import RandomForestClassifier
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]

randomforest = RandomForestClassifier()
rfmodel = randomforest.fit(X, y)

rfe = RFE(rfmodel, 4)
model = rfe.fit(X,y)
pandas.DataFrame(list(zip(X.columns,model.ranking_)), columns=['predictor','ranking'])
# backers_count, create_to_launch_days, deadline_day, launched_at_day

# Real model building
# Features were selected from the ones that were selected earlier with SFM and the one with RFE right before
# Tried every possible combination of the features, this is the best
X = ks_classification[["staff_pick","backers_count","create_to_launch_days", "launch_to_deadline_days", "launched_at_day"]]
y = ks_classification["state"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5)

for i in range(2,6):
    randomforest = RandomForestClassifier(max_features=i, random_state=5)
    rfmodel = randomforest.fit(X_train, y_train)
    y_test_pred = rfmodel.predict(X_test)
    print(accuracy_score(y_test, y_test_pred))
# best max_features is 4 and accuracy is 84.15

for i in range(2,10):
    randomforest = RandomForestClassifier(max_features=4, max_depth=i, random_state=5)
    rfmodel = randomforest.fit(X_train, y_train)
    y_test_pred = rfmodel.predict(X_test)
    print(accuracy_score(y_test, y_test_pred))
# best max_features is 6 and accuracy is 85.58

randomforest = RandomForestClassifier(max_features=4,max_depth=6)
rfmodel = randomforest.fit(X_train, y_train)
y_test_pred = rfmodel.predict(X_test)

accuracy_score(y_test, y_test_pred)
# Average accuracy: 85.57 (iteration:10)


################### SVM
from sklearn.svm import SVC
X = ks_classification[["staff_pick","backers_count","create_to_launch_days","launch_to_deadline_days"]]
y = ks_classification["state"]
svc_scaler = StandardScaler()
X_std = svc_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)

for i in range(1,21):
    svc = SVC(kernel="rbf", random_state=5, C=i)
    svcmodel = svc.fit(X_train, y_train)
    y_test_pred = svcmodel.predict(X_test)
    print(accuracy_score(y_test, y_test_pred))
# best C is 17 and accuracy is 82.90

svc = SVC(kernel="rbf", C=17)
svcmodel = svc.fit(X_train,y_train)

y_test_pred = svcmodel.predict(X_test)
accuracy_score(y_test, y_test_pred)
# Average accuracy score: 82.20 (iteration:10)


################### ANN
from sklearn.neural_network import MLPClassifier
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]
mlp_scaler = StandardScaler()
X_std = mlp_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)

for i in range(1,16):
    mlp = MLPClassifier(hidden_layer_sizes=(i), max_iter=1000, random_state=5)
    scores = cross_val_score(estimator=mlp, X=X_train, y=y_train, cv=5)
    print(i,":",numpy.average(scores))
# best hidden_layer_sizes is (4)

for i in range(1,11):
    mlp = MLPClassifier(hidden_layer_sizes=(4,i), max_iter=1000, random_state=5)
    scores = cross_val_score(estimator=mlp, X=X_train, y=y_train, cv=5)
    print(i,":",numpy.average(scores))
# best hidden_layer_sizes is (4,6)

mlp = MLPClassifier(hidden_layer_sizes=(4,6), max_iter=1000)
annmodel = mlp.fit(X_train, y_train)
y_test_pred = annmodel.predict(X_test)
accuracy_score(y_test, y_test_pred)
# Average accuracy: 87.47 (iteration:10)





###################################### Question 4 ######################################
# Develop a cluster model (i.e., an unsupervised-learning model which can group observations together)
# to group projects together.

ks_clustering = ks_dummied.copy()
for elem in ks_clustering['state'].unique():
    ks_clustering['state_'+str(elem)] = ks_clustering['state'] == elem
ks_clustering = ks_clustering.drop(["state"],axis=1)

# Selecting features based on state,
ks_clustering = ks_clustering[["staff_pick","backers_count",
                               "create_to_launch_days","launch_to_deadline_days","state_failed",
                               "state_successful","state_canceled","state_live","state_suspended"]]
X = ks_clustering.copy()
kmeans_scaler = StandardScaler()
X_std = kmeans_scaler.fit_transform(X)

from sklearn.cluster import KMeans
withinss = []
for i in range(2,20):
    kmeans = KMeans(n_clusters=i)
    kmeansmodel = kmeans.fit(X_std)
    withinss.append(kmeansmodel.inertia_)

pyplot.plot([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], withinss)
plt.show()
# 9 to 10 is the best number for the number of clusters

kmeans = KMeans(n_clusters=10)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

# Observing the characteristics in each cluster
X_with_clusters = pandas.concat([X.reset_index(drop=True),pandas.DataFrame(labels, columns=["labels"])], axis=1)

cluster0 = X_with_clusters.loc[X_with_clusters["labels"]==0]
cluster1 = X_with_clusters.loc[X_with_clusters["labels"]==1]
cluster2 = X_with_clusters.loc[X_with_clusters["labels"]==2]
cluster3 = X_with_clusters.loc[X_with_clusters["labels"]==3]
cluster4 = X_with_clusters.loc[X_with_clusters["labels"]==4]
cluster5 = X_with_clusters.loc[X_with_clusters["labels"]==5]
cluster6 = X_with_clusters.loc[X_with_clusters["labels"]==6]
cluster7 = X_with_clusters.loc[X_with_clusters["labels"]==7]
cluster8 = X_with_clusters.loc[X_with_clusters["labels"]==8]
cluster9 = X_with_clusters.loc[X_with_clusters["labels"]==9]

cluster0 = cluster0.describe(include='all').transpose()
cluster1 = cluster1.describe(include='all').transpose()
cluster2 = cluster2.describe(include='all').transpose()
cluster3 = cluster3.describe(include='all').transpose()
cluster4 = cluster4.describe(include='all').transpose()
cluster5 = cluster5.describe(include='all').transpose()
cluster6 = cluster6.describe(include='all').transpose()
cluster7 = cluster7.describe(include='all').transpose()
cluster8 = cluster8.describe(include='all').transpose()
cluster9 = cluster9.describe(include='all').transpose()

from sklearn.metrics import silhouette_score
silhouette_score(X_std, labels)
# Average score: 0.61 (iteration:10)

from sklearn.metrics import calinski_harabaz_score
from scipy.stats import f
score = calinski_harabaz_score(X_std, labels)
# 7935.7059706369155

df1 = 9 # df1=k-1
df2 = 16874 # df2=n-k
pvalue = 1-f.cdf(score, df1, df2)
pvalue
# 1.1102230246251565e-16






###################################### Validation ######################################


ks_prof = pandas.read_excel("/Users/Christine/Documents/INSY 652/Kickstarter.xlsx") # Please change the path
ks_prof = ks_prof.drop(['project_id','name','pledged','currency','deadline','state_changed_at', 'created_at',
              'launched_at','name_len','name_len_clean','blurb_len','blurb_len_clean','state_changed_at_weekday',
             'created_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr',
              'state_changed_at_hr','created_at_month','created_at_day',
              'created_at_yr','created_at_hr','launch_to_state_change_days'], axis=1)
ks_prof = ks_prof.dropna(axis=0, how='any') # dropping 9.07% of total data


ks_prof_dummied = ks_prof.copy()
for elem in ks_prof_dummied['country'].unique():
    ks_prof_dummied['country_'+str(elem)] = ks_prof_dummied['country'] == elem
for elem in ks_prof_dummied['category'].unique():
    ks_prof_dummied['category_'+str(elem)] = ks_prof_dummied['category'] == elem
for elem in ks_prof_dummied['deadline_weekday'].unique():
    ks_prof_dummied['deadline_weekday_'+str(elem)] = ks_prof_dummied['deadline_weekday'] == elem
for elem in ks_prof_dummied['launched_at_weekday'].unique():
    ks_prof_dummied['launched_at_weekday_'+str(elem)] = ks_prof_dummied['launched_at_weekday'] == elem

ks_prof_dummied = ks_prof_dummied.drop(["country","category","deadline_weekday","launched_at_weekday"], axis=1)


###### Regression
ks_prof_regression = ks_prof_dummied.copy()
for elem in ks_prof_regression['state'].unique():
    ks_prof_regression['state_'+str(elem)] = ks_prof_regression['state'] == elem
ks_prof_regression = ks_prof_regression.drop(["state"], axis=1)

X = ks_prof_regression[["backers_count","category_Hardware","category_Web","category_Gadgets","staff_pick","category_Wearables","category_Software"]]
y = ks_prof_regression["usd_pledged"]
knnr_scaler = StandardScaler()
X_std = knnr_scaler.fit_transform(X)

y_pred = knnrmodel.predict(X_std)
mse = mean_squared_error(y, y_pred)
print(mse)


###### Classification
ks_prof_classification = ks_prof_dummied.loc[(ks_prof_dummied["state"]=="successful")|(ks_prof_dummied["state"]=="failed")]
X = ks_prof_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_prof_classification["state"]
mlp_scaler = StandardScaler()
X_std = mlp_scaler.fit_transform(X)

y_pred = annmodel.predict(X_std)
accuracy_score(y, y_pred)
print(accuracy_score(y, y_pred))
