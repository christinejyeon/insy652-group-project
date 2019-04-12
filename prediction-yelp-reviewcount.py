import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

yelp_biz_hrs = pandas.read_csv("/Users/Christine/Downloads/yelp_business_hours.csv")
yelp_biz = pandas.read_csv("/Users/Christine/Downloads/yelp_business.csv")
yelp_biz.columns
yelp_biz = yelp_biz.drop(["name", "neighborhood", "address", "city", "postal_code"], axis=1)

temp = yelp_biz_hrs.copy()
temp = temp.drop(["business_id"], axis=1)
temp["Flag_notaddedyet"] = temp.apply(lambda x: min(x) == max(x), axis=1)
temp = temp.drop(["monday","tuesday","wednesday","thursday","friday","saturday","sunday"], axis=1)

yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),temp], axis=1)

def totalhrs(df,colnum):
    hrs = df.iloc[:,colnum].str.split("-", expand=True)
    hrs.columns = ["start", "end"]
    hrs.loc[hrs["start"].str.contains(":30"), "start"] = hrs.loc[hrs["start"].str.contains(":30"), "start"].str.split(":").str[0].astype(float) + 0.5
    hrs.loc[(hrs["start"].str.contains(":30") == False) & (hrs["start"] != "None"), "start"] = hrs.loc[(hrs["start"].str.contains(":30") == False) & (hrs["start"] != "None"), "start"].str.split(":").str[0].astype(float)
    hrs.loc[(hrs["start"].str.contains(":30") == False) & (hrs["start"] == "None"), "start"] = 0
    hrs.loc[hrs["end"].str.contains(":30") == True, "end"] = hrs.loc[hrs["end"].str.contains(":30") == True, "end"].str.split(":").str[0].astype(float) + 0.5
    hrs.loc[(hrs["end"].str.contains(":30") == False) & (hrs["end"] != "None"), "end"] = hrs.loc[(hrs["end"].str.contains(":30") == False) & (hrs["end"] != "None"), "end"].str.split(":").str[0].astype(float)
    hrs.loc[(hrs["end"].str.contains(":30") == False) & (hrs["end"] == "None"), "end"] = 0
    hrs.loc[hrs["start"] != "None", "total"] = hrs["end"] - hrs["start"]
    hrs.loc[hrs["total"] <= 0, "total"] = hrs["total"] + 24
    return hrs["total"]

yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,1)], axis=1)
yelp_biz_hrs.rename(columns={"total":"monday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,2)], axis=1)
yelp_biz_hrs.rename(columns={"total":"tuesday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,3)], axis=1)
yelp_biz_hrs.rename(columns={"total":"wednesday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,4)], axis=1)
yelp_biz_hrs.rename(columns={"total":"thursday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,5)], axis=1)
yelp_biz_hrs.rename(columns={"total":"friday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,6)], axis=1)
yelp_biz_hrs.rename(columns={"total":"saturday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,7)], axis=1)
yelp_biz_hrs.rename(columns={"total":"sunday_total"}, inplace=True)

yelp_biz_hrs = yelp_biz_hrs.drop(["monday","tuesday","wednesday","thursday","friday","saturday","sunday"], axis=1)
yelp_biz_hrs["week_total"] = yelp_biz_hrs.fillna(0)["monday_total"]+yelp_biz_hrs.fillna(0)["tuesday_total"]+yelp_biz_hrs.fillna(0)["wednesday_total"]+yelp_biz_hrs.fillna(0)["thursday_total"]+yelp_biz_hrs.fillna(0)["friday_total"]+yelp_biz_hrs.fillna(0)["saturday_total"]+yelp_biz_hrs.fillna(0)["sunday_total"]

yelp_biz.shape
yelp_biz_hrs.shape

og_data = pandas.merge(yelp_biz,yelp_biz_hrs)
og_data = og_data.drop(["business_id"], axis=1)
#og_data.to_csv("og_data.csv")
#og_data = pandas.read_csv("og_data.csv")
og_data = og_data.drop(["Unnamed: 0"], axis=1)

og_data["num_categories"] = og_data["categories"].str.split(";").apply(len)
og_data = og_data.drop(["categories"], axis=1)


###################################### Regression ######################################
og_regression = og_data.copy()
og_regression = og_regression.fillna(0)
og_regression = og_regression.drop(["state"],axis=1)

#### Feature selection for regression tasks
X = og_regression.drop(["review_count"], axis=1)
y = og_regression["review_count"]
lasso_scaler = StandardScaler()
X_std = lasso_scaler.fit_transform(X)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01, positive=True) # alpha here is the penalty term ?????
model.fit(X_std,y)
temp = pandas.DataFrame(list(zip(X.columns,model.coef_)), columns=['predictor','coefficient'])
temp = temp.sort_values(by='coefficient',ascending=False)
# sunday_total 10.10904
# saturday_total 5.74567
# num_categories 5.45640
# stars 2.31846
# is_open 1.79111

################### Linear Regression
from sklearn.linear_model import LinearRegression
X = og_regression.drop(["review_count"], axis=1)
y = og_regression["review_count"]
lir_scaler = StandardScaler()
X_std = lir_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=5)

lir = LinearRegression()
lirmodel = lir.fit(X_train, y_train)
y_test_pred = lirmodel.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(mse)

#8893.89329967535
#9442.034046201485


################### KNN regressor
from sklearn.neighbors import KNeighborsRegressor
X = og_regression.drop(["review_count"], axis=1)
y = og_regression["review_count"]
knnr_scaler = StandardScaler()
X_std = knnr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=5)

for i in range(2,31):
    knnr = KNeighborsRegressor(n_neighbors=i)
    knnrmodel = knnr.fit(X_train, y_train)
    y_val_pred = knnrmodel.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    print(i, ":", mse)
# Best n_neighbors = 30? ...
# Did not go too deep as it was taking too much time & it was of high possibility that it would just decrease more and more

start = time.time()
knnr = KNeighborsRegressor(n_neighbors=30)
knnrmodel = knnr.fit(X_train, y_train)
y_test_pred = knnrmodel.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
end = time.time()
end-start #19.17839002609253
#8625.571728151077


################### Random forest regressor
from sklearn.ensemble import RandomForestRegressor
X = og_regression.drop(["review_count"], axis=1)
y = og_regression["review_count"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=5)

for i in range(2,15):
    rfr = RandomForestRegressor(max_features=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_val_pred = rfrmodel.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    print(i, ":", mse)
# Best max_features = 3

for i in range(2,20):
    rfr = RandomForestRegressor(max_features=3, max_depth=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_val_pred = rfrmodel.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    print(i, ":", mse)
# Best max_depth = 15

for i in range(2,15):
    rfr = RandomForestRegressor(max_features=3, max_depth=15, min_samples_split=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_val_pred = rfrmodel.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    print(i, ":", mse)
# Best min_samples_split = 13

temp = []
for i in range(10):
    rfr = RandomForestRegressor(max_features=3, max_depth=15, min_samples_split=13)
    rfrmodel = rfr.fit(X_train, y_train)
    y_test_pred = rfrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    temp.append(mse)
sum(temp)/len(temp)

start = time.time()
rfr = RandomForestRegressor(max_features=3, max_depth=15, min_samples_split=13)
rfrmodel = rfr.fit(X_train, y_train)
y_test_pred = rfrmodel.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
end = time.time()
end-start #2.1161742210388184

# Average mse: 8231.240653630379 (iteration:10)


################### SVM regressor
from sklearn.svm import SVR
X = og_regression.drop(["review_count"], axis=1)
y = og_regression["review_count"]
svmr_scaler = StandardScaler()
X_std = svmr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=5)

for i in range(1,6):
    start = time.time()
    svr = SVR(kernel="linear", epsilon=0.1, C=i)
    svrmodel = svr.fit(X_train, y_train)
    y_val_pred = svrmodel.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    print(i, ":", mse)
    end = time.time()
    print(end-start)
# Best C = 5

svr = SVR(kernel="linear", epsilon=0.1, C=5)
svrmodel = svr.fit(X_train, y_train)
y_test_pred = svrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# Average mse: 21927.4464 (iteration:10)


################### ANN regressor
from sklearn.neural_network import MLPRegressor
X = og_regression.drop(["review_count"], axis=1)
y = og_regression["review_count"]
mlpr_scaler = StandardScaler()
X_std = mlpr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=5)

for i in range(1,31):
    annr = MLPRegressor(hidden_layer_sizes=(i), max_iter=1000, random_state=5)
    annrmodel = annr.fit(X_train, y_train)
    y_val_pred = annrmodel.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    print(mse)
# Best hidden_layer_size = 27

annr = MLPRegressor(hidden_layer_sizes=(27), max_iter=1000)
annrmodel = annr.fit(X_train, y_train)
y_test_pred = annrmodel.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# Average mse: 8834.02570033474 (iteration:10)
