import pandas as pd
import sklearn
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from math import sqrt
#---------------------------------------------------------------------------------
simplify = pd.read_csv('penguins_size.csv') #7 344
full = pd.read_csv('penguins_lter.csv') # 17 344
#---------------------------------------------------------------------------------
test_simplify = simplify.iloc[2::3].copy() # 2 5 8............
##print(test_simplify.index.tolist())
train_index = list(range(0,len(simplify),3)) + list(range(1,len(simplify),3))
train_index = sorted(train_index)
train_simplify = simplify.iloc[train_index].copy() # 0 1 3 4...............
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
##print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
##print(test_simplify_y)
##print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
##print(test_simplify_x_dummies)
#----------------------------------------------------------------------------------
dummy = test_simplify_x_dummies.columns.tolist()
print(dummy)
#-------------------------------------------------------------------------------------------------
# 建立 random forest 模型
max_sample = [0.6,0.7,0.8]
max_feature = [round(sqrt(len(dummy))-1),round(sqrt(len(dummy))),round(sqrt(len(dummy))+1)]
print(max_feature)
#fit--------------------------------------------------------------------------------------------------------------------
count = 0
i = 0
while i < len(max_sample):
    j = 0
    while j < len(max_feature):
        forest = ensemble.RandomForestClassifier(criterion="entropy",random_state=100,max_samples=max_sample[i],max_features=max_feature[j])
        model0 = forest.fit(train_simplify_x_dummies, train_simplify_y)
        # 預測
        test_simplify_y_predicted = model0.predict(test_simplify_x_dummies)
        predict = test_simplify_y_predicted.tolist()
        # -------------------------------------------------------
        test_simplify['predict'] = predict
        answer = {"species": test_simplify[factor[0]], "predict": predict}
        answer_df = pd.DataFrame(answer)
        print(answer_df)
        # -------------------------------------------------------------------------
        per_all = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]], test_simplify["predict"],
                                                                  average="micro")
        per = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]], test_simplify["predict"])
        class00 = test_simplify[test_simplify["species"] == "Adelie"]
        class01 = test_simplify[test_simplify["species"] == "Chinstrap"]
        class02 = test_simplify[test_simplify["species"] == "Gentoo"]
        acc00 = sklearn.metrics.accuracy_score(class00[factor[0]], class00["predict"])
        acc01 = sklearn.metrics.accuracy_score(class01[factor[0]], class01["predict"])
        acc02 = sklearn.metrics.accuracy_score(class02[factor[0]], class02["predict"])
        acc_all = sklearn.metrics.accuracy_score(test_simplify[factor[0]], test_simplify["predict"])
        # -------------------------------------------------------------------------------------------------------------
        df2 = pd.DataFrame([[acc00, acc01, acc02, acc_all], [0, 0, 0, 0]],
                           index=["accuracy", "accuracy1"], columns=["Adelie", "Chinstrap", "Gentoo", "All"])
        df2 = df2.drop(index="accuracy1")  # index=labels
        df = pd.DataFrame(per, index=["Precision", "Recall", "F-scroe", "Support"],
                          columns=["Adelie", "Chinstrap", "Gentoo"])
        df["All"] = per_all
        result0 = df2.append(df)
        print(result0)
        result0.to_csv("randomforest-result0" + str(count) + "-penguin.csv")
        count = count + 1
        j = j + 1
    i = i + 1
    #---------------------------------------------------------------------------------------------------------------------
#train:test = 3: 1---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
test_simplify = simplify.iloc[3::4].copy() # 3 7 11............
#print(test_simplify.index.tolist())
train_index = list(range(0,len(simplify),4)) + list(range(1,len(simplify),4)) + list(range(2,len(simplify),4))
train_index = sorted(train_index)
train_simplify = simplify.iloc[train_index].copy() # 0 1 2 4 5 6 8 9 10...............
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
##print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
##print(test_simplify_y)
##print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
##print(test_simplify_x_dummies)
#----------------------------------------------------------------------------------
dummy = test_simplify_x_dummies.columns.tolist()
print(dummy)
#-------------------------------------------------------------------------------------------------
# 建立 random forest 模型
max_sample = [0.6,0.7,0.8]
max_feature = [round(sqrt(len(dummy))-1),round(sqrt(len(dummy))),round(sqrt(len(dummy))+1)]
print(max_feature)
#fit--------------------------------------------------------------------------------------------------------------------
count = 0
i = 0
while i < len(max_sample):
    j = 0
    while j < len(max_feature):
        forest = ensemble.RandomForestClassifier(criterion="entropy",random_state=100,max_samples=max_sample[i],max_features=max_feature[j])
        model0 = forest.fit(train_simplify_x_dummies, train_simplify_y)
        # 預測
        test_simplify_y_predicted = model0.predict(test_simplify_x_dummies)
        predict = test_simplify_y_predicted.tolist()
        # -------------------------------------------------------
        test_simplify['predict'] = predict
        answer = {"species": test_simplify[factor[0]], "predict": predict}
        answer_df = pd.DataFrame(answer)
        print(answer_df)
        #-------------------------------------------------------------------------
        per_all = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]], test_simplify["predict"],average="micro")
        per = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]], test_simplify["predict"])
        class00 = test_simplify[test_simplify["species"] == "Adelie"]
        class01 = test_simplify[test_simplify["species"] == "Chinstrap"]
        class02 = test_simplify[test_simplify["species"] == "Gentoo"]
        acc00 = sklearn.metrics.accuracy_score(class00[factor[0]], class00["predict"])
        acc01 = sklearn.metrics.accuracy_score(class01[factor[0]], class01["predict"])
        acc02 = sklearn.metrics.accuracy_score(class02[factor[0]], class02["predict"])
        acc_all = sklearn.metrics.accuracy_score(test_simplify[factor[0]], test_simplify["predict"])
        # -------------------------------------------------------------------------------------------------------------
        df2 = pd.DataFrame([[acc00, acc01, acc02, acc_all], [0,0,0,0]],
                           index=["accuracy", "accuracy1"], columns=["Adelie", "Chinstrap", "Gentoo", "All"])
        df2 = df2.drop(index="accuracy1")  # index=labels
        df = pd.DataFrame(per, index=["Precision", "Recall", "F-scroe", "Support"],
                          columns=["Adelie", "Chinstrap", "Gentoo"])
        df["All"] = per_all
        result0 = df2.append(df)
        print(result0)
        result0.to_csv("randomforest-result1" + str(count) +"-penguin.csv")
        count = count + 1
        j = j + 1
    i = i + 1
    #---------------------------------------------------------------------------------------------------------------------