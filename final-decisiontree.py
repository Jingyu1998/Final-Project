import pandas as pd
from sklearn import tree
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
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
#--------------------------------------------------------------------------------
# 建立分類器
clf = tree.DecisionTreeClassifier(random_state=2)
species_clf = clf.fit(train_simplify_x_dummies, train_simplify_y)
cclf = tree.DecisionTreeClassifier(criterion = "entropy",random_state=2)
species_cclf = cclf.fit(train_simplify_x_dummies, train_simplify_y)
# 預測
test_simplify_y_predicted = species_clf.predict(test_simplify_x_dummies)
#---------------------------------------------------
test_simplify_y_predicted1 = species_cclf.predict(test_simplify_x_dummies)
predict = test_simplify_y_predicted.tolist()
predict1 = test_simplify_y_predicted1.tolist()
#-------------------------------------------------------
test_simplify['predict'] = predict
test_simplify['predict1'] = predict1
#print(test_simplify[[factor[0],'predict','predict1']])
#-----------------------------------------------------------------------------------------------------
answer = {"species": test_simplify[factor[0]],"predict": predict,"predict1": predict1}
answer_df = pd.DataFrame(answer)
#answer_df.to_csv("penguintree1.csv")
#------------------------------------------------------------------------------------------------------
per_all = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]],test_simplify["predict"],average="micro")
per_all1 = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]],test_simplify["predict1"],average="micro")
per = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]],test_simplify["predict"])
per1 = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]],test_simplify["predict1"])

class00 = test_simplify[test_simplify["species"] == "Adelie"]
class01 = test_simplify[test_simplify["species"] == "Chinstrap"]
class02 = test_simplify[test_simplify["species"] == "Gentoo"]
class10 = test_simplify[test_simplify["species"] == "Adelie"]
class11 = test_simplify[test_simplify["species"] == "Chinstrap"]
class12 = test_simplify[test_simplify["species"] == "Gentoo"]
acc00 = sklearn.metrics.accuracy_score(class00[factor[0]],class00["predict"])
acc01 = sklearn.metrics.accuracy_score(class01[factor[0]],class01["predict"])
acc02 = sklearn.metrics.accuracy_score(class02[factor[0]],class02["predict"])
acc10 = sklearn.metrics.accuracy_score(class10[factor[0]],class10["predict"])
acc11 = sklearn.metrics.accuracy_score(class11[factor[0]],class11["predict"])
acc12 = sklearn.metrics.accuracy_score(class12[factor[0]],class12["predict"])
acc_all = sklearn.metrics.accuracy_score(test_simplify[factor[0]],test_simplify["predict"])
acc1_all = sklearn.metrics.accuracy_score(test_simplify[factor[0]],test_simplify["predict1"])
#-------------------------------------------------------------------------------------------------------------
df2 = pd.DataFrame([[acc00, acc01, acc02,acc_all], [acc10, acc11, acc12,acc1_all]],index=["accuracy","accuracy1"] ,columns=["Adelie","Chinstrap","Gentoo","All"])
df2 = df2.drop(index = "accuracy1") #index=labels
df = pd.DataFrame(per,index=["Precision","Recall","F-scroe","Support"],columns=["Adelie","Chinstrap","Gentoo"])
df["All"] = per_all
result0 = df2.append(df)
print(result0)
result0.to_csv("decisiontree-result00-penguin.csv")
#------------------------------------------------------------------------------------------------------------
df2 = pd.DataFrame([[acc00, acc01, acc02,acc_all], [acc10, acc11, acc12,acc1_all]],index=["accuracy0","accuracy"] ,columns=["Adelie","Chinstrap","Gentoo","All"])
df2 = df2.drop(index = "accuracy0") #index=labels
df = pd.DataFrame(per1,index=["Precision","Recall","F-scroe","Support"],columns=["Adelie","Chinstrap","Gentoo"])
df["All"] = per_all1
result1 = df2.append(df)
print(result1)
result1.to_csv("decisiontree-result01-penguin.csv")
#------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#train:test = 3: 1---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
test_simplify = simplify.iloc[3::4].copy() # 3 7 11............
#print(test_simplify.index.tolist())
train_index = list(range(0,len(simplify),4)) + list(range(1,len(simplify),4)) + list(range(2,len(simplify),4))
train_index = sorted(train_index)
train_simplify = simplify.iloc[train_index].copy() # 0 1 2 4 5 6 8 9 10...............
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
#print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
#print(test_simplify_y)
#print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
#print(test_simplify_x_dummies)
#--------------------------------------------------------------------------------
# 建立分類器
clf = tree.DecisionTreeClassifier(random_state=11)
species_clf = clf.fit(train_simplify_x_dummies, train_simplify_y)
cclf = tree.DecisionTreeClassifier(criterion = "entropy",random_state=11)
species_cclf = cclf.fit(train_simplify_x_dummies, train_simplify_y)
# 預測
test_simplify_y_predicted = species_clf.predict(test_simplify_x_dummies)
#---------------------------------------------------
test_simplify_y_predicted1 = species_cclf.predict(test_simplify_x_dummies)
predict = test_simplify_y_predicted.tolist()
predict1 = test_simplify_y_predicted1.tolist()
#-------------------------------------------------------
test_simplify['predict'] = predict
test_simplify['predict1'] = predict1
#print(test_simplify[[factor[0],'predict','predict1']])
#-----------------------------------------------------------------------------------------------------
answer = {"species": test_simplify[factor[0]],"predict": predict,"predict1": predict1}
answer_df = pd.DataFrame(answer)
#answer_df.to_csv("penguintree1.csv")
#------------------------------------------------------------------------------------------------------
per_all = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]],test_simplify["predict"],average="micro")
per_all1 = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]],test_simplify["predict1"],average="micro")
per = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]],test_simplify["predict"])
per1 = sklearn.metrics.precision_recall_fscore_support(test_simplify[factor[0]],test_simplify["predict1"])

class00 = test_simplify[test_simplify["species"] == "Adelie"]
class01 = test_simplify[test_simplify["species"] == "Chinstrap"]
class02 = test_simplify[test_simplify["species"] == "Gentoo"]
class10 = test_simplify[test_simplify["species"] == "Adelie"]
class11 = test_simplify[test_simplify["species"] == "Chinstrap"]
class12 = test_simplify[test_simplify["species"] == "Gentoo"]
acc00 = sklearn.metrics.accuracy_score(class00[factor[0]],class00["predict"])
acc01 = sklearn.metrics.accuracy_score(class01[factor[0]],class01["predict"])
acc02 = sklearn.metrics.accuracy_score(class02[factor[0]],class02["predict"])
acc10 = sklearn.metrics.accuracy_score(class10[factor[0]],class10["predict"])
acc11 = sklearn.metrics.accuracy_score(class11[factor[0]],class11["predict"])
acc12 = sklearn.metrics.accuracy_score(class12[factor[0]],class12["predict"])
acc_all = sklearn.metrics.accuracy_score(test_simplify[factor[0]],test_simplify["predict"])
acc1_all = sklearn.metrics.accuracy_score(test_simplify[factor[0]],test_simplify["predict1"])
#-------------------------------------------------------------------------------------------------------------
df2 = pd.DataFrame([[acc00, acc01, acc02,acc_all], [acc10, acc11, acc12,acc1_all]],index=["accuracy","accuracy1"] ,columns=["Adelie","Chinstrap","Gentoo","All"])
df2 = df2.drop(index = "accuracy1") #index=labels
df = pd.DataFrame(per,index=["Precision","Recall","F-scroe","Support"],columns=["Adelie","Chinstrap","Gentoo"])
df["All"] = per_all
result0 = df2.append(df)
print(result0)
result0.to_csv("decisiontree-result10-penguin.csv")
#------------------------------------------------------------------------------------------------------------
df2 = pd.DataFrame([[acc00, acc01, acc02,acc_all], [acc10, acc11, acc12,acc1_all]],index=["accuracy0","accuracy"] ,columns=["Adelie","Chinstrap","Gentoo","All"])
df2 = df2.drop(index = "accuracy0") #index=labels
df = pd.DataFrame(per1,index=["Precision","Recall","F-scroe","Support"],columns=["Adelie","Chinstrap","Gentoo"])
df["All"] = per_all1
result1 = df2.append(df)
print(result1)
result1.to_csv("decisiontree-result11-penguin.csv")
#------------------------------------------------------------------------------------------------------
