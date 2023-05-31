import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

path=r'C:\Users\Darek\Documents\Magisterka\PracaMagPreproccessing\Results'


def initialize_result_file():
    file = open(path + "/results.csv", "w")
    file.write("org_dataset,dataset,group,scenario,xgb_score,rfc_score,knn_score\n")

def score(dataset,suffix,group,scenario,X_train,y_train,X_test,y_test):
    print("========= " + dataset + "_"+ str(suffix) + " =========")
    print("Scenario: " + scenario)
    xgb = xgboost.XGBClassifier(n_estimators=10)
    xgb.fit(X_train, y_train)
    xgb_score = xgb.score(X_test, y_test)
    print("Xgboost: " + str(xgb_score))
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    rfc.fit(X_train, y_train)
    rfc_score = rfc.score(X_test, y_test)
    print("Random Forest Classifier: " + str(rfc_score))
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print("KNeighbors Classifier: " + str(knn_score)) 
    file = open(path + "/results.csv", "a")
    file.write(dataset+","+dataset+'_'+str(suffix)+","+group+","+scenario+","+str(xgb_score)+","+str(rfc_score)+","+str(knn_score)+"\n")