

# General libraries.
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import *
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import *

convert_zeroes = True

# Read in the data from the file and load into a DataFrame
def get_data(filename):
    df = pd.read_csv(filename)
    # sort by VisitNumber to ensure everything is in the right order  
    df = df.sort_values('VisitNumber')
    df = df.reset_index()
    # del df['index']
    return df

# code to transform data to create new features nd 
def feature_transformation(df, df_other, convert_zeroes=False):

    # Replace nulls will 'Unknown' product categories
    df['Upc'] = df['Upc'].fillna('UnknownUpc')
    df['DepartmentDescription'] = df['DepartmentDescription'].fillna('UnknownDD')
    df['FinelineNumber'] = df['FinelineNumber'].fillna('UnknownFN')

    # Replace nulls will 'Unknown' product categories
    df_other['Upc'] = df_other['Upc'].fillna('UnknownUpc')
    df_other['DepartmentDescription'] = df_other['DepartmentDescription'].fillna('UnknownDD')
    df_other['FinelineNumber'] = df_other['FinelineNumber'].fillna('UnknownFN')
    
    # Create a group of field headers that include categories from both the training and the test data sets
    VisitNumber_u = list(sorted(df.VisitNumber.unique()))
    VisitNumber_u_other = list(sorted(df_other.VisitNumber.unique()))

    Upc_u = list(sorted(df.Upc.unique()))
    Upc_u_other = list(sorted(df_other.Upc.unique()))
    Upc_all = sorted(list(set(list(set(Upc_u)|set(Upc_u_other)))))

    FN_u = list(sorted(df.FinelineNumber.unique()))
    FN_u_other = list(sorted(df_other.FinelineNumber.unique()))
    FN_all = sorted(list(set(FN_u)|set(FN_u_other)))

    DD_u = list(sorted(df.DepartmentDescription.unique()))
    DD_u_other = list(sorted(df_other.DepartmentDescription.unique()))
    DD_all = sorted(list(set(DD_u)|set(DD_u_other)))
    
    # Convert Weekday text variable to numerical
    df['Weekday'] = df['Weekday'].map({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7})
 
    # Create a return dummy
    df['Return'] = np.where(df['ScanCount'] > 0, 0 , 1)

    # Convert ScanCount negatives to 0's
    if convert_zeroes:
        df['ScanCount'] = np.where(df['ScanCount'] < 0 , 0, df['ScanCount'])

    # Aggregate number of items by Department Description, i.e. sum ScanCount
    dd = df.groupby(['VisitNumber','DepartmentDescription'], as_index=False)['ScanCount'].sum()
    dd = dd.rename(columns={'ScanCount': 'ItemsDD'})
    df = pd.merge(left=df, right=dd, on=['VisitNumber','DepartmentDescription'], how='left')

    # Aggregate number of items by FinelineNumber, i.e. sum ScanCount
    fn = df.groupby(['VisitNumber','FinelineNumber'], as_index=False)['ScanCount'].sum()
    fn = fn.rename(columns={'ScanCount': 'ItemsFN'})
    df = pd.merge(left=df, right=fn, on=['VisitNumber','FinelineNumber'], how='left')        

    # Aggregate number of products by VisitNumber, i.e. count ScanCount
    wd = df.groupby(['VisitNumber','Weekday'], as_index=False)['ScanCount'].count()
    wd = wd.rename(columns={'ScanCount': 'NumProducts'})
    Weekday_u = list(sorted(wd.Weekday.unique()))
    df = pd.merge(left=df, right=wd, on=['VisitNumber','Weekday'], how='left')
    
    # Create a return dummy for each shopping visit
    rt = df.groupby(['VisitNumber'], as_index=False)['Return'].sum()
    rt['Return'] = np.where(rt['Return'] > 0 , 1, 0)

    # Aggregate number of items by VisitNumber, i.e sum ScanCount
    tt = df.groupby(['VisitNumber'], as_index=False)['ScanCount'].sum()
    tt = tt.rename(columns={'ScanCount': 'NumItems'})
    df = pd.merge(left=df, right=tt, on=['VisitNumber'], how='left')
    
    # Combine aggregates and sort by VisitNumber to get ordered row names of VisitNumber
    tt['NumProducts'] = wd.NumProducts
    tt['Return'] = rt.Return
    tt.sort_values('VisitNumber')
    aggs = tt[['NumItems', 'NumProducts', 'Return']] 

    # Isolate Visit Numbers
    visit_numbers = tt.VisitNumber

    # Create a sparse matrix of Weekday dummies for VisitNumber
    data = wd['Weekday'].tolist()
    row = wd.VisitNumber.astype('category', categories=VisitNumber_u).cat.codes
    col = wd.Weekday.astype('category', categories=Weekday_u).cat.codes
    Weekday_sm = csr_matrix((data, (row, col)), shape=(len(VisitNumber_u), len(Weekday_u)))

    # Create a sparse matrix of number of items by Upc for each VisitNumber
    data = df['ScanCount'].tolist()
    row = df.VisitNumber.astype('category', categories=VisitNumber_u).cat.codes
    col = df.Upc.astype('category', categories=Upc_all).cat.codes
    Upc_sm = csr_matrix((data, (row, col)), shape=(len(VisitNumber_u), len(Upc_all)))

    # Create a sparse matrix of number of items by FinelineNumber for each VisitNumber
    data = df['ItemsFN'].tolist()
    row = df.VisitNumber.astype('category', categories=VisitNumber_u).cat.codes
    col = df.FinelineNumber.astype('category', categories=FN_all).cat.codes
    FN_sm = csr_matrix((data, (row, col)), shape=(len(VisitNumber_u), len(FN_all)))
    
    # Create a sparse matrix of number of items by DepartmentDescription for each VisitNumber
    data = df['ItemsDD'].tolist()
    row = df.VisitNumber.astype('category', categories=VisitNumber_u).cat.codes
    col = df.DepartmentDescription.astype('category', categories=DD_all).cat.codes
    DD_sm = csr_matrix((data, (row, col)), shape=(len(VisitNumber_u), len(DD_all)))

    # Create a sparse matrix of the high level aggregate features
    aggs_u = ['NumItems', 'NumProducts', 'Return']
    aggs_sm = csr_matrix(aggs.values)
    aggs_sm

    # Horizontally stack the blocks
    sm_m = hstack(blocks=[aggs_sm,Weekday_sm,Upc_sm,FN_sm,DD_sm],format='csr')
    sm_l = hstack(blocks=[aggs_sm,Weekday_sm,DD_sm],format='csr')
    print sm_m.shape
    print sm_l.shape

    return sm_l, VisitNumber_u

def get_target(df):
    # Aggregate number of items by VisitNumber, i.e sum ScanCount
    tt = df.groupby(['VisitNumber','TripType'], as_index=False)['ScanCount'].sum()
    tt = tt.rename(columns={'ScanCount': 'NumItems'})
    target = tt.TripType
    return target

def model_run(X, y, single_classifier=True):
    # Make a log-loss scorer for use in GridSearchCV
    my_log_loss = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)    # Fit model using a single classifier on training data sparse matrix (sm)
    if single_classifier:
        # Set up classifier
        clf = LogisticRegression(C=1000,multi_class='multinomial',solver='newton-cg',n_jobs=-1,tol=1,max_iter=400,warm_start=True)
        # Fit grid search
        clf.fit(X,y)
        return clf
    # Fit model using GridSearchCV on the training data sparse matrix (sm)
    else:
        # Set up classifier
        clf = MultinomialNB()
        # clf = LogisticRegression(C=1000,multi_class='multinomial',solver='newton-cg',n_jobs=-1,tol=1,max_iter=200,warm_start=True)
        # clf = LogisticRegression(C=10,multi_class='multinomial',solver='newton-cg',n_jobs=-1,tol=0.0001,max_iter=200,warm_start=True)
        # Set up grid search
        # gs = GridSearchCV(estimator=clf, param_grid={'alpha': [round(float(i)/100,2) for i in range(20,31)]},n_jobs=-1,cv=4)
        gs = GridSearchCV(estimator=clf, param_grid={'alpha': [round(float(i)/100,2) for i in range(20,31)]},n_jobs=-1,cv=4,scoring=my_log_loss)
        # Fit the grid search
        gs.fit(X,y) 
        return gs
 
def decision_tree(X,y):
    dt = DecisionTreeClassifier(max_depth=20,max_leaf_nodes=40)
    dt.fit(X,y)
    return dt

def random_forest(X,y):
    rf = RandomForestClassifier(max_depth=20,max_leaf_nodes=40)
    rf.fit(X,y)
    return rf

def gradient_boosting(X,y):
    gb = GradientBoostingClassifier()
    gb.fit(X,y)
    return gb

def print_gridsearch(gs):
    # Report grid search results 
    print gs.grid_scores_
    print gs.best_estimator_
    print gs.best_score_
    print gs.best_params_
    
def predictions(clf, X, y, ys):
    # Make predictions on the training data sparse matrix (sm)
    train_preds = clf.predict(X) 

    # Collect predicted probabilities
    train_probs = clf.predict_proba(X)

    # Report various accuracy metrics
    print "Log loss:", round(metrics.log_loss(y,train_probs,eps=1e-15),3)
    print "F1 score:", round(metrics.f1_score(y,train_preds,average='micro'),3)
    print ""
    print "Clasification Report:"
    print classification_report(y,train_preds)
    print ""
    print "Summary confusion matrix:"
    cm = confusion_matrix(y,train_preds)
    for i in range(38):
        max_wrong = 0
        k = -1
        for j in range(38):
            if i != j:
                if cm[i][j] > max_wrong:
                    k = j
                    max_wrong = cm[i][j]
        print ys[i], ys[k], max_wrong 

def predictions_report(clf, X, y, ys):
    # Make predictions on the training data sparse matrix (sm)
    train_preds = clf.predict(X) 

    # Collect predicted probabilities
    train_probs = clf.predict_proba(X)

    # Report various accuracy metrics
    logloss = round(metrics.log_loss(y,train_probs,eps=1e-15),3)
    F1_score = round(metrics.f1_score(y,train_preds,average='micro'),3)
    return logloss, F1_score
    
# Function to write probabilities to a csv file in the correct submission format
def write_probs_to_file(vn,tt,probs):
    # open file to write results
    with open("walmart.csv", "w") as results:
        # write header
        my_str = ""
        my_str += '"VisitNumber"' + "," 
        for trip_num in tt[:-1]:
            my_str += '"TripType_' + str(trip_num) + '"' + ","
        my_str += '"TripType_' + str(tt[-1]) + '"' + "\n" 
        results.write(my_str)
        # write probs for each visit
        for i in range(probs.shape[0]):
            my_str = ""
            my_str += str(vn[i]) + "," 
            for j in range(probs.shape[1]):
                my_str += str(probs[i][j]) + ","
            my_str = my_str[:-1] + "\n"
            results.write(my_str)

def main():           
    # create data sets
    train_data = get_data('train.csv')
    del train_data['index']
    test_data = get_data('test.csv')
    del test_data['index']
    target = get_target(train_data)
    del train_data['TripType']

    # create sparse matrices
    train_X, train_visit_numbers = feature_transformation(train_data,test_data,True)
    train_y = target
    test_X, test_visit_numbers = feature_transformation(test_data,train_data,True)
    test_trip_types = sorted(list(set(target)))

    # scale features to be in the range [-1, 1] based on maximum absolute value of each feature
    # retains integrity of zero values - good for sparse data
    scaler = MaxAbsScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    #TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
    # reduce the dimensionality of the data
    dim_reducer = TruncatedSVD(n_components=8)
    train_X_scaled_dimr = dim_reducer.fit_transform(train_X_scaled)
    test_X_scaled_dimr = dim_reducer.transform(test_X_scaled)

    # print test_X_scaled_dimr.shape
    # print dim_reducer.explained_variance_ratio_
    # print np.sum(dim_reducer.explained_variance_ratio_)

    # Gradient Boosting unscaled features
    clf = GradientBoostingClassifier(max_depth=3,min_samples_leaf=4,max_leaf_nodes=40)
    clf.fit(train_X.toarray(),train_y)
    # print_gridsearch(clf)
    logloss, F1_score = predictions_report(clf,train_X.toarray(),train_y,test_trip_types)

    # write file to working directory
    test_probs = clf.predict_proba(test_X.toarray())
    write_probs_to_file(test_visit_numbers,test_trip_types,test_probs)
    return logloss, F1_score



