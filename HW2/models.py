import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance

def count_words_dictionary(list_tokens,dic):
    counter = 0
    for token in list_tokens:
        if token in dic['score']:
            counter += 1
    return counter

def sum_words_dictionary(list_tokens,dic):
    counter = 0
    for token in list_tokens:
        if token in dic['score']:
            counter = counter + dic['score'][token]
    return counter

def create_features(df,mourn_dic,nomourn_dic):
    df['NUM_TOKENS'] = df['text'].apply(lambda text: len(text))
    df['COUNT_MOURN'] = df['text'].apply(lambda text: count_words_dictionary(text,mourn_dic))
    df['SUM_MOURN'] = df['text'].apply(lambda text: sum_words_dictionary(text,mourn_dic))
    df['COUNT_NOMOURN'] = df['text'].apply(lambda text: count_words_dictionary(text,nomourn_dic))
    df['SUM_NOMOURN'] = df['text'].apply(lambda text: sum_words_dictionary(text,nomourn_dic))
    df['MOURN_PERC'] = df['COUNT_MOURN']/df['NUM_TOKENS']
    df['NOMOURN_PERC'] = df['COUNT_NOMOURN']/df['NUM_TOKENS']
    return df

def process_df(df,cols):
    df['Y']=np.where(df['tag'] == 'mourning',1,0)
    df=df.loc[:,cols]
    y = df['Y']
    X = df.drop(['Y'],axis=1).fillna(0)
    return X,y

def train_model(X,y,seed,test_prop,model):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=seed,test_size=test_prop)
    if model == 'NB':
        clf = GaussianNB()
    elif model == 'LR':
        clf = LogisticRegression(random_state=seed)
    elif model == 'DT':
        clf = DecisionTreeClassifier(random_state=seed)
    else:
        clf = RandomForestClassifier(random_state=seed)    
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test, clf

def get_results(X,y,seed,test_prop):
    model_list = ['NB','LR','DT','RF']
    result = pd.DataFrame(columns=['Model','Precision','Recall','F1 Score'])
    for model in model_list:
        y_pred_proba, y_test,_ = train_model(X,y,seed,test_prop,model)
        y_pred = np.where(y_pred_proba>=0.5,1,0)
        prec = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        dictionary_results = {'Model': model, 'Precision': prec, 'Recall': recall,
                          'F1 Score': f1}
        
        result = result.append(dictionary_results, ignore_index=True)
    
    return result

def get_rfimportance(X,y,seed,test_prop):

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=seed,test_size=test_prop)
    clf = RandomForestClassifier(random_state=seed)    
    clf.fit(X_train,y_train)
    r = permutation_importance(clf,X_test,y_test)
    return r





