from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def model_trainer():

    model_list = []

    logreg = LogisticRegression(
        solver='lbfgs' 
    )
    model_list.append(logreg)
    
    rf = RandomForestClassifier(
        criterion='gini',
        n_estimators=1000,
        max_features='sqrt'
    )
    model_list.append(rf)

    dt = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_features='sqrt'
    )
    model_list.append(dt)

    print('Model list created')
    
    return model_list


def initiate_model_trainer(X_train, X_test, y_train, y_test):

    models = model_trainer()

    models_score =  defaultdict(dict)

    for i in models:
        i.fit(X_train, y_train)
        y_pred = i.predict(X_test)

        model_name = type(i).__name__
        models_score[model_name]['F1 Score'] = f'{f1_score(y_test, y_pred):.2f}'
        models_score[model_name]['Accuracy Score'] = f'{accuracy_score(y_test, y_pred)*100:.2f}%'
        models_score[model_name]['Recall Score'] = f'{recall_score(y_test, y_pred):.2f}'
        models_score[model_name]['Precision Score'] = f'{precision_score(y_test, y_pred):.2f}'

    model_list = []

    for k, v in models_score.items():
        print(f'{k}')
        model_list.append(k)
        for k, v in v.items():
            print(f'{k} - {v}')
        print('\n')

    f1_list = sorted(
        model_list,
        key = lambda x: (models_score[x]['F1 Score']),
        reverse = True
    )    
        
    accuracy_list = sorted(
        model_list,
        key = lambda x: (models_score[x]['Accuracy Score']),
        reverse = True
    ) 

    recall_list = sorted(
        model_list,
        key = lambda x: (models_score[x]['Recall Score']),
        reverse = True
    )   

    precision_list = sorted(
        model_list,
        key = lambda x: (models_score[x]['Precision Score']),
        reverse = True
    )   

    print(f'F1 Score Ranking: {f1_list}')
    print(f'Accuracy Score Ranking: {accuracy_list}')
    print(f'Recall Score Ranking: {recall_list}')
    print(f'Precision Score Ranking: {precision_list}')