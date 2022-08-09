# Loan-Classification-Machine-Learning

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, ShuffleSplit, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, plot_confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
import category_encoders as ce

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
```
<img width="553" alt="image" src="https://user-images.githubusercontent.com/99155979/183617031-a152c467-d913-4f38-8fb3-963c24462eb1.png">

**Analytic Approach**

This project will aims at creating a model that is predicting (Predictive) client late payment probability using the data.The data will be analyze to obtain pattern that differentiate between client who will make a late payment and customer who is not make a late payment. With the classification model obatained prescribtion will be formulated to prevent company to approved client who is going to make a late payment.

**Metric Evaluation**

Type 1 error : False Positive (Model predicts client makes a late payment while it is on time)/REJECTED

Consequence: losing potention revenue (admin commission fee 1% and 3% flat each month).
            
Type 2 error : False Negative (Model predicts client makes an on time payment while it is late)/APPROVED

Consequence: late revenue and additional cost for catching up the late revenue.

RECALL(+) CHURN

Based on the consequences, what we will is to create a model that can prevent company to lose potention revenue, but without adding more cost for catching up the late revenue by the client. So we have to balance later between precision and recall from the positive class (potential candidate). So later the main metric that we will use is roc_auc.

```python
df=pd.read_csv('ind_app_train.csv')
df.head()
```
<img width="577" alt="image" src="https://user-images.githubusercontent.com/99155979/183617365-8b45e190-85a9-40d2-be39-51dc31ee124e.png">

```python
desc = []
for i in df.columns:
    desc.append([
        i,
        df[i].dtypes,
        df[i].isna().sum(),
        round(((df[i].isna().sum() / len(df)) * 100), 2),
        df[i].nunique(),
        df[i].drop_duplicates().sample(2).values
    ])

pd.DataFrame(desc, columns=[
    "Data Features",
    "Data Types",
    "Null",
    "Null Percentage",
    "Unique",
    "Unique Sample"])
```

<img width="359" alt="image" src="https://user-images.githubusercontent.com/99155979/183617919-2ec97895-0920-4ef0-9210-6fc42e00b357.png">

## Data Preprocessing
```python
df.drop(['Unnamed: 0','LN_ID','EXT_SCORE_1','EXT_SCORE_2','EXT_SCORE_3'], axis=1, inplace=True)
```
ID is being dropped because it has unique value for every row or data and also it doesn't have no effect on the target, external score also being dropped because it is a score from external data that we know nothing about.

```python
df = df[pd.notnull(df['ANNUITY'])]
df = df[pd.notnull(df['PRICE'])]
```
Drop missing values because the amount is small compared to the amount of data.

```python
def minus(x):
    if x <0:
        return x * (-1)
    else:
        return x
        
df['DAYS_AGE'] = df['DAYS_AGE'].apply(minus)
df['DAYS_WORK'] = df['DAYS_WORK'].apply(minus)
df['DAYS_REGISTRATION'] = df['DAYS_REGISTRATION'].apply(minus)
df['DAYS_ID_CHANGE'] = df['DAYS_ID_CHANGE'].apply(minus)
```
Change the negative value on every feature that has negative value.

```python
plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot = True, cbar = False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/183618758-8d377513-4d2a-48d2-a257-c68572112f63.png)

```python
import dython
from dython.nominal import associations, cramers_v, theils_u, correlation_ratio

assoc_cr = []
col = ['NUM_CHILDREN', 'INCOME', 'APPROVED_CREDIT', 'ANNUITY', 'PRICE', 'DAYS_AGE', 'DAYS_WORK', 'DAYS_REGISTRATION', 'DAYS_ID_CHANGE', 'HOUR_APPLY']
for i in df.drop(columns = col).columns:
    assoc = round(cramers_v(df['TARGET'], df[i]), 2)
    assoc_cr.append(assoc)

df_cr = pd.DataFrame(data = [assoc_cr], columns = df.drop(columns = col).columns, index = ['TARGET'])

plt.figure(figsize = (15,1))
sns.heatmap(df_cr, annot = True, cbar = False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/183618929-7527e7b9-e949-406b-9cc7-b25f8fe7a1bf.png)

```python
transformer = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), ['CONTRACT_TYPE', 'GENDER', 'INCOME_TYPE', 'EDUCATION', 'FAMILY_STATUS', 'HOUSING_TYPE', 'WEEKDAYS_APPLY', 'ORGANIZATION_TYPE'])
], remainder='passthrough')

X = df.drop(columns=['TARGET'])
y = df['TARGET']

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=42)

logreg = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()

models = [logreg,knn,dt,rf,xgb]
score=[]
rata=[]
std=[]

for i in models:
    skfold=StratifiedKFold(n_splits=5)
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',i)])
    model_cv=cross_val_score(estimator,X_train,y_train,cv=skfold,scoring='roc_auc')
    score.append(model_cv)
    rata.append(model_cv.mean())
    std.append(model_cv.std())
    
pd.DataFrame({'model':['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'],'mean roc_auc':rata,'sdev':std}).set_index('model').sort_values(by='mean roc_auc',ascending=False)
```
<img width="215" alt="image" src="https://user-images.githubusercontent.com/99155979/183619184-a5d51b4d-4209-425d-9b5c-65f9eb9710ec.png">
It can be seen that the XGBoost model is the best model for its roc_auc of any model that uses the default hyperparameter

```python
models = [logreg,knn,dt,rf,xgb]
score_roc_auc = []

def y_pred_func(i):
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',i)])
    X_train,X_test
    
    estimator.fit(X_train,y_train)
    return(estimator,estimator.predict(X_test),X_test)

for i,j in zip(models, ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost']):
    estimator,y_pred,X_test = y_pred_func(i)
    y_predict_proba = estimator.predict_proba(X_test)[:,1]
    score_roc_auc.append(roc_auc_score(y_test,y_predict_proba))
    print(j,'\n', classification_report(y_test,y_pred))
    
pd.DataFrame({'model':['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'],
             'roc_auc score':score_roc_auc}).set_index('model').sort_values(by='roc_auc score',ascending=False)
```
<img width="306" alt="image" src="https://user-images.githubusercontent.com/99155979/183619668-25810692-706a-43df-a65d-be9227f0132d.png">
<img width="308" alt="image" src="https://user-images.githubusercontent.com/99155979/183619751-a69eb4ca-c8f7-48ee-82d7-1dff06b66550.png">
<img width="293" alt="image" src="https://user-images.githubusercontent.com/99155979/183619791-342a7624-6e89-4f61-8987-d4a2d342ef46.png">
<img width="179" alt="image" src="https://user-images.githubusercontent.com/99155979/183619862-92af24a4-3c10-4732-959b-90437359fe05.png">
The XGBoost model is still the best performing on the test data.

Now I will try to oversampling our XGBoost model to see if we can get even better results.

## Test Oversampling with K-Fold Cross Validation
```python
def calc_train_error(X_train, y_train, model):
#     '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    predictProba = model.predict_proba(X_train)
    accuracy = accuracy_score(y_train, predictions)
    f1 = f1_score(y_train, predictions, average='macro')
    roc_auc = roc_auc_score(y_train, predictProba[:,1])
    recall = recall_score(y_train, predictions)
    precision = precision_score(y_train, predictions)
    report = classification_report(y_train, predictions)
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }
    
def calc_validation_error(X_test, y_test, model):
#     '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    predictProba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, predictProba[:,1])
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
#     '''fits model and returns the in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error
```
```python
K = 10
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
data = X_train
target = y_train
```
```python
train_errors_without_oversampling = []
validation_errors_without_oversampling = []

train_errors_with_oversampling = []
validation_errors_with_oversampling = []

for train_index, val_index in kf.split(data, target):
    
    # split data
    X_train_1, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train_1, y_val = target.iloc[train_index], target.iloc[val_index]
    
#     print(len(X_val), (len(X_train) + len(X_val)))
    ros = RandomOverSampler()

    X_ros, y_ros = ros.fit_resample(X_train_1, y_train_1)

    # instantiate model
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',xgb)
    ])

    #calculate errors
    train_error_without_oversampling, val_error_without_oversampling = calc_metrics(X_train_1, y_train_1, X_val, y_val, estimator)
    train_error_with_oversampling, val_error_with_oversampling = calc_metrics(X_ros, y_ros, X_val, y_val, estimator)
    
    # append to appropriate list
    train_errors_without_oversampling.append(train_error_without_oversampling)
    validation_errors_without_oversampling.append(val_error_without_oversampling)
    
    train_errors_with_oversampling.append(train_error_with_oversampling)
    validation_errors_with_oversampling.append(val_error_with_oversampling)
```

## Evaluation Metrics Without Oversampling
```python
listItem = []

for tr,val in zip(train_errors_without_oversampling,validation_errors_without_oversampling) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['recall'],val['recall'],tr['precision'],val['precision']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvaluate = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 
                            'Test Accuracy', 
                            'Train ROC AUC', 
                            'Test ROC AUC', 
                            'Train F1 Score',
                            'Test F1 Score',
                            'Train Recall',
                            'Test Recall',
                            'Train Precision',
                            'Test Precision'])

listIndex = list(dfEvaluate.index)
listIndex[-1] = 'Average'
dfEvaluate.index = listIndex
dfEvaluate
```
<img width="562" alt="image" src="https://user-images.githubusercontent.com/99155979/183620258-468ba633-b0ee-4b19-8f8f-89e6e7e954a5.png">

## Evaluation Metrics With Oversampling
```python
listItem = []

for tr,val in zip(train_errors_with_oversampling,validation_errors_with_oversampling) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['recall'],val['recall'],tr['precision'],val['precision']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvaluate = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 
                            'Test Accuracy', 
                            'Train ROC AUC', 
                            'Test ROC AUC', 
                            'Train F1 Score',
                            'Test F1 Score',
                            'Train Recall',
                            'Test Recall',
                            'Train Precision',
                            'Test Precision'])

listIndex = list(dfEvaluate.index)
listIndex[-1] = 'Average'
dfEvaluate.index = listIndex
dfEvaluate
```
<img width="565" alt="image" src="https://user-images.githubusercontent.com/99155979/183620403-d6d0a9c0-f08c-4643-a68e-6d40450d5f24.png">

From the evaluation metrics results, the recall(+) result is better after oversampling than before oversampling, but the precision(+) result is worse after oversampling than before oversampling. There is a tradeoff between recall and precision because the amount of data from the minority class is the same as the majority class.

## Hyperparameter Tuning
```python
pipe_XGB = Pipeline([
    ('prep', transformer),
    ('algo', XGBClassifier())
])

param_XGB = {
    "algo__n_estimators" : np.arange(50, 601, 50),
    "algo__max_depth" : np.arange(1, 10),
    "algo__learning_rate" : np.logspace(-3, 0, 4),
    "algo__gamma" : np.logspace(-3, 0, 6),
    "algo__colsample_bytree" : [0.3, 0.5, 0.7, 0.8],
    "algo__subsample" : [0.3, 0.5, 0.7, 0.8],
    "algo__reg_alpha" : np.logspace(-3, 3, 7),
    "algo__reg_lambda" : np.logspace(-3, 3, 7)
}

skf = StratifiedKFold(n_splits=10)

GS_XGB = GridSearchCV(pipe_XGB, param_XGB, cv = skf, scoring='roc_auc', verbose = 3, n_jobs=-1)
RS_XGB = RandomizedSearchCV(pipe_XGB, param_XGB,cv = skf, scoring='roc_auc', verbose = 3, n_jobs=-1 )
RS_XGB.fit(X_train, y_train)

RS_XGB.best_params_
```
<img width="208" alt="image" src="https://user-images.githubusercontent.com/99155979/183623562-e5fa1b5e-e84f-408d-8c05-6f29c9da86f5.png">

```python
XGB_Tuned = RS_XGB.best_estimator_
print(classification_report(y_test, XGB_Tuned.predict(X_test)))
```
<img width="308" alt="image" src="https://user-images.githubusercontent.com/99155979/183623710-54e16436-4907-4311-be1a-e25155053b4b.png">

```python
plot_confusion_matrix(XGB_Tuned, X_test, y_test, display_labels=['On time', 'Late'])
```
![download](https://user-images.githubusercontent.com/99155979/183623781-ea660126-6422-4b6d-9f9e-068cdab60761.png)

```python
coef1 = pd.Series(XGB_Tuned['algo'].feature_importances_, transformer.get_feature_names()).sort_values(ascending = False).head(10)
coef1.plot(kind='barh', title='Feature Importances')
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/183623858-2c7a3e3b-309f-4b26-9953-3f41093bded3.png)
