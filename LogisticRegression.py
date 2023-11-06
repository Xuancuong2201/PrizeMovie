import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def max_value(df3, variable, top):
    return np.where(df3[variable] > top, top, df3[variable])
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

    elif train == False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
df = pd.read_csv("Movie.csv")


IQR = df.popularity.quantile(0.75) - df.popularity.quantile(0.25)
Lower_fence = df.popularity.quantile(0.25) - (IQR * 3)
Upper_fence = df.popularity.quantile(0.75) + (IQR * 3)
print("Popularity outliers are values < {lowerboundary} or > {upperboundary}".format(lowerboundary=Lower_fence,
                                                                                         upperboundary=Upper_fence))
IQR = df.budget.quantile(0.75) - df.budget.quantile(0.25)
Lower_fence = df.budget.quantile(0.25) - (IQR * 3)
Upper_fence = df.budget.quantile(0.75) + (IQR * 3)
print('Budget outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,
                                                                                         upperboundary=Upper_fence))
IQR = df.vote_average.quantile(0.75) - df.vote_average.quantile(0.25)
Lower_fence = df.vote_average.quantile(0.25) - (IQR * 3)
Upper_fence = df.vote_average.quantile(0.75) + (IQR * 3)
print('Vote average outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,
                                                                                         upperboundary=Upper_fence))
IQR = df.Tomatoes.quantile(0.75) - df.Tomatoes.quantile(0.25)
Lower_fence = df.Tomatoes.quantile(0.25) - (IQR * 3)
Upper_fence = df.Tomatoes.quantile(0.75) + (IQR * 3)
print('Tomatoes outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,
                                                                                         upperboundary=Upper_fence))
IQR = df.Profit_Margin.quantile(0.75) - df.Profit_Margin.quantile(0.25)
Lower_fence = df.Profit_Margin.quantile(0.25) - (IQR * 3)
Upper_fence = df.Profit_Margin.quantile(0.75) + (IQR * 3)
print('Profit Margin outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,
                                                                                         upperboundary=Upper_fence))
X = df.drop(['Success'], axis=1)
y = df['Success']

# Chia X và y thành tập training và testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

for df in [X_train, X_test]:
    df['popularity'] = max_value(df, 'popularity', 30.01)
    df['budget'] = max_value(df, 'budget', 251000000)
    df['vote_average'] = max_value(df, 'vote_average', 10.60)
    df['Tomatoes'] = max_value(df, 'Tomatoes', 224)
    df['Profit_Margin'] = max_value(df, 'Profit_Margin',3.59)

X_train = pd.DataFrame(X_train, columns=['popularity', 'budget', 'vote_average', 'Tomatoes', 'Profit_Margin'])
X_test = pd.DataFrame(X_test, columns=['popularity', 'budget', 'vote_average', 'Tomatoes', 'Profit_Margin'])

# Scale dữ liệu
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf', gamma=1)
model.fit(X_train, y_train)
print_score(model, X_train, y_train, X_test, y_test, train=False)

logreg = LogisticRegression(solver='liblinear', random_state=0)
logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))

