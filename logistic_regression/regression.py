import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
# part d
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.special import expit

data = pd.read_csv("Auto.data", na_values=["?"]).dropna()
# data.replace({"?": np.nan}, inplace=True)
# data.dropna()
# part a
median = data["mpg"].median()
print("Median: {}".format(median))
data['mpg01'] = 0
# print(data)
for i, rows in data.iterrows():
    if rows['mpg'] > median:
        data.at[i, 'mpg01'] = 1
print(data)

# part b
# data.plot(kind='box')
# plt.show()
plot_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']
for value in plot_columns:
    boxplot = data.boxplot(column=[value, 'mpg01'])
    plt.show()

# part d
cols = ['horsepower', 'cylinders', 'weight', 'displacement']
X = data[cols]
y = data['mpg01']
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

print("Confusion matrix")
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Precision, recall, F-measure and support
print(classification_report(y_test, y_pred))


for col in cols:
    sns.lmplot(x=col, y="mpg01", data=data.sample(100), logistic=True, ci=None)
    plt.show()

# ROC
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()