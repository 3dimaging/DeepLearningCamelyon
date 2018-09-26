import numpy as np
from sklearn.metrics import roc_curve,  precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species'] # labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#figure out the important features

clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)


import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=data[2:]).sort_values(ascending=False)
print(feature_imp)

import matplotlib.pyplot as plt
import seaborn as sns
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# do the real training based on most important features

X=data[['petal length', 'petal width','sepal length']]  # Removed feature "sepal length"
y=data['species']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=5) # 70% training and 30% test
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
scores = clf.predict(test_data_sheet_for_random_forest)
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot(false_positive, true_positive, color='darkorange', label='Random Forest')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (area = %0.2f)' % auc)
plt.legend(loc='best')
plt.savefig('ROCProb.png')
plt.show()
