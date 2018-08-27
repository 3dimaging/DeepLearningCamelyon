from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

print('Predicting on test data')
#y_pred = np.rint(model.predict(X_test))
#y_pred = np.rint(model.predict(X_test))
y_pred = model.predict(X_test)

print('Accuracy')
#print(accuracy_score(y_test, y_pred[:,0]))
print(accuracy_score(y_test, np.rint(y_pred[:])))

print('AUC')
print(roc_auc_score(y_test, np.rint(y_pred[:])))


#print(z_test)
#print(y_pred)

for i in range(0, len(y_pred)):
    #print(y_pred[i])
    print('{} {}'.format(z_test[i], y_pred[i]))

