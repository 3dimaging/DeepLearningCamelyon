from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, np.rint(y_pred)))

cf_10_4 = confusion_matrix(y_test, np.rint(y_pred))
