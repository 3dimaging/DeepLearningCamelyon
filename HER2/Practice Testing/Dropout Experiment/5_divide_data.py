X_train, y_train, z_train = get_data(BASE_DIR + 'images/TRAIN/')
X_test, y_test, z_test = get_data(BASE_DIR + 'images/TEST/')

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

