model = get_model()

#y_train2 = np_utils.to_categorical(y_train, num_classes=2)

# fits the model on batches
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=epochs,
    verbose=0,
    shuffle=True,
    batch_size=batch_size)

model.save_weights('binary_model.h5')
