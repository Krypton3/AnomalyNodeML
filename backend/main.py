import tensorflow as tf
from data_processing.processing import load_and_process_data
from log import LogFile


def main():
    # extract the training and testing data
    X_train, X_test, y_train, y_test = load_and_process_data()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu',
                              input_shape=(X_train.shape[1], )),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Use the custom callback during training
    log_callback = LogFile('training_log.txt')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32,
              validation_split=0.2, callbacks=[log_callback])

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')


if __name__ == "__main__":
    main()