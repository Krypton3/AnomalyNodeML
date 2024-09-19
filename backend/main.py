import tensorflow as tf
from log import LogFile
import matplotlib.pyplot as plt
from data_processing.processing import load_and_process_data


def plot_history(history: dict):
    # plotting training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('data/loss.png')

    # plotting the training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('data/accuracy.png')

    fig, ax1 = plt.subplots()

    # Plot loss on primary y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss',
             color='orange')
    ax1.legend(loc='upper left')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(history.history['accuracy'], label='Training Accuracy',
             color='green')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy',
             color='red')
    ax2.legend(loc='upper right')

    plt.title('Loss and Accuracy over Epochs')
    plt.savefig('data/loss_accuracy.png')


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
    history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                        validation_split=0.2, callbacks=[log_callback])

    # plot the results
    plot_history(history)

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    log_callback.log_message(f'Test accuracy: {test_acc}')
    log_callback.log_message(f'Test loss: {test_loss}')


if __name__ == "__main__":
    main()
