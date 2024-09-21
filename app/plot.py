import matplotlib.pyplot as plt


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