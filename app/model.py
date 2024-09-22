import os
import json
import helper
import pandas as pd
import numpy as np
import tensorflow as tf
from log import LogFile
from plot import plot_history
from processing import load_and_process_data


# Use the custom callback during training
log_callback = LogFile('app_log/logs.txt')

MODEL_PATH = "data/anomaly_model.keras"
TRAIN_DATA_PATH = "data/"


async def model(train_data: str) -> json:
    try:
        # extract the training and testing data
        X_train, X_test, y_train, y_test = await load_and_process_data(
            TRAIN_DATA_PATH + train_data)
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

        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                            validation_split=0.2, callbacks=[log_callback])

        # plot the results
        plot_history(history)

        # save the model
        model.save(MODEL_PATH)

        if os.path.exists(MODEL_PATH):
            # Evaluate the model on the test data
            test_loss, test_acc = model.evaluate(X_test, y_test)
            log_callback.log_message(f'Test accuracy: {test_acc}')
            log_callback.log_message(f'Test loss: {test_loss}')

            # save the trained data to the global variable
            await helper.modify_global(train_data)
            log_callback.log_message(f'Currently trained data list: {helper.TRAINED_DATA}')

            # Return success response
            return json.dumps({
                "status": "success",
                "message": "Model created and saved successfully",
                "test_accuracy": test_acc,
                "test_loss": test_loss
            })
        else:
            return json.dumps({
                "status": "error",
                "message": "Model saving failed"
            })
    except Exception as e:
        # Log the exception and return an error response
        log_callback.log_message(f"An error occurred: {e}")
        return json.dumps({
            "status": "error",
            "message": f"An error occurred while training the model: {e}"
        })


async def model_evaluation(model_path: str, evaluate_path: str) -> json:
    # Load the Keras model
    # Load and process the CSV data
    try:
        model = tf.keras.models.load_model(model_path)
        X_train, X_test, y_train, y_test = await load_and_process_data(
            evaluate_path)
    except Exception as e:
        log_callback.log_message(f"An error occurred: {e}")
        return json.dumps({
            "status": "error",
            "message": f"An error occurred to extract the evaluation data: {e}"
        })

    # Concatenate X_train and X_test, y_train and y_test into one dataset
    X = np.concatenate((X_train, X_test), axis=0)
    y_true = np.concatenate((y_train, y_test), axis=0)

    # Make predictions using the model
    predictions = model.predict(X)

    # If it's binary classification, round predictions (0 or 1)
    y_pred = np.round(predictions).flatten()

    # Create a DataFrame with the features and true labels
    data = pd.DataFrame(X)
    data['true_label'] = y_true
    data['predicted_label'] = y_pred

    # Count correctly predicted samples
    correct_predictions = data[data['true_label'] == data['predicted_label']]
    correct_count = len(correct_predictions)

    # Count mispredicted samples
    incorrect_predictions = data[data['true_label'] != data['predicted_label']]
    incorrect_count = len(incorrect_predictions)

    # Calculate the percentage of correct predictions
    total_samples = correct_count + incorrect_count
    correct_percentage = (correct_count / total_samples) * 100
    incorrect_percentage = (incorrect_count / total_samples) * 100

    # Convert the correctly predicted samples to JSON format
    # correct_predictions_json = correct_predictions.to_dict(orient='records')

    # Return the counts, percentages, and correctly predicted samples
    return json.dumps({
        "status": "success",
        "correct_predictions": correct_count,
        "mispredicted_predictions": incorrect_count,
        "correct_percentage": correct_percentage,
        "incorrect_percentage": incorrect_percentage
    })
