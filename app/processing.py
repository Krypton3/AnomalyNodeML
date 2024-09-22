import json
import helper
import pandas as pd
import numpy as np
from log import LogFile
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Use the custom callback during training
log_callback = LogFile('logs/logs.txt')


def bytes_to_neumeric(value: str) -> float:
    value = str(value).strip()
    if 'M' in value:
        return float(value.replace('M', '').strip()) * 1e6
    elif 'K' in value:
        return float(value.replace('K', '').strip()) * 1e3
    else:
        return float(value)


async def data_processing(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                                       pd.Series, pd.Series]:
    # cleaning
    data = data.drop(columns=['Date first seen', 'attackType', 'attackID',
                              'attackDescription'])
    # handling categorical data
    label_encoder = LabelEncoder()

    for column in ['Proto', 'Src IP Addr', 'Dst IP Addr', 'Flags', 'class']:
        data[column] = label_encoder.fit_transform(data[column])

    # Apply the conversion to the 'Bytes' column
    data['Bytes'] = data['Bytes'].apply(bytes_to_neumeric)

    # handling missing values
    data = data.fillna(0)

    # features and targets
    X = data.drop(columns=['class'])
    y = data['class']

    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # standarize the data
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)

    return X_train, X_test, y_train, y_test


async def load_and_process_data(data: str) -> \
        Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    # load the data
    try:
        response = await helper.read_file(data)
        data_dict = json.loads(response)
        # Check for status in the returned dictionary
        if data_dict["status"] != 200:
            return {"status": "error", "message": data_dict["message"]}

        # Extract the actual data content
        data = data_dict["content"]
        df = pd.DataFrame(data)
        log_callback.log_message(f"Data loaded and converted to DataFrame: {df.head()}")
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Failed to decode JSON: {e}"}
    return await data_processing(df)
