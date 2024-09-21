import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def bytes_to_neumeric(value):
    value = str(value).strip()
    if 'M' in value:
        return float(value.replace('M', '').strip()) * 1e6
    elif 'K' in value:
        return float(value.replace('K', '').strip()) * 1e3
    else:
        return float(value)


def data_processing(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
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


def load_and_process_data() -> Tuple[np.ndarray, np.ndarray,
                                     pd.Series, pd.Series]:
    # load the data
    data = pd.read_csv("data/server_logs.csv")
    return data_processing(data)
