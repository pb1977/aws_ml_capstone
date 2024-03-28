import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

def model(X_train, y_train, epochs, batch_size, early_stop_patient, optimizer_type):
    """Generates the model instanciating the LSTMStockPredictor and makes it read
    for use"""
    model = keras.models.Sequential([
        keras.layers.LSTM(units = 264, return_sequences = True, input_shape= (30, 5)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units = 128, return_sequences = True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units = 64, return_sequences = False),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units = 32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units = 1)])

    
    model.compile(optimizer = optimizer_type,
                 loss = 'mse')

    early_stop = keras.callbacks.EarlyStopping(monitor = "loss",
                                               patience = early_stop_patient,
                                               restore_best_weights = True)

    model.fit(X_train, y_train,
             epochs = epochs,
             batch_size = batch_size,
             callbacks = [early_stop])

    #model.evaluate(X_test, y_test)
    return model


def _data_transformation(array, window = 30):
    """Cut the adjclose column in n columns to be feeded to the model."""
    X_data = []
    y_data = [] # Price on next day
    window = window
    num_shape = len(array)

    for i in range(window, num_shape):
        X_data.append(array[i - window:i, 0:array.shape[1]])
        y_data.append(array[i,0])
    return np.array(X_data), np.array(y_data)


def _load_training_data(base_dir):
    """Load the training data from S3."""
    train_data = pd.read_csv(os.path.join(base_dir, "train.csv")).values
    return _data_transformation(train_data)

def _load_testing_data(base_dir):
    """Load the test data from S3."""
    test_data = pd.read_csv(os.path.join(base_dir, "test.csv")).values
    return _data_transformation(test_data)

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type = int, default = 32)
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--early-stop', type = int, default = 10)

    # Environment variables given by the training image
    #parser.add_argument('--model-dir', type = str, default = os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model_dir', type=str) 
    #parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    #parser.add_argument('--train-data-dir', type = str, default = os.environ['SM_CHANNEL_TRAINING'])
    #parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])    
    parser.add_argument('--optimizer-type', type=str)
    parser.add_argument('--current-host', type = str, default = os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type = list, default = json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    X_train, y_train = _load_training_data(args.training)

    stock_predictor = model(X_train, y_train, 
                             args.epochs, 
                             args.batch_size, 
                             args.early_stop,
                             args.optimizer_type)

    
    
    # Save the model
    version = '00000002'
    ckpt_dir = os.path.join(os.environ['SM_MODEL_DIR'], version)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    stock_predictor.save(ckpt_dir)
    

