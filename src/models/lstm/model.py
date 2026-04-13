from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM as KerasLSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class LSTM:
    def __init__(
        self,
        sequence_length=50,
        n_features=5,
        lstm_units_1=128,
        lstm_units_2=64,
        dropout=0.3,
        dense_units=32,
        learning_rate=1e-3,
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate

        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            KerasLSTM(self.lstm_units_1, return_sequences=True),
            Dropout(self.dropout),
            KerasLSTM(self.lstm_units_2),
            Dense(self.dense_units, activation="relu"),
            Dense(1),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def fit_feature_scaler(self, X_train):
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        self.feature_scaler.fit(X_train_2d)

    def transform_features(self, X):
        X_2d = X.reshape(-1, X.shape[-1])
        X_scaled = self.feature_scaler.transform(X_2d)
        return X_scaled.reshape(X.shape)

    def fit_target_scaler(self, y_train):
        self.target_scaler.fit(y_train.reshape(-1, 1))

    def transform_target(self, y):
        return self.target_scaler.transform(y.reshape(-1, 1)).flatten()

    def inverse_transform_target(self, y_scaled):
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        batch_size=32,
        verbose=1,
        model_ckpt_path=None,
    ):
        self.fit_feature_scaler(X_train)
        self.fit_target_scaler(y_train)

        X_train_scaled = self.transform_features(X_train)
        X_val_scaled = self.transform_features(X_val)

        y_train_scaled = self.transform_target(y_train)
        y_val_scaled = self.transform_target(y_val)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        if model_ckpt_path is not None:
            model_ckpt_path = Path(model_ckpt_path)
            model_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_ckpt_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=0,
                )
            )

        history = self.model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        return history

    def predict(self, X):
        X_scaled = self.transform_features(X)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0).flatten()
        y_pred = self.inverse_transform_target(y_pred_scaled)
        return y_pred