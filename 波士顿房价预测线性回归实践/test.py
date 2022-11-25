from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

X = pd.read_csv('data/boston_preprocessing_X.csv')
y = pd.read_csv('data/boston_preprocessing_y.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = tf.keras.Sequential(
    (tf.keras.layers.Dense(1, input_dim=12))
)

print(model.summary())

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(X_train, y_train, epochs=5000)
