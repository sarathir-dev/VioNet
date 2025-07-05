import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.vionet import build_vionet
from utils.data_loader import load_data
from utils.trainer import get_callbacks
from utils.evaluator import evaluate_model
from utils.visualizer import plot_training_curves
from utils.predictor import predict_video

DATA_DIR = 'data/HockeyFights'
IMG_SIZE = 64
SEQ_LEN = 20

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if not gpus:
    raise RuntimeError(
        "No GPU found.")
else:
    print(f"Using GPU: {gpus[0].name}")

X, y = load_data(DATA_DIR, SEQ_LEN, IMG_SIZE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

with tf.device('/GPU:0'):
    model = build_vionet(input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=4,
        callbacks=get_callbacks()
    )

plot_training_curves(history)
evaluate_model(model, X_test, y_test)

predict_video(model, 'data/HockeyFights/Violence/fi185_xvid.avi')
