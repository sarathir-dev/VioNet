import os
import cv2
import numpy as np


def load_data(data_dir, seq_len=20, img_size=64):
    X, y = [], []
    for label in ['Violence', 'NonViolence']:
        folder = os.path.join(data_dir, label)
        for video in os.listdir(folder):
            cap = cv2.VideoCapture(os.path.join(folder, video))
            frames = []
            while len(frames) < seq_len and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (img_size, img_size)) / 255.0
                frames.append(frame)
            cap.release()
            if len(frames) == seq_len:
                X.append(frames)
                y.append(1 if label == 'Violence' else 0)
    return np.array(X), np.array(y)
