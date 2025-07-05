import cv2
import numpy as np


def predict_video(model, video_path, seq_len=20, img_size=64):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < seq_len and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size)) / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) != seq_len:
        print(f"Only {len(frames)} frames extracted. Required: {seq_len}.")
        return

    input_data = np.array(frames).reshape((1, seq_len, img_size, img_size, 3))
    prediction = model.predict(input_data)[0][0]

    print(
        f"Prediction: {prediction:.4f} - {'Violence' if prediction >= 0.5 else 'No Violence'}")
