from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, min_lr=1e-6)
    ]
