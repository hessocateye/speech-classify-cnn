import sys
import numpy as np
import librosa
import tensorflow as tf

# --------------------
# CONFIG (must match training)
# --------------------
N_MELS = 40
MAX_FRAMES = 200
TOP_DB = 20

MODEL_WEIGHTS = "emotion_cnn.weights.h5"
MODEL_PATH_KERAS   = "emotion_cnn_full_model.keras" # Will need Keras 3.x (same as in Colab)
MODEL_PATH_H5   = "emotion_cnn_full_model.h5" #Will work with Keras 2.x
CLASSES_PATH = "classes.npy"

# --------------------
# Audio preprocessing
# --------------------

def pad_or_crop(mel):
    t = mel.shape[1]
    if t < MAX_FRAMES:
        return np.pad(
            mel,
            ((0, 0), (0, MAX_FRAMES - t)),
            mode="constant",
            constant_values=mel.min()
        )
    start = (t - MAX_FRAMES) // 2
    return mel[:, start:start + MAX_FRAMES]

def compute_mel(audio_wav_filename):
    y, sr = librosa.load(audio_wav_filename, sr=None)
    print(sr)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # Mel spectrogram (EXACTLY like training)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        fmax=min(8000, sr / 2)
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = pad_or_crop(mel_db)

    # # Pad / truncate to fixed length
    # if mel_db.shape[1] < MAX_FRAMES:
    #     pad_width = MAX_FRAMES - mel_db.shape[1]
    #     mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    # else:
    #     mel_db = mel_db[:, :MAX_FRAMES]

    print('Mel: {}'.format(mel_db.mean(axis=1)))

    silence_ratio = (mel_db < -70).mean()
    print("Silence ratio:", silence_ratio)

    # Shape: (1, 40, 200, 1)
    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    print(mel_db.shape)
    print(mel_db.min(), mel_db.max())
    return mel_db

def build_model(n_convs, lr_schedule, num_classes, compile=True):
    inputs = tf.keras.Input(shape=(N_MELS, MAX_FRAMES, 1)) # Assuming 40x200 mel spectrograms
    x = inputs

    filters = [32, 64, 128, 256][:n_convs]

    for i, f in enumerate(filters):
        x = tf.keras.layers.Conv2D(
            f,
            (3,3),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        pool = (2,2) if i < 3 else (2,1)
        x = tf.keras.layers.MaxPooling2D(pool)(x)
        x = tf.keras.layers.Dropout(0.25 if i < 3 else 0.3)(x)
        # End For (filters)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    return model


# --------------------
# Prediction
# --------------------
def predict(wav_path):
    
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    try:
        try:
            print('Trying to load keras model.....')
            model = tf.keras.models.load_model(MODEL_PATH_KERAS, compile=False)
            print('Sucessfully loaded keras model!')
        except:
            print('Trying to load model from H5 file.....')
            model = tf.keras.models.load_model(MODEL_PATH_H5, compile=False)
            print('Sucessfully loaded model from H5 file!')
    except:
        # If everything fails, we will build model and load weights
        print('Trying to build model explicitly without loading....')
        model = build_model(
            n_convs=4,          # BEST VALUE, HARD-CODED
            lr_schedule=None,   # DUMMY, IGNORED
            num_classes=len(classes),
            compile=False
        )
        model.load_weights(MODEL_WEIGHTS)
        print('Successfully built model architecture from scratch and loaded weights')
    
    model.summary()

    mel = compute_mel(wav_path)
    probs = model.predict(mel, verbose=0)[0]

    pred_idx = np.argmax(probs)
    return classes[pred_idx], probs[pred_idx], probs

# --------------------
# CLI entry
# --------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio.wav>")
        sys.exit(1)

    wav_file = sys.argv[1]
    emotion, confidence, probs = predict(wav_file)

    print(f"Predicted emotion : {emotion}")
    print(f"Confidence        : {confidence:.3f}\n")

    print("Class probabilities:")
    for cls, p in zip(np.load(CLASSES_PATH, allow_pickle=True), probs):
        print(f"{cls:>12s} : {p:.3f}")
