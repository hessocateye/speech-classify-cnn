import sys
import numpy as np
import librosa
import tensorflow as tf

# --------------------
# CONFIG (must match training)
# --------------------
N_MELS = 40
MAX_FRAMES = 200
TOP_DB = 80

MODEL_PATH   = "emotion_cnn_full_model.keras"
CLASSES_PATH = "classes.npy"

# --------------------
# Audio preprocessing
# --------------------
def compute_mel(audio_wav_filename):
    y, sr = librosa.load(audio_wav_filename)

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

    # Pad / truncate to fixed length
    if mel_db.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :MAX_FRAMES]

    # Shape: (1, 40, 200, 1)
    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    return mel_db

# --------------------
# Prediction
# --------------------
def predict(wav_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH)

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
    for cls, p in zip(np.load(CLASSES_PATH), probs):
        print(f"{cls:>12s} : {p:.3f}")
