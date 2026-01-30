import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def save_spectrogram(audio, path):
    plt.figure(figsize=(10, 4))
    S = librosa.stft(audio)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(S)))
    plt.colorbar()
    plt.savefig(path)
    plt.close()
