import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display


def plot_spectrogram(S, sr=22050, title=None, save_path=None):
    """取得したmelspectrogramをプロットしてみる
    """
    _plot_spectrogram(S, sr=sr, title=title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_compared_spectrograms(
        S0, S1, sr=22050, title0=None, title1=None, save_path=None):
    """2つのmelspectrogramを比較するプロットをする関数
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    _plot_spectrogram(S0, sr=sr, title=title0, colorbar=True)

    plt.subplot(1, 2, 2)
    _plot_spectrogram(S1, sr=sr, title=title1, colorbar=True)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def _plot_spectrogram(S, sr=22050, title=None, colorbar=True):
    """melspectrogramのプロットするnaive関数
    """
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
                        S_dB, x_axis='time',
                        y_axis='mel', sr=sr,
                        fmax=sr//2)
    if colorbar:
        plt.colorbar(format='%+2.0f dB', shrink=0.6)
    if title is not None:
        plt.title(title, fontsize=16)
    else:
        plt.title('Mel-frequency spectrogram')
    