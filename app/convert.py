# 音声変換のためのコード置き場
import librosa


def reconstruct_audio(M, sr=22050):
    """スペクトログラムから音声を再構築する
    """
    audio = librosa.feature.inverse.mel_to_audio(M, sr=sr)
    return audio
