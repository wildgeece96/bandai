import glob
import os
import numpy as np
import wavio
import IPython.display

from convert import reconstruct_audio
from plot import (
    plot_spectrogram,
    plot_compared_spectrograms
)


def _load_binary(paths):
    specs = {}
    for path in paths:
        audio_id = int(path.split(".")[-2].split("_")[-1])
        spec = np.load(path)
        specs[audio_id] = spec
    return specs


def load_spectrograms(data_dir):
    """指定したdirectory以下に展開したdataファイルを読み込む
    """
    noised_paths = glob.glob(os.path.join(data_dir, "noised_tgt/*.npy"))
    raw_paths = glob.glob(os.path.join(data_dir, "raw/*.npy"))
    raw_specs = _load_binary(raw_paths)  # ノイズなしのスペクトログラム一覧
    noised_specs = _load_binary(noised_paths)  # ノイズ混じりのスペクトログラム一覧
    return raw_specs, noised_specs


def display_audio(audio, sr=22050):
    IPython.display.display(IPython.display.Audio(audio, rate=sr))


def save_audio(audio, path, sr=22050):
    audio /= audio.mean()
    audio *= 2**15 / audio.max()
    wavio.write(path, audio, sr, sampwidth=2)


def show_and_save(specs, save_name="noised", ids=[], convert_func=lambda x: x):
    """スペクトログラムの可視化と音声の保存。
    convert_funcに変換の音声を指定すれば変換結果を保存することができる。
    Notebook上で呼び出す前提の関数
    """
    os.makedirs(f"../out/{save_name}/fig", exist_ok=True)
    os.makedirs(f"../out/{save_name}/audio", exist_ok=True)
    iterator = specs.keys() if len(ids) == 0 else ids
    for idx in iterator:
        idx = int(idx)
        raw_spec = specs[idx].copy()
        spec = convert_func(raw_spec)
        plot_compared_spectrograms(
            raw_spec, spec,
            title0=f"raw audio id:{idx:03d}",
            title1=f"{save_name} audio id:{idx:03d}",
            save_path=f"../out/{save_name}/fig/{idx:03d}.png")
        audio = reconstruct_audio(spec, sr=22050)
        audio_path = f"../out/{save_name}/audio/{idx:03d}.wav"
        save_audio(audio, audio_path, sr=22050)
        display_audio(audio, sr=22050)


def save_specs(specs, save_dir="../out", mode='tgt'):
    """変換したスペクトログラムを指定したディレクトリ以下に一括で保存する
    Parameters:
    =========
    specs (dict):
        key (int): id of the audio.
        value (ndarray): output spectrogram
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx in specs:
        spec = specs[idx]
        if mode == 'tgt':
            file_name = f"tgt_{idx:03d}.npy"
        else:
            file_name = f"cln_{idx:03d}.npy"
        save_path = os.path.join(save_dir, file_name)
        np.save(save_path, spec)
