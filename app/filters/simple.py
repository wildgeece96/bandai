import numpy as np


def cut_weak_signal(spec, k=4):
    """強度が弱い値について、最も強い値に対して10**-k以下であればカットする
    """
    max_amp = spec.max()
    spec = np.where(np.log10(max_amp) - np.log10(spec) > k, 0, spec)
    return spec


def take_moving_average(spec, winfunc=np.hanning, k=5):
    """各周波数ごとに、時間軸方向に移動平均をとる
    単にならすのではなく、窓関数を使って真ん中にピークが来るような平滑化を行う
    """
    window = winfunc(k) / winfunc(k).sum()  # 合計が1になるよう正規化する
    for freq_idx in range(spec.shape[0]):
        spec[freq_idx] = np.convolve(spec[freq_idx], window, mode='same')
    return spec
