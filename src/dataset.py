from .process import process_record_peer
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import numpy as np

def reduce_peaks(tr, percentile=99, limit_norm=0.9):
    tr = np.asarray(tr)
    tr_abs = np.abs(tr)

    d_max = tr_abs.max()
    if d_max == 0:
        return tr

    n = tr_abs.size
    # index for percentile (clip for safety)
    k = int(np.ceil((percentile / 100.0) * n)) - 1
    k = 0 if k < 0 else (n - 1 if k >= n else k)

    d_per = np.partition(tr_abs, k)[k]

    if d_per / d_max >= limit_norm:
        return tr

    new_d = d_per / limit_norm
    delta_old = d_max - d_per
    if delta_old <= 0:
        return tr

    delta_new = new_d - d_per
    scale = delta_new / delta_old

    out = tr_abs.copy()
    mask = tr_abs > d_per
    out[mask] = d_per + (tr_abs[mask] - d_per) * scale
    np.clip(out, 0.0, new_d, out=out)

    return np.sign(tr) * out

def dis_freq_matrix(acc, dt, fc_hp_array, fc_lp_array, time_res=1024, zero_th=None):

    n_freqs = len(fc_hp_array)
    dis_freq_matrix = np.zeros((n_freqs, time_res))
    time_array = np.arange(len(acc)) * dt

    for i, (fc_hp, fc_lp) in enumerate(zip(fc_hp_array, fc_lp_array)):
        _, _, _dis = process_record_peer(acc, dt, fc_hp, fc_lp, zero_th=zero_th, acausal=True, polynomial_detrend=True)
        _dis = reduce_peaks(_dis, percentile=99, limit_norm=0.95)
        time_new = np.linspace(time_array.min(), time_array.max(), time_res)
        dis = np.interp(time_new, time_array, _dis)
        dis = dis / np.max(np.abs(dis))
        dis_freq_matrix[i, :] = dis

    return dis_freq_matrix

def dis_freq_matrix_parallel(acc, dt, fc_hp_array, fc_lp_array, time_res=1024, zero_th=None, max_workers=None):
    acc = np.ascontiguousarray(acc)
    fc_hp_array = np.asarray(fc_hp_array)
    fc_lp_array = np.asarray(fc_lp_array)

    n = acc.size
    t0 = 0.0
    t1 = (n - 1) * dt

    time_array = np.arange(n, dtype=np.float64) * dt
    time_new = np.linspace(t0, t1, time_res)

    n_freqs = len(fc_hp_array)
    out = np.empty((n_freqs, time_res), dtype=np.float32)

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1))

    def worker(i, fc_hp, fc_lp):
        _, _, dis = process_record_peer(
            acc, dt, float(fc_hp), float(fc_lp),
            zero_th=zero_th, acausal=True, polynomial_detrend=True
        )

        dis = reduce_peaks(dis, percentile=99, limit_norm=0.95)

        dis_rs = np.interp(time_new, time_array, dis).astype(np.float32, copy=False)
        m = float(np.max(np.abs(dis_rs)))
        if m > 0:
            dis_rs /= m
        else:
            dis_rs[:] = 0.0
        return i, dis_rs

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i, fc_hp_array[i], fc_lp_array[i]) for i in range(n_freqs)]
        for fut in as_completed(futures):
            i, row = fut.result()
            out[i, :] = row

    return out

from .process import zero_baseline, cosine_taper, zero_padding, fourier_spectrum, gl, gh, integrate_linear_acceleration
def fast_dis_freq_matrix(acc, dt, fc_hp_array, fc_lp=30.0, time_res=1024):
    n_freqs = len(fc_hp_array)
    dis_freq_matrix = np.zeros((n_freqs, time_res))
    time_array = np.arange(len(acc)) * dt
    time_new = np.linspace(time_array.min(), time_array.max(), time_res)

    acc = np.ascontiguousarray(acc)
    fc_hp_array = np.asarray(fc_hp_array)
    zero_th = 0.01 * dt * len(acc)
    acc = zero_baseline(acc, dt, type='zero-th-1', zero_th=zero_th)
    acc = cosine_taper(acc, dt, zero_th, 'both')
    n_org, acc = zero_padding(acc, dt)
    n = 5
    freq, amp_spectrum, fft_result = fourier_spectrum(acc, dt)
    _gl =  gl(freq, fc_lp, n)
    
    for i, fc_hp in enumerate(fc_hp_array):
        _gh = gh(freq, fc_hp, n)
        fft_filtered = _gl * fft_result * _gh
        new_data = np.fft.irfft(fft_filtered)
        new_data = new_data[:n_org]
        _, dis = integrate_linear_acceleration(new_data, dt)
        dis = reduce_peaks(dis, percentile=99, limit_norm=0.95)
        dis = np.interp(time_new, time_array, dis)
        dis = dis / np.max(np.abs(dis))
        dis_freq_matrix[i, :] = dis
    
    return dis_freq_matrix

def _one_fc_hp(fc_hp, freq, fft_result, gl_filt, n, n_org, dt, time_array, time_new):
    gh_filt = gh(freq, fc_hp, n)
    fft_filtered = gl_filt * fft_result * gh_filt
    new_data = np.fft.irfft(fft_filtered)[:n_org]

    _, dis = integrate_linear_acceleration(new_data, dt)
    dis = reduce_peaks(dis, percentile=99, limit_norm=0.95)
    dis = np.interp(time_new, time_array, dis)

    m = np.max(np.abs(dis))
    if m > 0:
        dis = dis / m
    return dis

def fast_dis_freq_matrix_threaded(acc, dt, fc_hp_array, fc_lp=30.0, time_res=1024, max_workers=None):
    fc_hp_array = np.asarray(fc_hp_array)

    time_array = np.arange(len(acc)) * dt
    time_new = np.linspace(time_array.min(), time_array.max(), time_res)

    acc = np.ascontiguousarray(acc)
    zero_th = 0.01 * dt * len(acc)
    acc = zero_baseline(acc, dt, type="zero-th-1", zero_th=zero_th)
    acc = cosine_taper(acc, dt, zero_th, "both")
    n_org, acc = zero_padding(acc, dt)

    n = 5
    freq, amp_spectrum, fft_result = fourier_spectrum(acc, dt)
    gl_filt = gl(freq, fc_lp, n)

    def worker(fc_hp):
        return _one_fc_hp(fc_hp, freq, fft_result, gl_filt, n, n_org, dt, time_array, time_new)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        dis_list = list(ex.map(worker, fc_hp_array))

    return np.vstack(dis_list)