import numpy as np
from scipy import signal
from ctypes import *

lib = CDLL('core/processing.so')

def integrate_linear_acceleration(data, dt):
    n = len(data)
    n_c = c_int(n)
    data_c = data.astype(c_float)
    dt_c = c_float(dt)
    vel_c = np.zeros(n, dtype=c_float)
    dis_c = np.zeros(n, dtype=c_float)

    lib.integration_linear_acceleration_(
        byref(n_c),
        byref(dt_c),
        data_c.ctypes.data_as(POINTER(c_float)),
        vel_c.ctypes.data_as(POINTER(c_float)),
        dis_c.ctypes.data_as(POINTER(c_float))
    )

    vel = vel_c.astype(float)
    dis = dis_c.astype(float)
    return vel, dis

def differentiate_displacement(dis, dt):
    n = len(dis)
    n_c = c_int(n)
    dis_c = dis.astype(c_float)
    dt_c = c_float(dt)
    vel_c = np.zeros(n, dtype=c_float)
    acc_c = np.zeros(n, dtype=c_float)

    lib.differentiate_displacement_(
        byref(n_c),
        byref(dt_c),
        dis_c.ctypes.data_as(POINTER(c_float)),
        vel_c.ctypes.data_as(POINTER(c_float)),
        acc_c.ctypes.data_as(POINTER(c_float))
    )

    vel = vel_c.astype(float)
    acc = acc_c.astype(float)
    return vel, acc

def process_record_peer(acc, dt, fc_hp, fc_lp, zero_th=None, acausal=True, polynomial_detrend=True):
    # PEER NGA-Sub: Bozorgnia et al. 2020 pp. 19-30
    # PEER NGA-East: Goulet et al., 2021

    # 0. zero-th order baseline correction
    if zero_th is None:
        zero_th = 0.01 * dt * len(acc)
    acc = zero_baseline(acc, dt, type='zero-th-1', zero_th=zero_th)

    # 1. Cosine Taper
    time_taper = max(0.5, 0.01 * dt * len(acc))
    acc = cosine_taper(acc, dt, time_taper, 'both')

    # 2. Filter the data
    if acausal:
        acc = butterworth_bandpass(acc, dt, fc_hp, fc_lp, padding=True)
    else:
        acc = butterworth_bandpass_iir(acc, dt, fc_hp, fc_lp, n=5, acausal=acausal)

    # 3. Integrated using linear acceleration
    vel, dis = integrate_linear_acceleration(acc, dt)

    # 4. Polynomial fit to displacement
    if polynomial_detrend:
        time_array = np.arange(0, len(dis)*dt, dt)
        p = np.polyfit(time_array, dis, 6)
        r_dis = np.polyval(p, time_array)
        dis = dis - r_dis
        vel, acc = differentiate_displacement(dis, dt)

    return acc, vel, dis

def zero_padding(data, dt):
    n_org = len(data)
    if dt == 0.02:
        nt = 2**18
    if dt == 0.01:
        nt = 2**19
    elif dt == 0.005:
        nt = 2**20
    elif dt == 0.002:
        nt = 2**22
    elif dt == 0.001:
        nt = 2**23
    elif dt == 0.0005:
        nt = 2**24

    new_data = np.concatenate((data, np.zeros(nt - n_org)))

    return n_org, new_data

def fourier_spectrum(data, dt):
    # Unnormalized amplitude spectrum physics‑correct
    fft_result = np.fft.rfft(data)
    freq = np.fft.rfftfreq(len(data), dt)
    amp_spectrum = np.abs(fft_result) * dt  
    amp_spectrum[1:-1] = 2 * amp_spectrum[1:-1]

    return freq, amp_spectrum, fft_result

def gl(f, fl, n):
    return ( (f/fl)**(2*n)/(1 + (f/fl)**(2*n)))**0.5

def gh(f, fh, n):
    return ( 1/(1 + (f/fh)**(2*n)) )**0.5

def butterworth_bandpass(data, dt, fl=None, fh=None, n=5, padding=False):

    if padding:
        n_org, data = zero_padding(data, dt)

    # Acausal Butterworth Filter
    freq, amp_spectrum, fft_result = fourier_spectrum(data, dt)
    # Highpass filter
    if not fl:
        _gl = np.ones(len(freq))
    else:
        _gl = gl(freq, fl, n)

    # Lowpass filter
    if not fh:
        _gh = np.ones(len(freq))
    else:
        _gh = gh(freq, fh, n)

    fft_filtered = _gl*fft_result*_gh

    new_data = np.fft.irfft(fft_filtered)

    if padding:
        new_data = new_data[:n_org]

    return new_data

def butterworth_bandpass_iir(data, dt, fl=None, fh=None, n=4, acausal=True):
    x = np.asarray(data)

    # No cutoffs -> no filtering
    if fl is None and fh is None:
        return x.copy()

    fs = 1.0 / dt
    nyq = 0.5 * fs

    # Validate cutoffs
    if fl is not None and (fl <= 0 or fl >= nyq):
        raise ValueError(f"fl must be in (0, {nyq}). Got fl={fl}")
    if fh is not None and (fh <= 0 or fh >= nyq):
        raise ValueError(f"fh must be in (0, {nyq}). Got fh={fh}")
    if fl is not None and fh is not None and fl >= fh:
        raise ValueError(f"Require fl < fh. Got fl={fl}, fh={fh}")

    # Select filter type
    if fl is not None and fh is not None:
        Wn = [fl / nyq, fh / nyq]
        btype = "bandpass"
    elif fl is not None:
        Wn = fl / nyq
        btype = "highpass"
    else:
        Wn = fh / nyq
        btype = "lowpass"

    sos = signal.butter(N=n, Wn=Wn, btype=btype, output="sos")

    return signal.sosfiltfilt(sos, x) if acausal else signal.sosfilt(sos, x)

def cosine_taper(data, dt, time_taper=0.5, type='end'):
    # PEER NGA-Sub: Bozorgnia et al. 2020 pp. 22
    ntaper = int(time_taper / dt)
    a = np.arange(0, ntaper)
    ws = 0.5 * ( 1 + np.cos(np.pi * (ntaper + a) / ntaper) )
    we = 0.5 * ( 1 + np.cos(np.pi * (a) / ntaper) )
    if type == 'start':
        data[:ntaper] *= ws
    elif type == 'end':
        data[-ntaper:] *= we
    elif type == 'both':
        data[:ntaper] *= ws
        data[-ntaper:] *= we
    return data

def zero_baseline(data, dt, zero_th, type = 'mean'):
    if type == 'mean':
        data = data - np.mean(data)
    elif type == 'linear':
        # make dtrend in three lines, linear in the beginning, linear in the end, and linear thah join them
        nzeroth = max(int(zero_th / dt), int(0.05*len(data)))
        detrend_start = data[:nzeroth] - signal.detrend(data[:nzeroth], type='linear')
        detrend_end = data[-nzeroth:] - signal.detrend(data[-nzeroth:], type='linear')
        dtrend = np.concatenate((
            detrend_start,
            np.linspace(detrend_start[-1], detrend_end[0], len(data) - 2 * nzeroth),
            detrend_end
        ))
        data = data - dtrend

    elif type == 'zero-th-1':
        nzeroth = max(int(zero_th / dt), int(0.05*len(data)))
        data =  data - np.mean(data[:nzeroth])

    elif type == 'zero-th-2':
        nzeroth = max(int(zero_th / dt), int(0.05*len(data)))
        acc_in =  np.mean(data[:nzeroth])
        acc_out = np.mean(data[-nzeroth:])
        dtrend = np.concatenate((
            np.full(nzeroth, acc_in),
            np.linspace(acc_in, acc_out, len(data) - 2 * nzeroth),
            np.full(nzeroth, acc_out)
        ))
        data = data - dtrend
    else:
        raise ValueError("Invalid type. Choose 'mean', 'zero-th-1', or 'zero-th-2'.")
    return data