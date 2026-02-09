from .filter import butterworth_bandpass, cosine_taper, zero_baseline, butterworth_bandpass_iir
from .integration import integrate_linear_acceleration, differentiate_displacement

import numpy as np

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

def rotate_component(data_ew, data_ns, theta, degrees=False):
    # theta is taken from EW to NS angle
    if degrees:
        theta = np.radians(theta)

    data_rot = data_ew * np.cos(theta) + data_ns * np.sin(theta)

    return data_rot

def compute_RotD(data_ew, data_ns, n_angles=180):
    # compute RotD00, RotD50, RotD100

    if len(data_ns) != len(data_ew):
        raise ValueError("NS and EW components must be the same length")
    
    n = len(data_ew)
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    rotds_max = []

    for i, angle in enumerate(angles):
        rot_serie = rotate_component(data_ew, data_ns, angle, degrees=False)
        rotds_max.append(np.max(np.abs(rot_serie)))

    RotD00 = np.percentile(rotds_max, 0)
    RotD50 = np.percentile(rotds_max, 50)
    RotD100 = np.percentile(rotds_max, 100)

    return RotD00, RotD50, RotD100