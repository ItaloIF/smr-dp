from obspy import read
import matplotlib.pyplot as plt
import time 

from src.plot import plot_acceleration, plot_dis_freq_matrix, plot_prediction
from src.dataset import dis_freq_matrix_parallel, fast_dis_freq_matrix, fast_dis_freq_matrix_threaded
from src.model import inference_model
from src.metrics import best_corner_local_gaussian

def test_making_matrix():
    st = read('data/20260106101800s.pickle')
    print(st)
    
    tr = st[2]
    acc = tr.data
    dt = tr.stats.delta
    
    # make fc_hp_array numpy
    fc_hps = [0.005 * i for i in range(1, 257)]
    fc_lps = [30.0] * len(fc_hps)

    start_time = time.time()
    
    # dfm = dis_freq_matrix_parallel(acc, dt, fc_hp_array=fc_hps, fc_lp_array=fc_lps, time_res=1024, zero_th=None, max_workers=10)
    # dfm = fast_dis_freq_matrix(acc, dt, fc_hps, fc_lp=30.0, time_res=1024)
    dfm = fast_dis_freq_matrix_threaded(acc, dt, fc_hps, fc_lp=30.0, time_res=1024, max_workers=2)
    end_time = time.time()
    print(f"Time taken to create the matrix: {end_time - start_time:.2f} seconds")
    plot_dis_freq_matrix(dfm, save_path='img/dis_freq_matrix.png')

    pred = inference_model(dfm)
    xy = best_corner_local_gaussian(pred, sigma=5)
    plot_prediction(pred, xy, save_path='img/prediction.png')
    p_arrival = xy[1] / 1024 * len(acc) * dt

    plot_acceleration(acc, dt, p_arrival=p_arrival, save_path='img/acceleration_plot.png')

if __name__ == "__main__":
    test_making_matrix()