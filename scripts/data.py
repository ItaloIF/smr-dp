from hashlib import new
from pydoc import doc
import tkcore
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided
from datetime import datetime, timedelta, timezone
from scipy.signal import decimate
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time

client = tkcore.mdb.get_mongo_client()

def generate_records_df():
    db = client['JP']
    collection = db['metadata']

    query = {
        'processing_flags.labeler_tool': {'$eq': True},
        'quality_flags.do_not_use': {'$ne': True},
        'original.PGA_gal.HM' : {'$gte': 50.0},
    }

    cursor = collection.find(query)
    dataset = []
    for doc in cursor:
        for ch in ['UD', 'NS', 'EW']:
            # check if fc_hp exists
            if 'processed' not in doc or 'fc_hp' not in doc['processed'] or ch not in doc['processed']['fc_hp']:
                fc_hp = 0.0
            else:
                fc_hp = doc['processed']['fc_hp'][ch]['d2']
            
            dt = doc['original']['dt']
            npts = doc['original']['npts']
            start_time = 0.0 if pd.isna(doc['time_markers']['start_time']) else doc['time_markers']['start_time']
            end_time = 0.0 if pd.isna(doc['time_markers']['end_time']) else doc['time_markers']['end_time']

            # round start_time and end_time to most clode integer
            start_time = round(start_time, 0)
            end_time = round(end_time, 0)

            if end_time > 0.0:
                record_duration = end_time - start_time
            else:
                record_duration = dt * npts - start_time

            data = {
                'id': str(doc['_id']) + f"_{ch}",
                'dt': dt,
                'record_duration': record_duration,
                'original_start_time': start_time,
                'original_end_time': end_time,
                'P_wave_arrival': doc['time_markers']['P_wave_arrival'] - start_time,
                # 'S_wave_arrival': doc['time_markers']['S_wave_arrival'] - start_time,
                # 'avs30_m/s': doc['site_data']['avs30_m/s'],
                'fc_hp': fc_hp,
                'pred_fc_hp': doc['processed']['fc_hp'][ch]['m4'],
                'dif_fc_hp': abs(fc_hp - doc['processed']['fc_hp'][ch]['m4']),
                
            }

            
            dataset.append(data)

    df = pd.DataFrame(dataset)

    #sort by record_duration ascending
    df = df.sort_values(by='dif_fc_hp', ascending=True)
    
    # count how many fc_hp is greater than 0.4
    count_fc_hp_gt_04 = len(df[df['fc_hp'] >= 0.4])
    print(f"Total records with fc_hp >= 0.4: {count_fc_hp_gt_04}")
    
    # # save to CSV
    # output_path = os.path.join('data', 'jp_dataset_ch_fc_hp.csv')
    # df.to_csv(output_path, index=False)

def generate_fc_hp_2D(data_name, df, mask_flag=False):
    fq_list = np.arange(0.005, 1.28, 0.005)
    time_len = 1024
    
    collection_ts = client['JP']['time_series']
    print(f"Total records to process: {len(df)}")
    df['processing_time'] = 0.0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        start_time = time.time()    
        dt = row['dt']
        record_duration = row['record_duration']
        start_time = row['original_start_time']
        end_time = row['original_end_time']
        fc_lp = 30.0

        if mask_flag:
            p_time = row['P_wave_arrival']
            # create mask
            mask = np.zeros((len(fq_list), time_len))

            # Frequency index (convert Hz to index)
            fc_index = int(row['fc_hp'] / 0.005)

            # Time index (scale p_time to time_len samples)
            p_index = int((p_time / record_duration) * time_len)

            # Set center pixel to 1.0
            if 0 <= fc_index < mask.shape[0] and 0 <= p_index < mask.shape[1]:
                mask[fc_index, p_index] = 1.0

            # Apply Gaussian blur with sigma = 3
            mask = gaussian_filter(mask, sigma=3)

            # Normalize mask to ensure maximum value = 1.0
            mask /= mask.max()

            # Save mask as .npy
            output_mask_path = os.path.join('data', data_name, f"mask/{row['id']}.npy")
            np.save(output_mask_path, mask)

        else:
            # get one document
            record_id_base = row['id'][:-3]
            ch = row['id'][-2:]
            query = {'_id': record_id_base}
            doc_ts = collection_ts.find_one(query, projection={'original.acc_gal': 1})

            if doc_ts:
                tr_base = np.array(doc_ts['original']['acc_gal'][ch])

                start_index = int(start_time / dt)
                end_index = start_index + int(record_duration / dt)
                tr_base = tr_base[start_index:end_index]

                record_dur = dt * len(tr_base)
                # check if record_dur is not equal to record_duration
                if abs(record_dur - record_duration) > dt:
                    print(f"Record duration mismatch for {row['id']}: calculated {record_dur}, expected {record_duration}")
                    continue

                data_array_2d = np.zeros((len(fq_list), time_len))

                # check if npy file already exists
                output_path = os.path.join('data', data_name, f"image/{row['id']}.npy")
                if os.path.exists(output_path):
                    continue
                else:
                    for index_fq, fc_hp in enumerate(fq_list):
                        tr = tr_base.copy()
                        acc, vel, dis = tkcore.quake.process.process_record_peer(tr, dt, fc_hp, fc_lp, acausal=True)
                        time_array = np.arange(len(dis)) * dt

                        dis = reduce_peaks(dis, percentile=99, limit_norm=0.95)

                        # resampling
                        time_new = np.linspace(time_array.min(), time_array.max(), time_len)
                        dis = np.interp(time_new, time_array, dis)

                        # normalize
                        dis = dis / np.max(np.abs(dis))
                        data_array_2d[index_fq, :] = dis

                    # save matrix
                    np.save(output_path, data_array_2d)
            
            else:
                print(f"Document with id {record_id_base} not found.")

        end_time = time.time()
        df.at[index, 'processing_time'] = end_time - start_time
    
    # print max, min, average processing time
    print(f"Max processing time: {df['processing_time'].max()} seconds")
    print(f"Min processing time: {df['processing_time'].min()} seconds")
    print(f"Average processing time: {df['processing_time'].mean()} seconds")

def review_fc_hp_2D():

    tkcore.plot.base.set_latex_style(fontsize=10, linewidth=0.7)

    df = pd.read_csv('/home/italoif/gmr_picker/models/HP_DeepLabV3Plus_03/test_results.csv')
    data_name = 'jp_dataset_2D'

    # df['fc_hp_dif'] = abs(df['pred_fc_hp'] - df['fc_hp'])
    # df =df[df['fc_hp_dif'] > 0.1]
    df = df[df['id'].str[:-3] == '20030926060800_KIK_HDKH04_00']
    # print(f"Total records to review: {len(df)}")
    
    df['dis_cm'] = 0.0
    df['pred_dis_cm'] = 0.0
    
    collection_ts = client['JP']['time_series']
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        dt = row['dt']
        record_duration = row['record_duration']
        start_time = row['original_start_time']
        end_time = row['original_end_time']
        p_time = row['P_wave_arrival']
        fh = 30.0

        # get one document
        record_id_base = row['id'][:-3]
        ch = row['id'][-2:]
        query = {'_id': record_id_base}
        doc_ts = collection_ts.find_one(query, projection={'original.acc_gal': 1})

        if doc_ts:
            tr_base = np.array(doc_ts['original']['acc_gal'][ch])

            start_index = int(start_time / dt)
            end_index = start_index + int(record_duration / dt)
            tr_base = tr_base[start_index:end_index]
            tr_base = tr_base - np.mean(tr_base)

            # make subplot of 3 row and 1 column
            fig, axs = plt.subplots(5, 1, figsize=(10, 10))

            # fist acc_gal
            ax = axs[0]
            time_array = np.arange(len(tr_base)) * dt
            peak_offset = np.max(np.abs(tr_base)) * 1.1
            ax = tkcore.plot.base.set_ax_format(
                ax,
                xlim=[0, record_duration],
                ylim=[-peak_offset, peak_offset],
                xticks=tkcore.plot.gm.get_ticks_values([0, record_duration], 8)[0],
                yticks=tkcore.plot.gm.get_ticks_values([-peak_offset, peak_offset], 5)[0],
                fmtx='%.0f',
                fmty='%.0f',
            )
            ax.plot(time_array, tr_base, color='black', linewidth=0.7)
            # add vertical line for predictected P wave arrival and true P wave arrival
            ax.axvline(x=row['pred_P_wave_arrival'], color='red', linestyle='--', linewidth=0.7)
            ax.axvline(x=p_time, color='blue', linestyle='--', linewidth=0.7)

            # second 2D image filtering fy fc_hp
            tr = tr_base.copy()
            time_taper = max(0.5, 0.01 * dt * len(tr))
            tr = tkcore.quake.filter.zero_baseline(tr, dt, type = 'zero-th-1', zeroth = time_taper)
            tr = tkcore.quake.filter.cosine_taper(tr, dt, time_taper, 'both')
            tr = tkcore.quake.filter.butterworth_bandpass(tr, dt, fl=row['pred_fc_hp'], fh=fh, n=5, padding=True)

            vel, dis = tkcore.quake.integration.integrate_linear_acceleration(tr, dt)
            time_array = np.arange(len(dis)) * dt
            p = np.polyfit(time_array, dis, 6)
            r_dis = np.polyval(p, time_array)
            dis = dis - r_dis
            dis_cm = np.max(np.abs(dis))
            df.at[index, 'pred_dis_cm'] = dis_cm

            ax = axs[1]
            peak_offset = np.max(np.abs(dis)) * 1.1
            yticks, fmty = tkcore.plot.gm.get_ticks_values([-peak_offset, peak_offset], 5)
            ax = tkcore.plot.base.set_ax_format(
                ax,
                xlim=[0, record_duration],
                ylim=[-peak_offset, peak_offset],
                xticks=tkcore.plot.gm.get_ticks_values([0, record_duration], 8)[0],
                yticks=yticks,
                fmtx='%.0f',
                fmty=fmty
            )
            ax.plot(time_array, dis, color='red', linewidth=0.7)

            # third 2D image from true fc_hp
            tr = tr_base.copy()
            time_taper = max(0.5, 0.01 * dt * len(tr))
            tr = tkcore.quake.filter.zero_baseline(tr, dt, type = 'zero-th-1', zeroth = time_taper)
            tr = tkcore.quake.filter.cosine_taper(tr, dt, time_taper, 'both')
            tr = tkcore.quake.filter.butterworth_bandpass(tr, dt, fl=row['fc_hp'], fh=fh, n=5, padding=True)
            vel, dis = tkcore.quake.integration.integrate_linear_acceleration(tr, dt)
            time_array = np.arange(len(dis)) * dt
            p = np.polyfit(time_array, dis, 6)
            r_dis = np.polyval(p, time_array)
            dis = dis - r_dis
            dis_cm = np.max(np.abs(dis))
            df.at[index, 'dis_cm'] = dis_cm
            

            ax = axs[2]
            peak_offset = np.max(np.abs(dis)) * 1.1
            yticks, fmty = tkcore.plot.gm.get_ticks_values([-peak_offset, peak_offset], 5)
            ax = tkcore.plot.base.set_ax_format(
                ax,
                xlim=[0, record_duration],
                ylim=[-peak_offset, peak_offset],
                xticks=tkcore.plot.gm.get_ticks_values([0, record_duration], 8)[0],
                yticks=yticks,
                fmtx='%.0f',
                fmty=fmty
            )
            ax.plot(time_array, dis, color='blue', linewidth=0.7)

            # fourth 2D image
            img_path = os.path.join('data', data_name, f"image/{row['id']}.npy")
            data_2d = np.load(img_path).astype(np.float32)

            ax = axs[3]
            ax = tkcore.plot.base.set_ax_format(
                ax,
                xlim=[0, data_2d.shape[1]],
                ylim=[data_2d.shape[0], 0],
                xticks=tkcore.plot.gm.get_ticks_values([0, data_2d.shape[1]], 7)[0],
                yticks=tkcore.plot.gm.get_ticks_values([0, data_2d.shape[0]], 7)[0],
                fmtx='%.0f',
                fmty='%.0f',
            )
            im = ax.imshow(data_2d, aspect='auto', cmap='seismic', extent=[0, data_2d.shape[1], 0, data_2d.shape[0]], origin='lower', vmin=-1.0, vmax=1.0)
            pred_fc_hp_pixel = row['pred_fc_hp_index']
            pred_p_wave_pixel = row['pred_P_wave_arrival_index']
            ax.scatter(pred_p_wave_pixel, pred_fc_hp_pixel, marker='o', edgecolors='white', color='red', s=50, label='Predicted fc_hp')
            
            fc_hp_pixel = row['fc_hp_index']
            p_wave_pixel = row['P_wave_arrival_index']
            ax.scatter(p_wave_pixel, fc_hp_pixel, marker='o', edgecolors='white', color='blue', s=50, label='True fc_hp')
            
            #  fifth mask image
            mask_pred_path = os.path.join('/home/italoif/gmr_picker/models/HP_DeepLabV3Plus_03/out', f"pred_{row['id']}.npy")
            mask_data = np.load(mask_pred_path).astype(np.float32)
            ax = axs[4]
            ax = tkcore.plot.base.set_ax_format(
                ax,
                xlim=[0, mask_data.shape[1]],
                ylim=[mask_data.shape[0], 0],
                xticks=tkcore.plot.gm.get_ticks_values([0, mask_data.shape[1]], 7)[0],
                yticks=tkcore.plot.gm.get_ticks_values([0, mask_data.shape[0]], 7)[0],
                fmtx='%.0f',
                fmty='%.0f',
            )
            im = ax.imshow(mask_data, aspect='auto', cmap='plasma', extent=[0, mask_data.shape[1], 0, mask_data.shape[0]], origin='lower', vmin=0.0, vmax=1.0)

            tkcore.plot.base.save_fig(fig, f'data/{data_name}/review/{row["id"]}.png', dpi=300)
        
    # save updated dataframe
    df.to_csv('data/jp_dataset_ch_fc_hp_dis.csv', index=False)

def update_fc_hp_in_df():
    df = pd.read_csv('/home/italoif/gmr_picker/models/HP_DeepLabV3Plus_aug/test_results.csv')

    # if fc_hp is zero or greater than 0.425 set pred_fc_hp to fc_hp
    for index, row in df.iterrows():
        if row['fc_hp'] == 0.0 or row['fc_hp'] > 0.3:
            df.at[index, 'fc_hp'] = row['pred_fc_hp']

    # compute fc_hp_index and p_wave_index
    df['fc_hp_index'] = (df['fc_hp'] / 0.005).astype(int)
    df['P_wave_arrival_index'] = ((df['P_wave_arrival'] / df['record_duration']) * 1024).astype(int)
    # sort by 'fc_hp'
    df = df.sort_values(by='fc_hp', ascending=False)
    df.to_csv('data/jp_dataset_ch_fc_hp_updated.csv', index=False)

def make_new_csv():
    query = {
        'processing_flags.labeler_tool': {'$eq': True},
        'quality_flags.do_not_use': {'$ne': True},
        'original.PGA_gal.HM' : {'$gte': 20.0},
        'processed.im': {'$exists': False},
        # _id start with 202512
        '$or': [
            {'_id': {'$regex': '^202512'}},
            {'_id': {'$regex': '^202611'}},
        ],
        # add if time_markers.P_wave_arrival exists check greater than 5.0 or equal to None
        '$or': [
            {'time_markers.P_wave_arrival': {'$gt': 5.0}},
            {'time_markers.P_wave_arrival': {'$eq': None}},
        ]
    }

    db = client['JP']
    collection = db['metadata']

    # count total documents
    total_docs = collection.count_documents(query)
    print(f"Total documents to process: {total_docs}")
    
    docs = collection.find(query)
    dataset = []
    for doc in docs:
        start_time = 0.0 if pd.isna(doc['time_markers']['start_time']) else doc['time_markers']['start_time']
        end_time = 0.0 if pd.isna(doc['time_markers']['end_time']) else doc['time_markers']['end_time']
        
        start_time = round(start_time, 0)
        end_time = round(end_time, 0)

        if end_time > 0.0:
            record_duration = end_time - start_time
        else:
            record_duration = doc['original']['dt'] * doc['original']['npts'] - start_time

        for ch in ['UD', 'NS', 'EW']:
            data = {
                'id': str(doc['_id']) + f"_{ch}",
                'dt': doc['original']['dt'],
                'original_start_time' : start_time,
                'original_end_time' : end_time,
                'record_duration': record_duration,
                'mag': doc['event_data']['magnitude'],
            }
            dataset.append(data)
    
    df = pd.DataFrame(dataset)
    
    output_path = os.path.join('data', 'jp_dataset_2026.csv')
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    df = pd.read_csv(os.path.join('data', 'jp_dataset_2026.csv'))
    data_name = 'jp_dataset_2D'
    generate_fc_hp_2D(data_name, df, mask_flag=False)