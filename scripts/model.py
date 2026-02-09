import os
import pandas as pd
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

from source.config import load_config, parse_args
from source.dataset import GMRecords2DDataset, GMRecords2DDataset_3CH
from source.metrics import MetricsLogger, best_corner_local_gaussian, calculate_batch_pixel_distance_best
from source.losses import WeightedMSELoss, WeightedMSELoss_3CH

def train_from_config(cfg):
    SEED = cfg.get("SEED", 40)
    
    device = cfg.get("DEVICE", "cpu")

    DATA_DIR = cfg["DATA"]["DIR"]
    DF_PATH = cfg["DATA"]["DF"]
    NUM_WORKERS = cfg["DATA"].get("NUM_WORKERS", 0)
    DATASET_TYPE = cfg["DATA"]["DATASET_TYPE"]
    SPLIT_TYPE = cfg["DATA"].get("SPLIT_TYPE", "random")
    SPLIT_RATIO = cfg["DATA"].get("SPLIT_RATIO", 0.8)

    MODEL_NAME = cfg["MODEL"]["NAME"]
    MODEL_TYPE = cfg["MODEL"]["TYPE"]
    model_dir = f'models/{MODEL_NAME}'
    os.makedirs(model_dir, exist_ok=True)

    LEARNING_RATE = cfg["TRAINING"].get("LR", 1e-4)
    NUM_EPOCHS = cfg["TRAINING"].get("NUM_EPOCHS", 50)
    BATCH_SIZE = cfg["TRAINING"].get("BATCH_SIZE", 8)
    LOSS = cfg["TRAINING"].get("LOSS", "MSELoss")
    OPTIMIZER = cfg["TRAINING"].get("OPTIMIZER", "Adam")
    SIGMA = cfg["TRAINING"].get("MASK_SMOOTH_SIGMA", 5)

    df = pd.read_csv(DF_PATH)
    if SPLIT_TYPE == "dif_record":
        # split the dataset randomkly but ensure that {19960523183600_KNT_MYG002_00}_UD the first part not be in the train and val set, i need all channel if there are more than one will be in the same set
        unique_records = df['id'].str[:-3].unique()
        np.random.shuffle(unique_records)
        split_index = int(len(unique_records) * SPLIT_RATIO)
        train_records = unique_records[:split_index]
        val_records = unique_records[split_index:]
        train_df = df[df['id'].str[:-3].isin(train_records)].reset_index(drop=True)
        val_df = df[df['id'].str[:-3].isin(val_records)].reset_index(drop=True)
        print(f"Train samples: {len(train_df)}")

        # then similar split for test set 0.5 0.5
        test_records = val_records[int(len(val_records)*0.5):]
        val_records = val_records[:int(len(val_records)*0.5)]
        test_df = df[df['id'].str[:-3].isin(test_records)].reset_index(drop=True)
        val_df = df[df['id'].str[:-3].isin(val_records)].reset_index(drop=True)
        print(f"Test samples: {len(test_df)}")
        print(f" Val. samples: {len(val_df)}")

        # save dfs in models dir
        if DATASET_TYPE == "GMRecords2DDataset_3CH":
            # merge each set by id removing _XX and generating fc_hp_{ch} columns for each channel
            def consolidate_df(df):
                """
                Merges rows belonging to the same seismic record (EW, NS, UD) 
                into a single row, preparing it for 3-channel loading.
                """
                # 1. Create a base 'id' column without the channel suffix (e.g., '..._EW' -> '...')
                df['base_id'] = df['id'].str[:-3]
                df['channel'] = df['id'].str[-2:]
                
                # 2. Extract the 'fc_hp_index' for each channel and rename it
                df_pivot = df.pivot(
                    index='base_id', 
                    columns='channel', 
                    values='fc_hp_index'
                ).rename(columns={'EW': 'fc_hp_EW_index', 'NS': 'fc_hp_NS_index', 'UD': 'fc_hp_UD_index'})

                # 3. Get the common, channel-independent columns (like P_wave_arrival_index)
                # Use the UD channel row as the base since all common values should be identical.
                df_common = df[df['channel'] == 'UD'].drop(columns=['fc_hp_index', 'channel', 'id']).set_index('base_id')
                
                # 4. Merge the channel-specific indices with the common data
                df_final = df_common.merge(df_pivot, left_index=True, right_index=True, how='inner').reset_index()
                
                # 5. Rename the base_id column back to 'id' for the 3CH dataset to use
                df_final = df_final.rename(columns={'base_id': 'id'})
                
                return df_final
            
            train_df = consolidate_df(train_df)
            val_df = consolidate_df(val_df)
            test_df = consolidate_df(test_df)

        train_df.to_csv(f"{model_dir}/train_df.csv", index=False)
        val_df.to_csv(f"{model_dir}/val_df.csv", index=False)
        test_df.to_csv(f"{model_dir}/test_df.csv", index=False)

    elif SPLIT_TYPE == "dif_quake":
        quake_key = df['id'].str[:14]
        # set seed for reproducibility
        np.random.seed(SEED)

        # count rows per earthquake
        quake_sizes = (
            df.assign(quake=quake_key)
            .groupby('quake')
            .size()
            .reset_index(name='n_rows')
            .sort_values('n_rows', ascending=False)
        )

        total_rows = len(df)
        target_train = int(total_rows * SPLIT_RATIO)
        target_val   = int((total_rows - target_train) * 0.5)
        target_test  = total_rows - target_train - target_val

        splits = {
            "train": {"target": target_train, "rows": 0, "quakes": []},
            "val":   {"target": target_val,   "rows": 0, "quakes": []},
            "test":  {"target": target_test,  "rows": 0, "quakes": []},
        }

        # greedy assignment
        for _, row in quake_sizes.iterrows():
            quake = row["quake"]
            n = row["n_rows"]

            # choose split that is most underfilled relative to target
            split_name = min(
                splits.keys(),
                key=lambda k: splits[k]["rows"] - splits[k]["target"]
            )

            splits[split_name]["quakes"].append(quake)
            splits[split_name]["rows"] += n

        # build dataframes
        train_df = df[quake_key.isin(splits["train"]["quakes"])].reset_index(drop=True)
        val_df   = df[quake_key.isin(splits["val"]["quakes"])].reset_index(drop=True)
        test_df  = df[quake_key.isin(splits["test"]["quakes"])].reset_index(drop=True)

        # safety checks
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert set(splits["train"]["quakes"]).isdisjoint(splits["val"]["quakes"])
        assert set(splits["train"]["quakes"]).isdisjoint(splits["test"]["quakes"])
        assert set(splits["val"]["quakes"]).isdisjoint(splits["test"]["quakes"])

        print(f"Target rows  -> Train: {target_train}, Val: {target_val}, Test: {target_test}")
        print(f"Actual rows  -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        train_df.to_csv(f"{model_dir}/train_df.csv", index=False)
        val_df.to_csv(f"{model_dir}/val_df.csv", index=False)
        test_df.to_csv(f"{model_dir}/test_df.csv", index=False)

    elif SPLIT_TYPE == "files":
        DF_TRAIN_PATH = cfg["DATA"]["DF_TRAIN"]
        DF_VAL_PATH = cfg["DATA"]["DF_VAL"]
        train_df = pd.read_csv(DF_TRAIN_PATH)
        val_df = pd.read_csv(DF_VAL_PATH)
    else:
        raise NotImplementedError(f"Split type {SPLIT_TYPE} not implemented.")
    
    if DATASET_TYPE == "GMRecords2DDataset":
        train_dataset = GMRecords2DDataset(train_df, DATA_DIR, train=True, augment=True, test_mode=False, mask_smooth_sigma=SIGMA)
        val_dataset = GMRecords2DDataset(val_df, DATA_DIR, train=False, augment=False, test_mode=False, mask_smooth_sigma=SIGMA)
    elif DATASET_TYPE == "GMRecords2DDataset_3CH":
        train_dataset = GMRecords2DDataset_3CH(train_df, DATA_DIR, train=True, augment=True, test_mode=False, mask_smooth_sigma=SIGMA)
        val_dataset = GMRecords2DDataset_3CH(val_df, DATA_DIR, train=False, augment=False, test_mode=False, mask_smooth_sigma=SIGMA)
    else:
        raise NotImplementedError(f"Dataset type {DATASET_TYPE} not implemented.")

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model initialization
    if MODEL_TYPE == "HP_ImageUNet_50":
        # model = HP_ImageUNet(n_channels=1, n_classes=1).to(device)
        model = smp.Unet(
            encoder_name="resnet50",        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)

    elif MODEL_TYPE == "HP_UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name="resnet50",        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
            # decoder_attention_type="scse"
        ).to(device)

    elif MODEL_TYPE == "HP_FPN":
        model = smp.FPN(
            encoder_name="efficientnet-b3",        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)

    elif MODEL_TYPE == "HP_DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b3',        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)

    elif MODEL_TYPE == "HP_DeepLabV3Plus_3CH":
        model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b3',        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights=None,     # Use pre-trained weights (helps convergence)
            in_channels=3,                  # Model input channels (3 for your seismic images with 3 channels)
            classes=3,                      # Model output channels (3 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)
    else:
        raise NotImplementedError(f"Model {MODEL_TYPE} not implemented.")
    
    # Loss function
    if LOSS == "BCELoss":
        loss_fn = torch.nn.BCELoss()
    elif LOSS == "MSELoss":
        loss_fn = torch.nn.MSELoss()
    elif LOSS == "WeightedMSELoss":
        loss_fn = WeightedMSELoss(positive_weight=100)
    elif LOSS == "WeightedMSELoss_3CH":
        loss_fn = WeightedMSELoss_3CH(positive_weight=100)
    else:
        raise NotImplementedError(f"Loss function {LOSS} not implemented.")

    # Optimizer
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,   
            patience=5,     
        )
    else:
        raise NotImplementedError(f"Optimizer {OPTIMIZER} not implemented.")

    # Training loop
    best_score = float('inf')
    metrics_logger = MetricsLogger(filepath=f'{model_dir}/metrics.csv')

    for epoch in range(NUM_EPOCHS):
        model.train()

        train_loss_sum = 0.0
        train_samples = 0
        train_pixel_error_sum = 0.0

        with tqdm(train_dataset_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", total=len(train_dataset_loader)) as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)
                batch_size = images.size(0)

                # Forward pass
                preds = model(images) # Output is 0-1 heatmap
                loss = loss_fn(preds, masks)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # accumulate
                train_loss_sum += loss.item() * batch_size
                train_samples += batch_size

                # --- CALCULATE METRIC (Pixel Distance) ---
                if MODEL_TYPE == "HP_DeepLabV3Plus_3CH":
                    preds_detached = preds.detach()
                    batch_dist_sum_EW, _ = calculate_batch_pixel_distance_best(preds_detached[:, 0:1, :, :], masks[:, 0:1, :, :], sigma=SIGMA)
                    batch_dist_sum_NS, _ = calculate_batch_pixel_distance_best(preds_detached[:, 1:2, :, :], masks[:, 1:2, :, :], sigma=SIGMA)
                    batch_dist_sum_UD, batch_count = calculate_batch_pixel_distance_best(preds_detached[:, 2:3, :, :], masks[:, 2:3, :, :], sigma=SIGMA)
                    
                    # Average the three distances
                    batch_dist_sum = (batch_dist_sum_EW + batch_dist_sum_NS + batch_dist_sum_UD) / 3.0
                else:
                        batch_dist_sum, batch_count = calculate_batch_pixel_distance_best(preds, masks, sigma=SIGMA)
                train_pixel_error_sum += batch_dist_sum

                # current average loss (so far)
                current_avg = train_loss_sum / train_samples
                current_avg_pixel_error = train_pixel_error_sum / train_samples
                pbar.set_postfix({"Train Loss": f"{current_avg:.6f}", "Train PxErr": f"{current_avg_pixel_error:.2f} px"})

        # final train loss averaged over dataset
        train_loss = train_loss_sum / len(train_dataset_loader.dataset)
        train_avg_pixel_error = train_pixel_error_sum / len(train_dataset_loader.dataset)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        total_pixel_error = 0.0

        with torch.no_grad():
            with tqdm(val_dataset_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", total=len(val_dataset_loader)) as pbar:
                for batch_idx, (images, masks) in enumerate(pbar):
                    images = images.to(device)
                    masks = masks.to(device)
                    batch_size = images.size(0)

                    # Forward pass
                    preds = model(images)
                    loss = loss_fn(preds, masks)

                    # Accumulate Loss
                    val_loss_sum += loss.item() * batch_size
                    val_samples += batch_size

                    # --- CALCULATE METRIC (Pixel Distance) ---
                    if MODEL_TYPE == "HP_DeepLabV3Plus_3CH":
                        preds_detached = preds.detach()
                        batch_dist_sum_EW, _ = calculate_batch_pixel_distance_best(preds_detached[:, 0:1, :, :], masks[:, 0:1, :, :], sigma=SIGMA)
                        batch_dist_sum_NS, _ = calculate_batch_pixel_distance_best(preds_detached[:, 1:2, :, :], masks[:, 1:2, :, :], sigma=SIGMA)
                        batch_dist_sum_UD, batch_count = calculate_batch_pixel_distance_best(preds_detached[:, 2:3, :, :], masks[:, 2:3, :, :], sigma=SIGMA)
                        
                        # Average the three distances
                        batch_dist_sum = (batch_dist_sum_EW + batch_dist_sum_NS + batch_dist_sum_UD) / 3.0
                    else:
                        batch_dist_sum, batch_count = calculate_batch_pixel_distance_best(preds, masks, sigma=SIGMA)
                    total_pixel_error += batch_dist_sum

                    # Update Progress Bar
                    cur_val_loss = val_loss_sum / val_samples
                    cur_avg_pixel_error = total_pixel_error / val_samples
                    
                    pbar.set_postfix({
                        "Val Loss": f"{cur_val_loss:.6f}",
                        "Avg Pixel Err": f"{cur_avg_pixel_error:.2f} px"
                    })

        # Finalize validation metrics
        val_loss = val_loss_sum / len(val_dataset_loader.dataset)
        avg_pixel_error = total_pixel_error / len(val_dataset_loader.dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Avg Train Pixel Error: {train_avg_pixel_error:.2f} px | Val Loss: {val_loss:.6f} | Avg Val Pixel Error: {avg_pixel_error:.2f} px")
        scheduler.step(avg_pixel_error)
        # --- LOGGING & SAVING ---
        
        # Log metrics (Assuming your logger accepts **kwargs)
        metrics_logger.log(
            epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            train_pixel_error=train_avg_pixel_error,
            val_pixel_error=avg_pixel_error
        )

        # Save best model
        # Logic: We want the LOWEST pixel error
        if avg_pixel_error < best_score:
            old_best = best_score
            best_score = avg_pixel_error
            torch.save(model.state_dict(), f"{model_dir}/best_model.pt")
            print(f"✅ Saved new best model (Error improved from {old_best:.2f} to {best_score:.2f} px)")
            
def test_from_config(cfg):
    device = cfg.get("DEVICE", "cpu")
    # load model and best pth
    MODEL_NAME = cfg["MODEL"]["NAME"]
    MODEL_TYPE = cfg["MODEL"]["TYPE"]
    DATASET_TYPE = cfg["DATA"]["DATASET_TYPE"]

    model_dir = f'models/{MODEL_NAME}'
    if MODEL_TYPE == "HP_ImageUNet_50":
        # model = HP_ImageUNet(n_channels=1, n_classes=1).to(device)
        model = smp.Unet(
            encoder_name="resnet50",        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)
    elif MODEL_TYPE == "HP_ImageUNet_enb3":
        model = smp.Unet(
            encoder_name="efficientnet-b3",        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
            decoder_attention_type="scse"
        ).to(device)
    elif MODEL_TYPE == "HP_DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b3',        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)
    elif MODEL_TYPE == "HP_DeepLabV3Plus_res":
        model = smp.DeepLabV3Plus(
            encoder_name='resnet50',        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)
    elif MODEL_TYPE == "HP_FPN":
        model = smp.FPN(
            encoder_name="efficientnet-b3",        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)
    elif MODEL_TYPE == "HP_UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
            # decoder_attention_type="scse"
        ).to(device)
    elif MODEL_TYPE == "HP_DeepLabV3Plus_3CH":
        model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b3',        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights=None,     # Use pre-trained weights (helps convergence)
            in_channels=3,                  # Model input channels (3 for your seismic images with 3 channels)
            classes=3,                      # Model output channels (3 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)
    else:
        raise NotImplementedError(f"Model {MODEL_TYPE} not implemented.")
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", map_location=device))
    model.eval()

    # Testing parameters
    DATA_DIR = cfg["TESTING"]["DIR"]
    DF_PATH = cfg["TESTING"]["DF"]
    BATCH_SIZE = cfg["TESTING"].get("BATCH_SIZE", 8)
    SIGMA = cfg["TESTING"].get("MASK_SMOOTH_SIGMA", 5)

    df = pd.read_csv(DF_PATH)
    if DATASET_TYPE == "GMRecords2DDataset":
        test_dataset = GMRecords2DDataset(df, DATA_DIR, train=False, augment=False, test_mode=True)
    elif DATASET_TYPE == "GMRecords2DDataset_3CH":
        test_dataset = GMRecords2DDataset_3CH(df, DATA_DIR, train=False, augment=False, test_mode=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    #  make output dir
    os.makedirs(f"{model_dir}/out", exist_ok=True)

    if DATASET_TYPE == "GMRecords2DDataset_3CH":
        df['pred_fc_hp_EW_index'] = 0
        df['pred_fc_hp_NS_index'] = 0
        df['pred_fc_hp_UD_index'] = 0
        df['pred_P_wave_arrival_index'] = 0
        df['pred_P_wave_arrival'] = 0.0
        df['pred_fc_hp_EW'] = 0.0
        df['pred_fc_hp_NS'] = 0.0
        df['pred_fc_hp_UD'] = 0.0
        with torch.no_grad():
            with tqdm(test_dataset_loader, desc="Testing", total=len(test_dataset_loader)) as pbar:
                for batch_idx, (images, masks) in enumerate(pbar):
                    images = images.to(device)
                    # masks = masks.to(device)

                    preds = model(images)

                    # save probability maps as numpy arrays
                    preds_np = preds.cpu().numpy()
                    for i in range(preds_np.shape[0]):
                        pred_heatmap_EW = preds_np[i,0,:,:]
                        pred_heatmap_NS = preds_np[i,1,:,:]
                        pred_heatmap_UD = preds_np[i,2,:,:]
                        # get record id
                        record_pos_index = batch_idx * BATCH_SIZE + i
                        record_label_index = df.index[record_pos_index]
                        record_id = df.loc[record_label_index]['id']
                        record_duration = df.loc[record_label_index]['record_duration']
                        
                        # EW channel
                        max_coord_EW = best_corner_local_gaussian(pred_heatmap_EW, sigma=SIGMA)
                        time_px_EW = max_coord_EW[1]
                        freq_px_EW = max_coord_EW[0]
                        if freq_px_EW == 0:
                            print(f"Warning: freq_px is 0 for record {record_id} EW channel, adjusting to 1 to avoid zero fc_hp.")
                        pred_p_arrival_time_EW = (time_px_EW / pred_heatmap_EW.shape[1]) * record_duration
                        pred_fc_hp_EW = (freq_px_EW * 0.005)  # assuming freq bin size is 0.005 Hz

                        # NS channel
                        max_coord_NS = best_corner_local_gaussian(pred_heatmap_NS, sigma=SIGMA)
                        time_px_NS = max_coord_NS[1]
                        freq_px_NS = max_coord_NS[0]
                        if freq_px_NS == 0:
                            print(f"Warning: freq_px is 0 for record {record_id} NS channel, adjusting to 1 to avoid zero fc_hp.")
                        pred_p_arrival_time_NS = (time_px_NS / pred_heatmap_NS.shape[1]) * record_duration
                        pred_fc_hp_NS = (freq_px_NS * 0.005)  # assuming freq bin size is 0.005 Hz
                        # UD channel
                        max_coord_UD = best_corner_local_gaussian(pred_heatmap_UD, sigma=SIGMA)
                        time_px_UD = max_coord_UD[1]
                        freq_px_UD = max_coord_UD[0]
                        if freq_px_UD == 0:
                            print(f"Warning: freq_px is 0 for record {record_id} UD channel, adjusting to 1 to avoid zero fc_hp.")
                        pred_p_arrival_time_UD = (time_px_UD / pred_heatmap_UD.shape[1]) * record_duration
                        pred_fc_hp_UD = (freq_px_UD * 0.005)  # assuming freq bin size is 0.005 Hz
                        # average time px
                        avg_time_px = int((time_px_EW + time_px_NS + time_px_UD) / 3)

                        df.loc[record_label_index, 'pred_P_wave_arrival_index'] = avg_time_px
                        df.loc[record_label_index, 'pred_fc_hp_EW_index'] = freq_px_EW
                        df.loc[record_label_index, 'pred_fc_hp_NS_index'] = freq_px_NS
                        df.loc[record_label_index, 'pred_fc_hp_UD_index'] = freq_px_UD
                        df.loc[record_label_index, 'pred_P_wave_arrival'] = (pred_p_arrival_time_EW + pred_p_arrival_time_NS + pred_p_arrival_time_UD) / 3.0
                        df.loc[record_label_index, 'pred_fc_hp_EW'] = pred_fc_hp_EW
                        df.loc[record_label_index, 'pred_fc_hp_NS'] = pred_fc_hp_NS
                        df.loc[record_label_index, 'pred_fc_hp_UD'] = pred_fc_hp_UD

                        # save heatmaps
                        np.save(f"{model_dir}/out/pred{record_id}_EW.npy", pred_heatmap_EW)
                        np.save(f"{model_dir}/out/pred{record_id}_NS.npy", pred_heatmap_NS)
                        np.save(f"{model_dir}/out/pred{record_id}_UD.npy", pred_heatmap_UD)

    else:
        df['pred_fc_hp_index'] = 0
        df['pred_P_wave_arrival_index'] = 0
        df['pred_P_wave_arrival'] = 0.0
        df['pred_fc_hp'] = 0.0
        with torch.no_grad():
            with tqdm(test_dataset_loader, desc="Testing", total=len(test_dataset_loader)) as pbar:
                for batch_idx, (images, masks) in enumerate(pbar):
                    images = images.to(device)
                    # masks = masks.to(device)

                    preds = model(images)

                    # save probability maps as numpy arrays
                    preds_np = preds.cpu().numpy()
                    for i in range(preds_np.shape[0]):
                        pred_heatmap = preds_np[i,0,:,:]
                        # get record id
                        record_pos_index = batch_idx * BATCH_SIZE + i
                        record_label_index = df.index[record_pos_index]
                        record_id = df.loc[record_label_index]['id']
                        record_duration = df.loc[record_label_index]['record_duration']
                        max_coord = best_corner_local_gaussian(pred_heatmap, sigma=SIGMA)
                        # max_coord = np.unravel_index(np.argmax(pred_heatmap, axis=None), pred_heatmap.shape)
                        time_px = max_coord[1]
                        freq_px = max_coord[0]
                        if freq_px == 0:
                            print(f"Warning: freq_px is 0 for record {record_id}, adjusting to 1 to avoid zero fc_hp.")
                        pred_p_arrival_time = (time_px / pred_heatmap.shape[1]) * record_duration
                        pred_fc_hp = (freq_px * 0.005)  # assuming freq bin size is 0.005 Hz
                        df.loc[record_label_index, 'pred_P_wave_arrival_index'] = time_px
                        df.loc[record_label_index, 'pred_fc_hp_index'] = freq_px
                        df.loc[record_label_index, 'pred_P_wave_arrival'] = pred_p_arrival_time
                        df.loc[record_label_index, 'pred_fc_hp'] = pred_fc_hp

                        # print(f"Record: {record_id}, Pred Time Px: {time_px}, Freq Px: {freq_px}")
                        # print(f"Predicted P-wave arrival time: {pred_p_arrival_time}, Predicted fc_hp: {pred_fc_hp}")

                        # save heatmap
                        np.save(f"{model_dir}/out/pred_{record_id}.npy", pred_heatmap)

    # save results in file with same name as df_path but in model_dir and adding res before extension
    out_path = os.path.basename(DF_PATH)
    out_path = os.path.splitext(out_path)[0]  # Remove the file extension
    out_path = os.path.join(model_dir, out_path +  '_out.csv')
    df.to_csv(out_path, index=False)
    # compute MAE, R2 and RMSE for fc_hp and P_wave_arrival
    
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    mae_fc_hp = mean_absolute_error(df['fc_hp'], df['pred_fc_hp'])
    r2_fc_hp = r2_score(df['fc_hp'], df['pred_fc_hp'])
    rmse_fc_hp = root_mean_squared_error(
        df["fc_hp"],
        df["pred_fc_hp"]
    )
    mae_p_wave = mean_absolute_error(df['P_wave_arrival'], df['pred_P_wave_arrival'])
    r2_p_wave = r2_score(df['P_wave_arrival'], df['pred_P_wave_arrival'])
    rmse_p_wave = root_mean_squared_error(
        df["P_wave_arrival"],
        df["pred_P_wave_arrival"]
    )

    # compute Mean localization Error in pixels 2D as sqrt of (dx^2 + dy^2)
    pixel_errors = []
    for index, row in df.iterrows():
        true_x = row['P_wave_arrival_index']
        true_y = row['fc_hp_index']
        pred_x = row['pred_P_wave_arrival_index']
        pred_y = row['pred_fc_hp_index']
        pixel_error = np.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
        pixel_errors.append(pixel_error)
    mean_localization_error = np.mean(pixel_errors)

    print(f"P_wave_arrival_index R2: {r2_p_wave:.4f} MAE: {mae_p_wave:.2f}, RMSE: {rmse_p_wave:.2f}")
    print(f"fc_hp_index R2: {r2_fc_hp:.4f} MAE: {mae_fc_hp:.2f}, RMSE: {rmse_fc_hp:.2f}")
    print(f"Mean 2D Localization Error (pixels): {mean_localization_error:.2f} px")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        train_from_config(cfg)
    elif args.mode == "test":
        test_from_config(cfg)
    elif args.mode == "inference":
        raise NotImplementedError("Inference mode is not implemented yet.")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")