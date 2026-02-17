import torch
import segmentation_models_pytorch as smp

def inference_model(input_array):
    device = torch.device('cpu')

    model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b3',        # Choose encoder, e.g. 'resnet34' or 'efficientnet-b0'
            encoder_weights="imagenet",     # Use pre-trained weights (helps convergence)
            in_channels=1,                  # Model input channels (1 for your seismic images)
            classes=1,                      # Model output channels (1 for your heatmap)
            activation='sigmoid',           # IMPORTANT: This adds the Sigmoid layer automatically!
        ).to(device)

    model.load_state_dict(torch.load(f"models/HP_DeepLabV3Plus/best_model.pt", map_location=device))
    model.eval()
    pred = model(torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0).float().to(device))
    return pred.squeeze().cpu().detach().numpy()