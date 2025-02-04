import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_features_from_dataset(slide_seperated_dataset, pretrained_models):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    batch_size = 128
    # original_dataset = LabeledImageDataset("data/candidates", with_index=True)
    # dataset = reduce_dataset(original_dataset, discard_ratio=0.0)
    dataset = slide_seperated_dataset
    for ModelClass in pretrained_models:
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = ModelClass(hidden_layers=0)
        model.to(device)

        output_csv_path = f"output/{model.__class__.__name__}_{model.pretrained_output_size}_features.csv"
        # Open the CSV file and write header (if needed)
        with open(output_csv_path, mode='w') as f:
            header = ','.join([f'feature_{i}' for i in range(model.pretrained_output_size)] + ["label", "file_path"])
            f.write(header + '\n')

        # Stream-writing each batch to the CSV file
        file_paths = np.array(dataset.file_paths)
        with torch.no_grad(), open(output_csv_path, mode='a') as f:
            for batch_x, batch_y, idx in tqdm(dataset_loader):
                batch_x = batch_x.to(device)
                logits = model.pretrained_model.forward(batch_x)

                # Move logits to CPU, detach, and convert to numpy
                logits = logits.cpu().detach().numpy()

                # Convert logits to DataFrame and write to CSV in append mode
                batch_df = pd.DataFrame(logits)
                batch_df['label'] = batch_y
                batch_df['file_path'] = file_paths[idx]
                batch_df.to_csv(f, header=False, index=False)
