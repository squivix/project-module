import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.LabeledImageDataset import LabeledImageDataset
from utils import reduce_dataset

model = torch.load("pickled_model_20.pth")
dataset = LabeledImageDataset("data/candidates")
dataset = reduce_dataset(dataset, discard_ratio=0.0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 256
data_loader = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=False)
k = batch_size
most_confident_probs = []
most_confident_indexes = []
for i, batch in enumerate(tqdm(data_loader)):
    with torch.no_grad():
        x_test, y_test, batch_index = batch
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        batch_index = batch_index.numpy(force=True)

        test_probs = model.forward(x_test)[:, 0].detach().cpu()
        confident = torch.topk(test_probs, min(test_probs.shape[0], k))
        most_confident_probs.extend(confident[0].numpy(force=True))
        most_confident_indexes.extend(batch_index[confident[1]])

sorted_indexes_values = sorted(zip(most_confident_probs, most_confident_indexes), reverse=True)
sorted_values, sorted_indexes = zip(*sorted_indexes_values)

print(sorted_values)
# print(sorted_indexes)
print([dataset.dataset.file_paths[i][16:] for i in sorted_indexes])
