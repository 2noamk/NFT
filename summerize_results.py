import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from torch.nn.functional import mse_loss
import pickle
import torch
import sys

sys.path.append('/home/noam.koren/multiTS/NFT/models/NFT/')
from NFT import NFT

sys.path.append('/home/noam.koren/multiTS/NFT/')
from models.training_functions import train_model, plot_predictions

LOOKBACK = 10
HORIZON = 5
NUM_VARS = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "/home/noam.koren/multiTS/NFT/models/NFT/chorales_model_10l_5h_1b.pth"

model_object = NFT().to(device)
model_object.load_state_dict(torch.load(model_path))
model_object.eval()

data_path = "/home/noam.koren/multiTS/NFT/data/chorales/chorales_10l_5h"

def read_pkl_to_torch(file_path, device=None):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    tensor_data = torch.tensor(data, device=device)
    return tensor_data
  
train_X = read_pkl_to_torch(data_path + 'train_X.pkl', device)
train_y = read_pkl_to_torch(data_path + 'train_y.pkl', device)
val_X = read_pkl_to_torch(data_path + 'val_X.pkl', device)
val_y = read_pkl_to_torch(data_path + 'val_y.pkl', device)
test_X = read_pkl_to_torch(data_path + 'test_X.pkl', device)
test_y = read_pkl_to_torch(data_path + 'test_y.pkl', device)


print(f"shape of:\n"
      f"train: X {train_X.shape}, y: {train_y.shape}\n"
      f"val: X {val_X.shape}, y: {val_y.shape}\n"
      f"test: X {test_X.shape}, y: {test_y.shape}")

print(f"The highest value in train is: {torch.max(train_X).item()}")
print(f"The lowest value in the train is: {torch.min(train_X).item()}")

print(f"The highest value in val is: {torch.max(val_X).item()}")
print(f"The lowest value in the val is: {torch.min(val_X).item()}")

print(f"The highest value in test is: {torch.max(test_X).item()}")
print(f"The lowest value in test train is: {torch.min(test_X).item()}")

y_train_pred = torch.tensor(model_object.predict(train_X), device=device).float()
y_val_pred = torch.tensor(model_object.predict(val_X), device=device).float()
y_test_pred = torch.tensor(model_object.predict(test_X), device=device).float()


accumulated_train_loss = mse_loss(y_train_pred, train_y)
accumulated_val_loss = mse_loss(y_val_pred, val_y)
accumulated_test_loss = mse_loss(y_test_pred, test_y)

print(f"accumulated losses:\n"
f"train loss = {accumulated_train_loss}\n"
f"val loss = {accumulated_val_loss}\n"
f"test loss = {accumulated_test_loss}")

train_loss_per_point = (accumulated_train_loss / train_X.shape[0]) / (HORIZON * NUM_VARS)
val_loss_per_point = (accumulated_val_loss / val_X.shape[0]) / (HORIZON * NUM_VARS)
test_loss_per_point = (accumulated_test_loss / test_X.shape[0]) / (HORIZON * NUM_VARS)

print(f"per-point loss:\n"
f"train loss = {train_loss_per_point}\n"
f"val loss = {val_loss_per_point}\n"
f"test loss = {test_loss_per_point}")