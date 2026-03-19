import torch
from frame_utils import AVHevcDataset, DaliHevcDataset

DatasetClass = DaliHevcDataset if torch.cuda.is_available() else AVHevcDataset
