
# Updated utils.py
from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.detach().cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32))
    t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32))
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1)
    mask = mask >= lengths.unsqueeze(1)
    return mask

def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)
    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)

# Key Training Script Updates
# Updated train_finetune.py - Key sections showing major changes

# Replace the deprecated torch.gt with modern alternative in length_to_mask
def length_to_mask_updated(lengths):
    """Updated version using modern PyTorch operations"""
    max_len = lengths.max()
    batch_size = lengths.shape[0]
    
    # Create range tensor on same device as lengths
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    # Use >= instead of torch.gt for cleaner code
    mask = mask >= lengths.unsqueeze(1)
    return mask

# Updated DataParallel class with better attribute handling
class ModernDataParallel(torch.nn.DataParallel):
    """Updated DataParallel with better attribute access"""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

# Key changes needed in the training loop:

# 1. Replace .data.cpu() with .detach().cpu()
# 2. Use device-aware tensor creation
# 3. Update loss functions to use reduction parameter explicitly
# 4. Use torch.cuda.amp for mixed precision training (recommended)

# Example of modern mixed precision training setup:
def setup_mixed_precision_training():
    """Modern mixed precision training setup"""
    scaler = torch.cuda.amp.GradScaler()
    return scaler

# Updated loss computation with explicit reduction
def compute_losses_modern(y_pred, y_true):
    """Modern loss computation with explicit parameters"""
    # Use reduction='mean' explicitly instead of relying on defaults
    l1_loss = F.l1_loss(y_pred, y_true, reduction='mean')
    mse_loss = F.mse_loss(y_pred, y_true, reduction='mean')
    
    # For cross entropy, ensure proper shape handling
    if len(y_pred.shape) > 2:
        y_pred = y_pred.view(-1, y_pred.size(-1))
        y_true = y_true.view(-1)
    
    ce_loss = F.cross_entropy(y_pred, y_true, reduction='mean')
    
    return l1_loss, mse_loss, ce_loss

# Modern tensor operations
def modern_tensor_ops(x, device):
    """Examples of modern tensor operations"""
    # Use device parameter in tensor creation
    zeros = torch.zeros_like(x, device=device)
    ones = torch.ones(x.shape, device=device, dtype=x.dtype)
    
    # Use torch.clamp instead of deprecated functions
    clamped = torch.clamp(x, min=0.0, max=1.0)
    
    # Use torch.stack with dim parameter explicitly
    if isinstance(x, list):
        stacked = torch.stack(x, dim=0)
    
    return zeros, ones, clamped

# Updated model saving/loading
def save_checkpoint_modern(model, optimizer, epoch, path):
    """Modern checkpoint saving"""
    state = {
        'model_state_dict': {key: model[key].state_dict() for key in model},
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'torch_version': torch.__version__
    }
    torch.save(state, path)

def load_checkpoint_modern(model, optimizer, path, device):
    """Modern checkpoint loading with device handling"""
    checkpoint = torch.load(path, map_location=device)
    
    for key in model:
        if key in checkpoint['model_state_dict']:
            model[key].load_state_dict(checkpoint['model_state_dict'][key])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0)

# Modern data loading with proper worker handling
def create_modern_dataloader(dataset, batch_size, num_workers=4):
    """Modern dataloader creation"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )

# Updated random number generation
def set_random_seeds(seed=42):
    """Modern random seed setting"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
