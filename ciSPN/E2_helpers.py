import numpy as np
import torch
from torch.utils.data import DataLoader

from ciSPN.helpers.determinism import make_deterministic_worker
from ciSPN.models.CNNModel import SimpleCNNModelC
from E1_helpers import get_experiment_name


def get_E2_loss_path(dataset_name, loss_load_seed, provide_interventions=True):
    loss_folder = get_experiment_name(dataset_name, "ciCNNSPN", loss_load_seed, "NLLLoss", None, None, None, E=2, provide_interventions=provide_interventions)
    return loss_folder


def create_dataloader(dataset, seed, num_workers=0, batch_size=100, multi_thread_data_loading=True, shuffle=True, drop_last=True):
    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers if multi_thread_data_loading else 0,
        worker_init_fn=make_deterministic_worker,
        pin_memory=True,
        generator=g,
        drop_last=drop_last,
        persistent_workers=True if multi_thread_data_loading else False,
        prefetch_factor=2 #math.ceil(batch_size/num_workers) if multi_thread_data_loading else 2,
    )
    return dataloader


def img_batch_processor(batch):
    x = batch["image"].to(device="cuda").float()
    y = batch["target"].to(device="cuda").float()  # .float()
    return x, y

def img_batch_processor_np(batch):
    x = batch["image"].astype(np.float)
    y = batch["target"].astype(np.float)
    return x, y


def create_cnn_model(num_condition_vars, num_target_vars, num_channels):
    cnn = SimpleCNNModelC(
        num_sum_weights=None, num_leaf_weights=num_target_vars, num_channels=num_channels)
    return cnn


def create_cnn_for_spn(spn, num_condition_vars, num_sum_params, num_leaf_params, num_channels):
    cnn = SimpleCNNModelC(num_sum_weights=num_sum_params, num_leaf_weights=num_leaf_params, num_channels=num_channels)
    return cnn
