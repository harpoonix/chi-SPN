import os
import pickle

import torch

from ciSPN.models.CISPN import CiSPN
from ciSPN.models.model_creation import create_nn_for_spn


def load_model_params(model, load_dir, file_name="nn.model"):
    # load_dir = conf.load_dir
    with open(load_dir / file_name, "rb") as f:
        model.load_state_dict(torch.load(f))
    model.cuda()


def save_model_params(save_dir, model, file_name="nn.model"):
    torch.save(model.state_dict(), save_dir / file_name)


def save_spn(save_dir, spn, args, rg, file_name="spn.model"):
    #save_dir = conf.ckpt_dir

    with open(save_dir / "regionGraph.pkl", "wb") as f:
        pickle.dump(rg, f)

    with open(save_dir / "args.pkl", "wb") as f:
        pickle.dump(args, f)

    save_model_params(save_dir, spn, file_name=file_name)


def load_spn(num_condition_vars, load_dir, discrete_ids, file_name="spn.model", print_spn_info=True,
              nn_provider=None, nn_provider_args=None):
    # load_dir = conf.load_dir

    with open(load_dir / "args.pkl", "rb") as f:
        args = pickle.load(f)

    # build spn graph
    with open(load_dir / "regionGraph.pkl", "rb") as f:
        rg = pickle.load(f)

    spn = CiSPN(rg, args, discrete_ids=discrete_ids).cuda()
    num_leaf_params, num_sum_params = spn.num_parameters()

    if nn_provider is not None:
        if nn_provider_args is None:
            nn_provider_args = {}
        nn = nn_provider(spn, num_condition_vars, num_sum_params, num_leaf_params, **nn_provider_args)
    else:
        nn = create_nn_for_spn(num_condition_vars, num_sum_params, num_leaf_params)
    spn.set_nn(nn)

    if print_spn_info:
        spn.print_structure_info()

    load_model_params(spn, load_dir, file_name)

    return spn, rg, args
