import math
import torchvision
from ciSPN.E1_helpers import get_experiment_name
from ciSPN.E2_helpers import create_cnn_for_spn, create_cnn_model, img_batch_processor
from ciSPN.gradcam.gradcam import GradCam
from ciSPN.models.nn_wrapper import NNWrapper
from helpers.determinism import make_deterministic
import numpy as np

import torch
from datasets.hiddenObjectDataset import HiddenObjectDataset
from helpers.configuration import Config
import argparse
from matplotlib import cm

from environment import environment, get_dataset_paths

from models.spn_create import load_spn, load_model_params
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)
parser.add_argument("--model", choices=["cnn", "ciCNNSPN"], default='cnn')
parser.add_argument("--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default='MSELoss')
parser.add_argument("--sample_id", type=int)
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument("--loss2_factor", default="1.0")  # factor by which loss2 is added to the loss term
parser.add_argument("--dataset", choices=["hiddenObject"], default="hiddenObject")
parser.add_argument("--provide_interventions", choices=["true", "false"], default="true")  # provide intervention vector during training
parser.add_argument("--inspect_output", type=int, default=None)
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.model_name = cli_args.model
conf.loss_name = cli_args.loss
conf.sample_id = cli_args.sample_id
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed
conf.provide_interventions = (cli_args.provide_interventions.lower() == "true")
conf.inspect_output = cli_args.inspect_output

conf.explicit_load_part = conf.model_name

if __name__ == "__main__":

    make_deterministic(conf.seed, deterministic_cudnn=False)

    if conf.loss2_name is not None:
        raise RuntimeError("Not supported yet")  # see fixme for gradcam_path below

    intervention_str = "" if conf.provide_interventions else "_noInterventions"

    # setup experiments folder
    runtime_base_dir = environment["experiments"]["base"] / "E2" / "runtimes"
    gradcam_dir = environment["experiments"]["base"] / "E2" / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    partial_str = f"_part{conf.inspect_output}" if conf.inspect_output is not None else ""
    gradcam_path = gradcam_dir / f"{conf.sample_id}_{conf.model_name}_{conf.loss_name}{intervention_str}{partial_str}.png" # FIXME add loss2 and factor to path if needed

    experiment_name = get_experiment_name(conf.dataset, conf.model_name, conf.seed, conf.loss_name, conf.loss2_name,
                                          conf.loss2_factor, E=2, provide_interventions=conf.provide_interventions)
    load_dir = runtime_base_dir / experiment_name

    print("Arguments:", cli_args)

    # setup dataset
    if cli_args.dataset == "hiddenObject":
        dataset_split = "test"
        hidden_object_base_dir = get_dataset_paths("hiddenObject", dataset_split, get_base=True)
        dataset = HiddenObjectDataset(hidden_object_base_dir, split=dataset_split,
                                      add_intervention_channel=conf.provide_interventions)
    else:
        raise RuntimeError(f"Unknown dataset ({cli_args.dataset}).")

    num_condition_vars = dataset.num_observed_variables
    num_target_vars = dataset.num_hidden_variables
    nn_provider_args = {"num_channels": 4 if conf.provide_interventions else 3}


    print(f"Loading {conf.explicit_load_part}")
    if conf.explicit_load_part == 'ciCNNSPN':
        spn, _, _ = load_spn(num_condition_vars, load_dir=load_dir, nn_provider=create_cnn_for_spn, nn_provider_args=nn_provider_args)
        eval_wrapper = spn
    elif conf.explicit_load_part == 'cnn':
        nn = create_cnn_model(num_condition_vars, num_target_vars, **nn_provider_args)
        load_model_params(nn, load_dir=load_dir)
        eval_wrapper = NNWrapper(nn)
    else:
        raise ValueError(f"invalid load part {conf.explicit_load_part}")
    eval_wrapper.eval()



    batch = dataset[conf.sample_id]
    batch["image"] = batch["image"].clone().detach()  # FIXME use transform
    batch["target"] = torch.tensor(batch["target"])
    condition_batch, target_batch = img_batch_processor(batch)
    condition_batch = condition_batch.unsqueeze(0)  # add batch dimension
    target_batch = target_batch.unsqueeze(0)  # add batch dimension


    grad_cam = GradCam(eval_wrapper)
    cam = grad_cam.generate_cam(condition_batch, target_batch, partial_grad=conf.inspect_output)

    input_image = condition_batch[0, :3, :, :].cpu().numpy()
    input_image = np.moveaxis(input_image, 0, -1)

    cmap = cm.get_cmap('jet')
    color_cam = cam[:, :]
    color_cam = cmap(color_cam)
    color_cam = color_cam[:, :, :3]

    alpha = 0.5
    overlay = (alpha * input_image) + ((1 - alpha) * color_cam)

    print(f"saving to: {gradcam_path}")
    im = Image.fromarray((overlay * 255).astype(np.uint8))
    im.save(gradcam_path)
