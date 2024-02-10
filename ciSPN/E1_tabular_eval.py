from ciSPN.evals.classifyStats import ClassifyStats
from helpers.determinism import make_deterministic
from ciSPN.E1_helpers import get_experiment_name
from datasets.tabularDataset import TabularDataset
from descriptions.description import get_data_description
from models.nn_wrapper import NNWrapper
from datasets.batchProvider import BatchProvider
from helpers.configuration import Config
import torch
import argparse
from ciSPN.models.model_creation import create_nn_model

from environment import environment, get_dataset_paths

from models.spn_create import load_spn, load_model_params

from libs.pawork.log_redirect import PrintLogger

print_progress = True


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606) # 606, 1011, 3004, 5555, 12096
parser.add_argument("--model", choices=["mlp", "ciSPN"], default='mlp')
parser.add_argument("--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default='MSELoss')
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument("--loss2_factor", default="1.0")  # factor by which loss2 is added to the loss term
parser.add_argument("--dataset", choices=["CHC", "ASIA", "CANCER", "EARTHQUAKE", "CHAIN"], default="CHC") # CausalHealthClassification
parser.add_argument("--provide_interventions", choices=["true", "false"], default="true")  # provide intervention vector during training
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.model_name = cli_args.model
conf.batch_size = 8000
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed
conf.provide_interventions = (cli_args.provide_interventions.lower() == "true")

conf.explicit_load_part = conf.model_name


make_deterministic(conf.seed)

# setup experiments folder
runtime_base_dir = environment["experiments"]["base"] / "E1" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E1" / "eval_logs"

experiment_name = get_experiment_name(conf.dataset, conf.model_name, conf.seed, conf.loss_name, conf.loss2_name,
                                      conf.loss2_factor, provide_interventions=conf.provide_interventions)
load_dir = runtime_base_dir / experiment_name

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)

print("Arguments:", cli_args)


# setup dataset
X_vars, Y_vars, interventionProvider = get_data_description(conf.dataset, no_interventions=not conf.provide_interventions)
dataset_paths = get_dataset_paths(conf.dataset, "test")
dataset = TabularDataset(dataset_paths, X_vars, Y_vars, None, part_transformer=interventionProvider)
provider = BatchProvider(dataset, conf.batch_size, provide_incomplete_batch=True)

num_condition_vars = dataset.X.shape[1]
num_target_vars = dataset.Y.shape[1]


print(f"Loading {conf.explicit_load_part}")
if conf.explicit_load_part == 'ciSPN':
    spn, _, _ = load_spn(num_condition_vars, load_dir=load_dir)
    eval_wrapper = spn
elif conf.explicit_load_part == 'mlp':
    nn = create_nn_model(num_condition_vars, num_target_vars)
    load_model_params(nn, load_dir=load_dir)
    eval_wrapper = NNWrapper(nn)
else:
    raise ValueError(f"invalid load part {conf.explicit_load_part}")
eval_wrapper.eval()


with torch.no_grad():
    # test performance on test set, unseen data
    stat = ClassifyStats()

    # zero out target vars, to avoid evaluation errors, if marginalization is not working
    demo_target_batch, demo_condition_batch = provider.get_sample_batch()
    # demo target batch shape: torch.Size([1000, 3])
    placeholder_target_batch = torch.zeros_like(demo_target_batch).cuda()
    marginalized = torch.ones_like(demo_target_batch).cuda()

    i = 0
    while provider.has_data():
        condition_batch, target_batch = provider.get_next_batch()

        reconstruction = eval_wrapper.predict(condition_batch, placeholder_target_batch, marginalized)

        stat.eval(target_batch, reconstruction, i == 0)

        if print_progress:
            print(f"Processed batch {i}.", end="\r")
        i += 1

print(stat.get_eval_result_str())
logger.close()
