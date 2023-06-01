import argparse
import pickle

import torch

from ciSPN.E3_helpers import create_E3_model, create_E3_score, get_E3_experiment_name
from ciSPN.datasets.tabularDataset import TabularDataset
from descriptions.description import get_data_description
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger
from trainers.decisionTree.DecisionTreeTrainer import BinaryDecisionTreeTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)

parser.add_argument("--model", choices=["DT"], default='DT')
parser.add_argument("--dataset", choices=["ASIA", "CANCER", "EARTHQUAKE", "CHAIN"], default="ASIA")
parser.add_argument("--score", choices=["CausalLossScore", "GiniIndex", "GICL"], default="CausalLossScore")
parser.add_argument("--score_alpha", default=None)
parser.add_argument("--loss_load_seed", type=int, default=None) # is set to seed if none
parser.add_argument("--provide_interventions", choices=["true", "false"], default="true")  # provide intervention vector during training and use inintervened data
cli_args = parser.parse_args()


conf = Config()
conf.model_name = cli_args.model
conf.batch_size = 1000
conf.seed = cli_args.seed
conf.loss_load_seed = cli_args.seed if cli_args.loss_load_seed is None else cli_args.loss_load_seed
conf.dataset = cli_args.dataset
conf.score = cli_args.score
conf.score_alpha = None if cli_args.score_alpha is None else float(cli_args.score_alpha)
conf.provide_interventions = (cli_args.provide_interventions.lower() == "true")



make_deterministic(conf.seed)

# setup experiments folder
runtime_loss_base_dir = environment["experiments"]["base"] / "E1" / "runtimes"
runtime_base_dir = environment["experiments"]["base"] / "E3" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E3" / "logs"


experiment_name = get_E3_experiment_name(conf.dataset, conf.model_name, conf.seed, conf.score, loss_params=conf.score_alpha,
                                         provide_interventions=conf.provide_interventions)
save_dir = runtime_base_dir / experiment_name
save_dir.mkdir(exist_ok=True, parents=True)

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)


print("Arguments:", cli_args)

torch.set_grad_enabled(False)

# setup dataset
X_vars, Y_vars, interventionProvider = get_data_description(conf.dataset, no_interventions=not conf.provide_interventions)
dataset_paths = get_dataset_paths(conf.dataset, "train", no_interventions=not conf.provide_interventions)
dataset = TabularDataset(dataset_paths, X_vars, Y_vars, conf.seed, part_transformer=interventionProvider)

num_condition_vars = dataset.X.shape[1]
num_target_vars = dataset.Y.shape[1]


decision_tree = create_E3_model(conf.model_name)

scorer = create_E3_score(cli_args.score, conf, score_alpha=conf.score_alpha, batch_size=conf.batch_size,
                         num_condition_vars=num_condition_vars, runtime_loss_base_dir=runtime_loss_base_dir, provide_interventions=conf.provide_interventions)

trainer = BinaryDecisionTreeTrainer(scorer)
trainer.fit(decision_tree, dataset)
decision_tree.prune()


decision_tree.clean()
decision_tree.to_torch(to_torch=False)
with open(save_dir / "tree.pkl", "wb") as f:
    pickle.dump(decision_tree, f)

print(f'Final parameters saved to "{save_dir}"')
logger.close()
