from helpers.determinism import make_deterministic
from ciSPN.E1_helpers import get_experiment_name

from libs.pawork.log_redirect import PrintLogger

from datasets.tabularDataset import TabularDataset
from descriptions.description import get_data_description
from helpers.configuration import Config
import numpy as np
import argparse
from pgmpy.models import BayesianModel, LinearGaussianBayesianNetwork
import pandas as pd
import regex

from environment import environment, get_dataset_paths

print("ok")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["ASIA", "CANCER", "EARTHQUAKE", "CHAIN"], default="ASIA")
parser.add_argument("--eval_mode", choices=["causal", "correlation"], default="causal")
cli_args = parser.parse_args()

print("Arguments:", cli_args)

conf = Config()
conf.dataset = cli_args.dataset
conf.model_name = "BN"
conf.dataset = cli_args.dataset
conf.mode = cli_args.eval_mode
conf.seed = 0


make_deterministic(conf.seed)

# setup experiments folder
#runtime_base_dir = environment["experiments"]["base"] / "E1" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E1" / "eval_logs"

experiment_name = get_experiment_name(conf.dataset, conf.model_name, conf.seed, None, None, None, specific=conf.mode)

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)


print("Arguments:", cli_args)


# setup dataset
X_vars, Y_vars, interventionProvider = get_data_description(conf.dataset)
X_vars.pop() # remove 'interventions' entry
dataset_paths_train = get_dataset_paths(conf.dataset, "train", no_interventions=conf.mode == "correlation")
dataset_paths_test = get_dataset_paths(conf.dataset, "test")


def create_model(dataset_name, intervention):
    if dataset_name == "CHC":
        raise ValueError("Use BN_CHC_eval")
        #bn = BayesianModel() # isn't viable for continous variables
        bn = LinearGaussianBayesianNetwork()  # Error: "fit method has not been implemented for LinearGaussianBayesianNetwork." ...
        bn.add_edge("A", "F")
        bn.add_edge("A", "H")
        bn.add_edge("F", "H")
        bn.add_edge("H", "M")
        bn.add_edge("A", "D1")
        bn.add_edge("A", "D2")
        bn.add_edge("A", "D3")
        bn.add_edge("F", "D1")
        bn.add_edge("F", "D2")
        bn.add_edge("F", "D3")
        bn.add_edge("H", "D1")
        bn.add_edge("H", "D2")
        bn.add_edge("H", "D3")
        bn.add_edge("M", "D1")
        bn.add_edge("M", "D2")
        bn.add_edge("M", "D3")
    elif dataset_name == "ASIA":
        bn = BayesianModel()
        bn.add_edge("A", "T")
        bn.add_edge("T", "E")
        bn.add_edge("S", "L")
        bn.add_edge("L", "E")
        bn.add_edge("S", "B")
        bn.add_edge("E", "X")
        bn.add_edge("E", "D")
        bn.add_edge("B", "D")
    elif dataset_name == "CANCER":
        bn = BayesianModel()
        bn.add_edge("P", "C")
        bn.add_edge("S", "C")
        bn.add_edge("C", "X")
        bn.add_edge("C", "D")
    elif dataset_name == "EARTHQUAKE":
        bn = BayesianModel()
        bn.add_edge("B", "A")
        bn.add_edge("E", "A")
        bn.add_edge("A", "J")
        bn.add_edge("A", "M")
    elif dataset_name == "CHAIN":
        bn = BayesianModel()
        bn.add_edge("A", "B")
        bn.add_edge("B", "C")
    else:
        raise ValueError("unknown dataset")

    if intervention is not None:
        bn.do([intervention], inplace=True)
    return bn


overall_samples = 0
correct_samples = 0


if conf.mode == "causal":
    # learn a BN for every intervention
    data_paths = zip(dataset_paths_train, dataset_paths_test)
elif conf.mode == "correlation":
    # do a single eval, fitted on the unintervened data
    data_paths = [(dataset_paths_train[0], dataset_paths_test)]
else:
    raise RuntimeError(f"Unknown mode: {conf.mode}")


for (dataset_path_train, dataset_path_test) in data_paths:
    # extract intervention name from path
    if conf.mode == "causal":
        intervention_name = regex.search(r"(?|do\((.*?)\)|(None))", dataset_path_train.name).group(1)
        if intervention_name == 'None':
            intervention_name = None
    else:
        intervention_name = None

    if not isinstance(dataset_path_test, list):
        dataset_path_test = [dataset_path_test]

    # load data - we do not add intervention data, as it is the same within every dataset split anyways
    dataset_train = TabularDataset([dataset_path_train], X_vars, Y_vars, None, store_as_torch_tensor=False) #, part_transformer=interventionProvider)
    dataset_test = TabularDataset(dataset_path_test, X_vars, Y_vars, None, store_as_torch_tensor=False) #, part_transformer=interventionProvider)


    bn = create_model(conf.dataset, intervention_name if conf.mode == "causal" else None)

    n_jobs = -1

    # put cond and class vars back together
    data = {
        **{n: dataset_train.X[:, i] for i, n in enumerate(X_vars)},
        **{n: dataset_train.Y[:, i] for i, n in enumerate(Y_vars)}
    }
    data = pd.DataFrame(data)
    bn.fit(data, n_jobs=n_jobs)


    # put condition vars
    test_data = {
        **{n: dataset_test.X[:, i] for i, n in enumerate(X_vars)},
    }
    test_data = pd.DataFrame(test_data)
    prediction = bn.predict(test_data, stochastic=False, n_jobs=n_jobs)
    prediction = np.vstack([prediction.loc[:, var] for var in Y_vars]).T  # read variables in correct order from frame (to_numpy does not guarantee correct ordering!)

    all = np.all(prediction == dataset_test.Y, axis=1)
    correct = np.sum(all)

    num_samples = len(dataset_test.X)
    print(f"Intervention: {intervention_name}")
    print(f"Correct {correct} out of {num_samples} ({correct/num_samples})")
    overall_samples += num_samples
    correct_samples += correct

accuracy = correct_samples/overall_samples
print(f"Total Accuracy: {accuracy}")
