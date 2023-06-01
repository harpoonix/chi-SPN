from helpers.determinism import make_deterministic
from ciSPN.E1_helpers import get_experiment_name

from libs.pawork.log_redirect import PrintLogger

from datasets.tabularDataset import TabularDataset
from descriptions.description import get_data_description
from helpers.configuration import Config
import numpy as np
import argparse
from lgnpy import LinearGaussian
import pandas as pd
import regex

from environment import environment, get_dataset_paths

import warnings

print("ok")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["CHC"], default="CHC")
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
    if dataset_name != 'CHC':
        raise ValueError("unsupported dataset")

    if dataset_name == "CHC":
        lg = LinearGaussian()

        edges = [
            ("A", "F"),
            ("A", "H"),
            ("F", "H"),
            ("H", "M"),
            ("A", "D1"),
            ("A", "D2"),
            ("A", "D3"),
            ("F", "D1"),
            ("F", "D2"),
            ("F", "D3"),
            ("H", "D1"),
            ("H", "D2"),
            ("H", "D3"),
            ("M", "D1"),
            ("M", "D2"),
            ("M", "D3")
        ]

        if intervention is not None:
            # 'cut' all edges to parents
            edges = list(filter(lambda t: t[1] != intervention, edges))

        lg.set_edges_from(edges)
    else:
        raise ValueError("unknown dataset")

    return lg


def validate(x):
    # sometimes lgnpy returns strings ...
    val = [e if isinstance(e, (int, float)) else 0 for e in x]
    return val


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

    # put cond and class vars back together
    data = {
        **{n: dataset_train.X[:, i] for i, n in enumerate(X_vars)},
        **{n: dataset_train.Y[:, i] for i, n in enumerate(Y_vars)}
    }
    data = pd.DataFrame(data)
    bn.set_data(data)

    prediction = np.empty(dataset_test.Y.shape, dtype=np.int)
    num_samples = len(dataset_test.X)

    # Supressing:
    # LinearGaussian.py:106: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'The pandas.np module is deprecated', category=FutureWarning)

        for i in range(num_samples):
            sample = {n: dataset_test.X[i, j] for j, n in enumerate(X_vars)}  # create dict with evidence
            bn.set_evidences(sample)
            result = bn.run_inference(debug=False)
            pred = [result["Mean_inferred"]["D1"], result["Mean_inferred"]["D2"], result["Mean_inferred"]["D3"]]
            # bn.clear_evidences() # not needed as the next sample will overwrite the sample keys with new evidence
            #pred = np.round(np.array(validate(pred)))
            pred = np.round(validate(pred))
            prediction[i, :] = pred
            if (i+1) % 5000 == 0:
                print(f"Processed sample {i+1}")

    all = np.all(prediction == dataset_test.Y, axis=1)
    correct = np.sum(all)

    num_samples = len(dataset_test.X)
    print(f"Intervention: {intervention_name}")
    print(f"Correct {correct} out of {num_samples} ({correct / num_samples})")
    overall_samples += num_samples
    correct_samples += correct

accuracy = correct_samples / overall_samples
print(f"Total Accuracy: {accuracy}")
