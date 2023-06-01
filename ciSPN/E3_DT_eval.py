import argparse

import torch

from ciSPN.E3_helpers import get_E3_experiment_name, load_E3_model
from ciSPN.datasets.tabularDataset import TabularDataset
from ciSPN.libs.pawork.log_redirect import PrintLogger
from descriptions.description import get_data_description
from environment import environment, get_dataset_paths
from figures.plot_config import get_plot_config
from helpers.configuration import Config
from helpers.determinism import make_deterministic

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)

parser.add_argument("--model", choices=["DT", "DTSciKit"], default='DT')
parser.add_argument("--dataset", choices=["ASIA", "CANCER", "EARTHQUAKE", "CHAIN"], default="ASIA")
parser.add_argument("--score", choices=["CausalLossScore", "GiniIndex", "GICL"], default="CausalLossScore")
parser.add_argument("--score_alpha", default=None)
parser.add_argument("--do_plot", default="true")
parser.add_argument("--do_eval", default="true")
parser.add_argument("--provide_interventions", choices=["true", "false"], default="true")  # provide intervention vector during training
cli_args = parser.parse_args()


do_eval = cli_args.do_eval.upper() == "TRUE"
do_plot = cli_args.do_plot.upper() == "TRUE"


conf = Config()
conf.model_name = cli_args.model
conf.dataset = cli_args.dataset
conf.batch_size = 1000
conf.seed = cli_args.seed
conf.score = cli_args.score
conf.score_alpha = None if cli_args.score_alpha is None else float(cli_args.score_alpha)
conf.provide_interventions = (cli_args.provide_interventions.lower() == "true")

make_deterministic(conf.seed)

torch.set_grad_enabled(False)

# setup experiments folder
runtime_base_dir = environment["experiments"]["base"] / "E3" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E3" / "eval_logs"
plot_base_dir = environment["experiments"]["base"] / "E3" / "dt_plots"


experiment_name = get_E3_experiment_name(conf.dataset, conf.model_name, conf.seed, conf.score, loss_params=conf.score_alpha,
                                         provide_interventions=conf.provide_interventions)
load_dir = runtime_base_dir / experiment_name

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)

print("Arguments:", cli_args)


decision_tree = load_E3_model(conf.model_name, load_dir)

if do_plot:
    X, Y, _ = get_data_description(conf.dataset)
    X = X[:-1]  # remove "intervention" entry

    # create an index -> var name dict
    feature_names_v = [(name, False) for i, name in enumerate(X)]
    feature_names_i = [(f"do({name})", True) for i, name in enumerate(X)]  # add interventions
    feature_names = [*feature_names_v, *feature_names_i]
    target_names = list(Y)


    def custom_content_str(node):
        t = "True"
        f = "False"
        if node.is_leaf():
            # TODO assumes binary decision attributes
            decisions = [f"{target_names[i]} = {t if (node._class[i] == 1) else f}" for i, name in enumerate(target_names)]
            node_content = "\n".join(decisions)
            is_intervention_var = False
        else:
            try:
                decision_info = feature_names[int(node.decision_feature)]
            except IndexError:
                raise RuntimeError(f"Unknown feature id: {node.decision_feature}")
            node_content, is_intervention_var = decision_info
            node_content += "?"

        if is_intervention_var:
            node_content = {
                "name": node_content,
                "fillcolor": "#fae8e8",  # bg color
                "color": "#b55e5e",  # border color
            }
        else:
            #if node.is_leaf():
            #    node_content = {
            #        "name": node_content,
            #        "fillcolor": "#fafafa",  # bg color
            #        "color": "#b5b5b5",  # border color
            #    }
            #else:
            node_content = {
                "name": node_content,
                "fillcolor": "#e9f2faff",  # bg color
                "color": "#5f92b6",  # border color
            }
        return node_content

    dot = decision_tree.dot({"comment": ""}, {
        "shape": "box",
        "labeljust": "l",
        "style": "filled",
        "fillcolor": "#e9f2faff",  # bg color
        "color": "#5f92b6",  # border color
        "fontcolor": "#434343"  # text color
    }, {}, content_func=custom_content_str)
    dot.format = 'svg' #get_plot_config('ext')
    dot.dpi = get_plot_config('dpi')

    plot_base_dir.mkdir(exist_ok=True, parents=True)
    dot.render(plot_base_dir / experiment_name, view=False)


if do_eval:
    # setup dataset
    X_vars, Y_vars, interventionProvider = get_data_description(conf.dataset, no_interventions=not conf.provide_interventions)
    dataset_paths = get_dataset_paths(conf.dataset, "test")
    dataset = TabularDataset(dataset_paths, X_vars, Y_vars, None, part_transformer=interventionProvider)

    condition_data, class_data = dataset.get_all_data()

    num_samples = len(condition_data)
    correct = 0
    for i in range(num_samples):
        data_input = condition_data[i, :]
        data_class = class_data[i, :]

        prediction = decision_tree.predict(data_input)
        if torch.all(prediction == data_class):
            correct += 1

    print(f"Classified {num_samples} samples. Correct: {correct}. Accuracy: {correct/num_samples}")

logger.close()
