import pickle

from ciSPN.E1_helpers import create_loss, get_E1_loss_path, get_experiment_name, get_E1_experiment_name
from ciSPN.models.decisionTree.DecisionTree import DecisionTree
from ciSPN.trainers.decisionTree.GiniIndex import GiniIndex
from ciSPN.trainers.decisionTree.causalLossScore import CausalLossScore
from ciSPN.trainers.decisionTree.combinedScore import CombinedScore


def get_E3_experiment_name(dataset, model_name, seed, loss_name=None, loss2_name=None, loss2_factor_str=None, loss_params=None,
                           provide_interventions=True, specific=None):
    return get_experiment_name(dataset, model_name, seed, loss_name=loss_name,
                               loss2_name=loss2_name, loss2_factor_str=loss2_factor_str, loss_params=loss_params, E=3,
                               provide_interventions=provide_interventions, specific=specific)


def create_E3_score(score_name, conf, score_alpha=None, batch_size=None, num_condition_vars=None, runtime_loss_base_dir=None, provide_interventions=True):
    if score_name == "CausalLossScore" or score_name == "GICL":
        _, spn = create_loss("causalLoss", num_condition_vars=num_condition_vars,
                             load_dir=runtime_loss_base_dir / get_E1_loss_path(conf.dataset, conf.loss_load_seed, provide_interventions=provide_interventions))

    if score_name == "CausalLossScore":
        scorer = CausalLossScore(spn, batch_size)
    elif score_name == "GiniIndex":
        scorer = GiniIndex()
    elif score_name == "GICL":
        scorer = CombinedScore(score_alpha, GiniIndex(), CausalLossScore(spn, batch_size))
    else:
        raise ValueError(f"Unkown score: {score_name}")
    return scorer


def create_E3_model(model_name):
    if model_name == "DT":
        decision_tree = DecisionTree("root")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return decision_tree


def load_E3_model(model_name, load_dir):
    if model_name == "DT" or model_name == "DTSciKit":
        with open(load_dir / "tree.pkl", "rb") as f:
            decision_tree = pickle.load(f)
        decision_tree.clean()
        decision_tree.to_torch(to_torch=True, cuda=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return decision_tree
