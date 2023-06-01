from ciSPN.trainers.losses import NLLLoss, MSELoss, CausalLoss
from models.spn_create import load_spn


def get_experiment_name(dataset, model_name, seed, loss_name=None, loss2_name=None, loss2_factor_str=None,
                        loss_params=None, E=1, provide_interventions=True, specific=None):
    exp_name = f"E{E}_{dataset}_{model_name}"
    if loss_name is not None:
        exp_name += f"_{loss_name}"
    if loss_params is not None:
        exp_name += f"_{loss_params}"
    if loss2_name is not None:
        exp_name += f"_{loss2_name}_{loss2_factor_str}"
    if not provide_interventions:
        exp_name += "_noInterventions"
    if specific is not None:
        exp_name += f"_{specific}"

    exp_name = f"{exp_name}/{seed}"
    return exp_name


def get_E1_experiment_name(dataset, model_name, seed, loss_name=None, loss2_name=None, loss2_factor_str=None, loss_params=None, provide_interventions=True):
    return get_experiment_name(dataset, model_name, seed, loss_name=loss_name,
                               loss2_name=loss2_name, loss2_factor_str=loss2_factor_str, loss_params=loss_params, E=1, provide_interventions=provide_interventions)


def get_loss_path(dataset_name, loss_load_seed, E=1, provide_interventions=True):
    loss_folder = get_experiment_name(dataset_name, "ciSPN", loss_load_seed, "NLLLoss", None, None, E=E, provide_interventions=provide_interventions)
    return loss_folder


def get_E1_loss_path(dataset_name, loss_load_seed, provide_interventions=True):
    return get_loss_path(dataset_name, loss_load_seed, E=1, provide_interventions=provide_interventions)


def create_loss(loss_name, conf=None, num_condition_vars=None, load_dir=None, nn_provider=None, nn_provider_args=None):
    #FIXME conf arg is deprecated
    loss_spn = None
    if loss_name == "NLLLoss":
        loss = NLLLoss()
    elif loss_name == "MSELoss":
        loss = MSELoss()
    elif loss_name == "causalLoss":
        loss_spn, _, _ = load_spn(num_condition_vars, load_dir=load_dir, nn_provider=nn_provider, nn_provider_args=nn_provider_args)
        loss_spn.eval()
        loss = CausalLoss(loss_spn)
    else:
        raise ValueError(f"unknown loss name: {loss_name}")

    return loss, loss_spn
