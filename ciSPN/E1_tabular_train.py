from helpers.determinism import make_deterministic
from ciSPN.E1_helpers import get_experiment_name, create_loss, get_loss_path
from datasets.tabularDataset import TabularDataset
from descriptions.description import get_data_description
from models.nn_wrapper import NNWrapper
from trainers.dynamicTrainer import DynamicTrainer
from trainers.losses import AdditiveLoss
from datasets.batchProvider import BatchProvider
from helpers.configuration import Config
import pickle
import time
import argparse
from ciSPN.models.model_creation import create_spn_model, create_nn_model

from environment import environment, get_dataset_paths

from models.spn_create import save_spn, save_model_params

from libs.pawork.log_redirect import PrintLogger


print("start")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606) # 606, 1011, 3004, 5555, 12096
parser.add_argument("--model", choices=["mlp", "ciSPN"], default='mlp')
parser.add_argument("--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default='MSELoss')
parser.add_argument("--lr", type=float, default=1e-3) # default is 1e-3
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument("--loss2_factor", default="1.0")  # factor by which loss2 is added to the loss term
parser.add_argument("--epochs", type=int, default=50) # or 50
parser.add_argument("--loss_load_seed", type=int, default=None) # is set to seed if none
parser.add_argument("--dataset", choices=["CHC", "ASIA", "CANCER", "EARTHQUAKE", "CHAIN"], default="CHC") # CausalHealthClassification
parser.add_argument("--provide_interventions", choices=["true", "false"], default="true")  # provide intervention vector during training and use inintervened data
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.model_name = cli_args.model
conf.num_epochs = cli_args.epochs # Causal Health: prev was 130 - but 80 is enough ...
conf.num_epochs_load = conf.num_epochs # set to 80 when e.g. using a causal loss trained with 80 epochs
conf.loss_load_seed = cli_args.seed if cli_args.loss_load_seed is None else cli_args.loss_load_seed
conf.optimizer_name = "adam"
conf.lr = float(cli_args.lr)
conf.batch_size = 1000
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed
conf.provide_interventions = (cli_args.provide_interventions.lower() == "true")


make_deterministic(conf.seed)

# setup experiments folder
runtime_base_dir = environment["experiments"]["base"] / "E1" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E1" / "logs"

experiment_name = get_experiment_name(conf.dataset, conf.model_name, conf.seed, conf.loss_name, conf.loss2_name,
                                      conf.loss2_factor, provide_interventions=conf.provide_interventions)
save_dir = runtime_base_dir / experiment_name
save_dir.mkdir(exist_ok=True, parents=True)

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)


print("Arguments:", cli_args)


# setup dataset
X_vars, Y_vars, interventionProvider = get_data_description(conf.dataset, no_interventions=not conf.provide_interventions)
dataset_paths = get_dataset_paths(conf.dataset, "train", no_interventions=not conf.provide_interventions)
dataset = TabularDataset(dataset_paths, X_vars, Y_vars, conf.seed, part_transformer=interventionProvider)
provider = BatchProvider(dataset, conf.batch_size)

num_condition_vars = dataset.X.shape[1]
num_target_vars = dataset.Y.shape[1]

if conf.model_name == "ciSPN":
    # build spn graph
    rg, params, spn = create_spn_model(num_target_vars, num_condition_vars, conf.seed)
    model = spn
elif conf.model_name == "mlp":
    nn = create_nn_model(num_condition_vars, num_target_vars)
    model = NNWrapper(nn)

model.print_structure_info()


loss, loss_ispn = create_loss(conf.loss_name, conf, num_condition_vars,
                              load_dir=runtime_base_dir / get_loss_path(conf.dataset, conf.loss_load_seed, provide_interventions=conf.provide_interventions))

if conf.loss2_name is not None:
    loss2, loss2_ispn, = create_loss(
        conf.loss2_name, conf, num_condition_vars,
        load_dir=runtime_base_dir / get_loss_path(conf.dataset, conf.loss_load_seed, provide_interventions=conf.provide_interventions))
    final_loss = AdditiveLoss(loss, loss2, float(conf.loss2_factor))
else:
    final_loss = loss


trainer = DynamicTrainer(
    model, conf, final_loss, train_loss=False, pre_epoch_callback=None,
    optimizer=conf.optimizer_name, lr=conf.lr)

t0 = time.time()
loss_curve = trainer.run_training(provider)
training_time = time.time() - t0
print(f'TIME {training_time:.2f}')


if conf.model_name == "ciSPN":
    save_spn(save_dir, spn, params, rg, file_name="spn.model")
elif conf.model_name == "mlp":
    save_model_params(save_dir, nn, file_name="nn.model")

# save loss curve
with open(save_dir / "loss.pkl", "wb") as f:
    pickle.dump(loss_curve, f)

with open(save_dir / "runtime.txt", "wb") as f:
    pickle.dump(training_time, f)

print(f'Final parameters saved to "{save_dir}"')
logger.close()
