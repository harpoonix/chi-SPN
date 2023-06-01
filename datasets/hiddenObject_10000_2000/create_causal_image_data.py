import numpy as np
import copy
import pathlib

rng = np.random.default_rng(123)

num_train_samples = 10000
num_test_samples = 2000


def draw_categorical(probabilities):
    cs = np.cumsum(probabilities)
    cs[-1] = 1.00001 # just to make sure ...
    return np.argmax(cs>rng.uniform())


obj_a_position = lambda vars: draw_categorical([0.25, 0.5, 0.25])  # left, middle, right
obj_a_color = lambda vars: rng.integers(0, 4)  # red, green, blue, orange
obj_a_shape = lambda vars: draw_categorical([0.25, 0.25, 0.4, 0.6]) # cube pyramid cylinder sphere


def obj_b_position(vars):
    pos = [0, 1, 2]
    pos.remove(vars["ap"])
    return pos[draw_categorical([0.4, 0.6])]
def obj_b_color(vars):
    return (vars["ac"] + vars["bp"] + draw_categorical([0.7, 0.3]) + 4) % 4
def obj_b_shape(vars):
    # return rng.integers(0, 4)

    # in the correlational setting, bs will be a better estimator of cs, but
    # it is not the true cause!
    return vars["cs"]


def obj_c_position(vars):
    ap = vars["ap"]
    bp = vars["bp"]

    r = rng.uniform(0, 1)
    if r < 0.98:
        return (ap + bp)//2
    #elif r < 0.01:
    #    return 0 # red
    else:
        return rng.integers(0, 3)
def obj_c_color(vars):
    return ((vars["bc"] + draw_categorical([0.9, 0.1]) - 1) + 4) % 4
def obj_c_shape(vars):
    #return round(0.5*vars["ac"] + 0.2 * vars["as"] + 0.2 * vars["bs"] + draw_categorical([0.86, 0.14])) % 4
    return ((vars["as"] + draw_categorical([0.1, 0.8, 0.1]) - 1) + 4) % 4

var_descriptions = [
    {"name": "ap", "generator": obj_a_position, "range": [0, 3]},
    {"name": "ac", "generator": obj_a_color, "range": [0, 4]},
    {"name": "as", "generator": obj_a_shape, "range": [0, 4]},
    {"name": "cs", "generator": obj_c_shape, "range": [0, 4]},
    {"name": "bp", "generator": obj_b_position, "range": [0, 3]},
    {"name": "bc", "generator": obj_b_color, "range": [0, 4]},
    {"name": "bs", "generator": obj_b_shape, "range": [0, 4]},
    {"name": "cp", "generator": obj_c_position, "range": [0, 3]},
    {"name": "cc", "generator": obj_c_color, "range": [0, 4]},
]

# "cp", "cc", "cs" are target variables
interventions = [None, "ap", "ac", "as", "bp", "bc", "bs"]

var_ordering = ["ap", "ac", "as", "bp", "bc", "bs", "cp", "cc", "cs"]

assert len(var_ordering) == len(var_descriptions)

def do_sample(intervention_name):
    var_descriptions_mod = var_descriptions.copy()

    if intervention_name is not None:
        iidx = [idx for idx, val in enumerate(var_descriptions_mod) if val["name"] == intervention_name][0]

        # replace variable generator with a uniform intervention
        intervened_description = var_descriptions_mod[iidx].copy()
        intervened_description["generator"] = lambda vars: rng.integers(
            intervened_description["range"][0], intervened_description["range"][1])
        var_descriptions_mod[iidx] = intervened_description

    vars = {}
    vector = [-1]*len(var_ordering)
    for description in var_descriptions_mod:
        value = description["generator"](vars)
        vars[description["name"]] = value
        vector[var_ordering.index(description["name"])] = value
    return vector

def sample_set(N):
    data = np.empty((N, len(var_descriptions)), dtype=int)
    interventions_data = np.empty((N), dtype=int)
    for i in range(N):
        intervention = rng.integers(len(interventions)) - 1

        data[i,:] = do_sample(interventions[intervention + 1])
        interventions_data[i] = intervention
    return data, interventions_data

train_set = sample_set(num_train_samples)
test_set = sample_set(num_test_samples)


dataset_base_name = "causalImage"
dataset_full_name = f"{dataset_base_name}_{num_train_samples}_{num_test_samples}"

dataset_path = pathlib.Path('./')

np.savetxt(dataset_path / "train_data.csv", train_set[0], fmt='%s', delimiter=",")
np.savetxt(dataset_path / "train_intervention.csv", train_set[1], fmt='%s', delimiter=",")
np.savetxt(dataset_path / "test_data.csv", test_set[0], fmt='%s', delimiter=",")
np.savetxt(dataset_path / "test_intervention.csv", test_set[1], fmt='%s', delimiter=",")


print("done")
