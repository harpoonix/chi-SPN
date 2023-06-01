from ciSPN.datasets.interventionHelpers import InterventionProvider

var_sets = {
    "CHC": {
        "X": ["A", "F", "H", "M", "intervention"],
        "Y": ["D1", "D2", "D3"]
    },
    "ASIA": {
        "X": ["A", "T", "B", "L", "E", "intervention"],
        "Y": ["S", "X", "D"]
    },
    "CANCER": {
        "X": ["S", "C", "intervention"],
        "Y": ["P", "X", "D"]
    },
    "EARTHQUAKE": {
        "X": ["B", "E", "A", "intervention"],
        "Y": ["J", "M"]
    },
    "CHAIN": {
        "X": ["A", "C", "intervention"],
        "Y": ["B"]
    }
}

def get_data_description(dataset_abrv, no_interventions=False):
    var_set = var_sets[dataset_abrv]
    X = var_set["X"]
    Y = var_set["Y"]

    if no_interventions:
        intervention_provider = None
        assert X[-1] == "intervention"
        X = X[:-1] # remove intervention entry
    else:
        intervention_provider = InterventionProvider(dataset_abrv, no_interventions=no_interventions)

    return X, Y, intervention_provider
