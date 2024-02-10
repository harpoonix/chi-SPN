import numpy as np

intervention_vars_dict = {
    'CHC': ['A', 'F', 'H', 'M'],
    'ASIA': ["A", "T", "B", "L", "E"],
    'CANCER': ["S", "C"],
    'EARTHQUAKE': ["B", "E", "A"],
    'CHAIN': ["A", "C"],
    'JOB': ["I", "E", "Sk", "B"],
    'STUDENT': ["Q", "M", "C", "T"]
}

intervention_vector_providers = {
    "CHC": lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["CHC"]
    ),
    "ASIA": lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["ASIA"]
    ),
    "CANCER": lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["CANCER"]
    ),
    "EARTHQUAKE": lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["EARTHQUAKE"]
    ),
    "WATERING": lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["WATERING"]
    ),
    "TOY1": lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["TOY1"]
    ),
    "TOY2": lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["TOY2"]
    ),
    "TOY1I": lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["TOY1I"]
    ),
    "JOB" : lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["JOB"]
    ),
    "STUDENT" : lambda intervention: get_intervention_vector(
        intervention, intervention_vars_dict["STUDENT"]
    ),
}

def get_intervention_vector(intervention, intervention_vars):
    if intervention == 'None' or intervention is None:
        # make sure to not accidentally detect 'N'
        intervention_var = ''
    else:
        # intervention_var = intervention.split('(')[1].split(")")[0]
        intervention_var = intervention.split('_')
        intervention_var = list(map(lambda x : x.split('(')[1].split(')')[0], intervention_var))
        print(intervention_var)
    intervention_vector = np.array([1 if var in intervention_var else 0 for var in intervention_vars])
    return intervention_vector


def get_no_intervention_provider(intervention, intervention_vars):
    # always produces a vector of zeros
    intervention_vector = np.zeros(len(intervention_vars))
    return intervention_vector


reference_vars = {
    "CANCER": "P",
    "EARTHQUAKE": "B",
    "ASIA": "A",
    "CHC": "A",
    "CHAIN": "A",
    "JOB": "E",
    "STUDENT" : "Sc"
}


class InterventionProvider:
    """
    Adds the intervention vector to the data
    """

    def __init__(self, dataset_name, field_name="intervention"):
        self.field_name = field_name
        self.dataset_name = dataset_name

        self.reference_var = reference_vars[dataset_name]
        self.intervention_graph_provider = intervention_vector_providers[dataset_name]
        self.intervention = []

    def __call__(self, path, data, known_intervention):
        intervention_str = path.name.split("n_")[1].split("_N")[0]
        # intervention_str = path.name.split("_")[1].split("_")[0]
        intervention_vector = self.intervention_graph_provider(
            intervention_str
        ).flatten()

        self.intervention.append(intervention_str)
        if known_intervention:
            # add the specific intervention value (usually, you only know the intervened variable)
            if intervention_str == "None" or intervention_str is None:
                # make sure to not accidentally detect 'N'
                intervention_var = ""
            else:
                intervention_var = intervention_str.split("(")[1].split(")")[0]
            intervention_info = np.tile(
                intervention_vector, (len(data[self.reference_var]), 1)
            )
            interv_value = (
                np.zeros_like(data[self.reference_var])
                if intervention_var is ""
                else data[intervention_var]
            )
            data[self.field_name] = np.concatenate(
                (intervention_info, interv_value), axis=1
            )
        else:
            # only use the intervention_vector (indicates intervened variable with a 1)
            data[self.field_name] = np.tile(
                intervention_vector, (len(data[self.reference_var]), 1)
            )

