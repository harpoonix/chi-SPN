import numpy as np

intervention_vars_dict = {
    'CHC': ['A', 'F', 'H', 'M'],
    'ASIA': ["A", "T", "B", "L", "E"],
    'CANCER': ["S", "C"],
    'EARTHQUAKE': ["B", "E", "A"],
    'CHAIN': ["A", "C"]
}


def get_intervention_vector(intervention, intervention_vars):
    if intervention == 'None' or intervention is None:
        # make sure to not accidentally detect 'N'
        intervention_var = ''
    else:
        intervention_var = intervention.split('(')[1].split(")")[0]
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
    "CHAIN": "A"
}


class InterventionProvider:
    """
    Adds the intervention vector to the data
    """

    def __init__(self, dataset_name, field_name="intervention", no_interventions=False):
        self.field_name = field_name
        self.dataset_name = dataset_name

        self.reference_var = reference_vars[dataset_name]
        if no_interventions:
            self.intervention_vector_provider = lambda intervention: get_no_intervention_provider(intervention, intervention_vars_dict[dataset_name])
        else:
            self.intervention_vector_provider = lambda intervention: get_intervention_vector(intervention, intervention_vars_dict[dataset_name])
        self.intervention = []

    def __call__(self, path, data):
        intervention_str = path.name.split('_')[1].split('_')[0]
        intervention_vector = self.intervention_vector_provider(intervention_str).flatten()

        self.intervention.append(intervention_str)
        data[self.field_name] = np.tile(intervention_vector, (len(data[self.reference_var]), 1))
