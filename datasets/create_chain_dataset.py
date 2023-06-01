from pathlib import Path
import numpy as np

from interventionSCM import InterventionSCM, create_dataset_train_test

"""
 Chain Dataset

            'A': A
            'B': B
            'C': C 
    
"""

dataset_name = 'CHAIN'  # used as filename prefix
save_dir = Path(f"./{dataset_name}/")  # base folder
save = True
save_plot_and_info = True

if save:
    save_dir.mkdir(exist_ok=True, parents=True)


class SCM_Chain(InterventionSCM):

    def __init__(self, seed):
        super().__init__(seed)
        yes = 1
        no = 0

        a = lambda size: self.rng.binomial(1, 0.5, size=(size, 1))
        noise = 0.25
        b = lambda size, a: np.where(a == yes, self.rng.binomial(1, (1-noise), size=(size, 1)), self.rng.binomial(1, noise, size=(size, 1)))
        c = lambda size, b: np.where(b == yes, np.ones((size, 1)), np.zeros((size, 1)))

        self.equations = {
            'A': a,
            'B': b,
            'C': c,
        }

    def create_data_sample(self, sample_size, domains=True):
        As = self.equations['A'](sample_size)
        Bs = self.equations['B'](sample_size, As)
        Cs = self.equations['C'](sample_size, Bs)

        data = {'A': As, 'B': Bs, 'C': Cs}
        return data

"""
parameters
"""

variable_names = ['A', 'B', 'C']
variable_abrvs = ['A', 'B', 'C']
intervention_vars = ['A', 'C']  # B is used as target var
exclude_vars = []  # exclude intermediate variables from the final dataset

interventions = [(None, "None"), *[(iv, f"do({iv})=UBin({iv})") for iv in intervention_vars]]


seed = 123
np.random.seed(seed)
N = 100000
test_split = 0.2

for i, interv in enumerate(interventions):
    _, interv_desc = interv
    scm = SCM_Chain(seed+i)
    create_dataset_train_test(
        scm, interv_desc, N, dataset_name,
        test_split=test_split,
        save_dir=save_dir,
        save_plot_and_info=save_plot_and_info,
        variable_names=variable_names,
        variable_abrvs=variable_abrvs,
        exclude_vars=exclude_vars)
