import pickle
import numpy as np
import torch

factors = {"CHC" : 100, "JOB" : 10, "STUDENT" : 20}

class TabularDataset:

    def __init__(self, dataset_paths, X_vars, Y_vars, seed=None, known_intervention = False,
                 store_as_torch_tensor=True, part_transformers=None, normalise_max = False, dataset_name = None):
        self.rng = None if seed is None else np.random.default_rng(seed=seed)

        self.store_as_torch_tensor = store_as_torch_tensor

        parts_X = []
        parts_Y = []
        for data_path in dataset_paths:
            # print(f'data from {data_path}')
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                # print(f'example of data is {data}')
                if part_transformers is not None:
                    for part_transformer in part_transformers:
                        print(f'known intervention is {known_intervention}')
                        part_transformer(data_path, data, known_intervention)
                # exit(0);
                X_data = [np.expand_dims(data[x], -1) if len(data[x].shape) == 1 else data[x] for x in X_vars]
                Y_data = [np.expand_dims(data[y], -1) if len(data[y].shape) == 1 else data[y] for y in Y_vars]
                # print(f'X_data shape is {X_data[0].shape} len is {len(X_data)}')
                # print(f'Y_data shape is {Y_data[0].shape} len is {len(Y_data)}')
                parts_X.append(np.hstack(X_data))
                parts_Y.append(np.hstack(Y_data))
        self.X = np.vstack(parts_X)
        self.Y = np.vstack(parts_Y)
        # exit(0);
        
        
        # exit(0);

        # self.shuffle_data()

        # fixme assumes unique values in range 0..n!
        self.num_classes = int(np.max(self.Y)) + 1

        self.discrete_ids = {}
        self.discrete_ids = self.discrete_support()
        print(f'discrete_ids is {self.discrete_ids}')
        
        if normalise_max:
            self.normalise_max(factor = factors[dataset_name] if dataset_name in factors else 100)

        print(f'example of X data is {self.X[0]}')
        print(f'example of Y data is {self.Y[0]}')

        if self.store_as_torch_tensor:
            self.X = torch.tensor(self.X, dtype=torch.float).cuda().detach()
            self.Y = torch.tensor(self.Y, dtype=torch.float).cuda().detach()

    def get_all_data(self):
        return self.X, self.Y

    def shuffle_data(self):
        if self.rng is None:
            return
        permutation = self.rng.permutation(self.X.shape[0])
        self.X = self.X[permutation, :]
        self.Y = self.Y[permutation, :]
    
    def discrete_support(self):
        """Check for support of discrete random variables
        For each variable, find the number of unique values it has in the dataset.
        If the number of unique values is less than 20, it is discrete."""
        
        discrete_ids = list(filter(lambda i: len(list(filter(lambda x : not np.isnan(x), np.unique(self.Y[:, i])))) < 20, range(self.Y.shape[1])))
        print(f'discrete_ids is {discrete_ids}')
        num_categories = []
        for d in discrete_ids:
            ix = np.where(np.logical_not(np.isnan(self.Y[:,d])))[0]
            l = np.unique(self.Y[ix,d])
            num_categories.append(len(l))

            self.Y[ix,d] = np.array([np.where(l == x)[0][0] for x in self.Y[ix,d]])

        return dict(zip(discrete_ids, num_categories))
    
    def normalise_max(self, factor = 100):
        """Normalise each column of the data by its maximum value"""
        # self.X = self.X / np.max(self.X, axis=0)
        for col in range(self.Y.shape[1]):
            if col not in self.discrete_ids:
                self.Y[:,col] = 2*self.Y[:,col] / factor - 1 #now expecting from roughly -1 to 1

    def __len__(self):
        return self.X.shape[0]
