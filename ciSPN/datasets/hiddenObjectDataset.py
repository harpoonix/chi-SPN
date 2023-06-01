import numpy as np
import torch
from PIL import Image


class HiddenObjectDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, image_transform=None, add_intervention_channel=True, observational_only=False, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.add_intervention_channel = add_intervention_channel

        self.num_observed_variables = 6
        self.num_interventions = self.num_observed_variables

        if self.split == "train":
            self._full_data = np.genfromtxt(self.root_dir / "train_data.csv", delimiter=',', dtype=np.int)
            self.interventions = np.genfromtxt(self.root_dir /"train_intervention.csv", delimiter=',', dtype=np.int)
            self.img_dir = self.root_dir / "train_renders/"
            self._idx_map = np.array(list(range(len(self.interventions))))
            if observational_only:
                self.filter_data_from_mask(np.logical_or((self.interventions < 0), (self.interventions == self.num_interventions)))
        elif self.split == "test":
            self._full_data = np.genfromtxt(self.root_dir / "test_data.csv", delimiter=',', dtype=np.int)
            self.interventions = np.genfromtxt(self.root_dir / "test_intervention.csv", delimiter=',', dtype=np.int)
            self.img_dir = self.root_dir / "test_renders/"
            self._idx_map = np.array(list(range(len(self.interventions))))
            if observational_only:
                raise RuntimeError("Testing without interventions.")
        else:
            raise RuntimeError(f"Unknown dataset split {split}")

        # 3 objects with 3 attributes. The first two are observed via the image. The last one is hidden
        # fulldata: [ap, ac, as, bp, bc, bs, cp, cc, cs]
        self.hidden_data = self._full_data[:, -3:]  # select [cp, cc, cs]

        self.num_hidden_variables = 3

        self._num_samples = len(self.interventions)

    def filter_data_from_mask(self, mask):
        self._full_data = self._full_data[mask]
        self.interventions = self.interventions[mask]
        self._idx_map = self._idx_map[mask]

    def _create_intervention_image(self, intervention, img_size):
        height = img_size[0]
        width = img_size[1]

        iimg = np.zeros((1, height, width))
        if intervention != -1:
            section_width = width / self.num_interventions
            # set intervention information at lower part of the image
            iimg[0, int(height * 0.75):, int(intervention * section_width):int((intervention + 1) * section_width)] = 1
        return iimg

    def __len__(self):
        return self._num_samples

    def __getitem__(self, item):
        image_path = self.img_dir / f"{self._idx_map[item]}.jpg"
        image = np.moveaxis(np.array(Image.open(image_path), dtype=np.float), -1, 0) / 255.0

        hidden_data = self.hidden_data[item, :]

        if self.add_intervention_channel:
            iimg = self._create_intervention_image(self.interventions[item], (image.shape[1], image.shape[2]))
            image = np.concatenate([image, iimg], axis=0)

        if self.image_transform is not None:
            image = self.image_transform(image)

        image = torch.tensor(image)

        return {
            "image": image,
            #"intervention": intervention,
            "target": hidden_data
        }
