# original author: Fabian David Schmidt

import numpy as np
from torch.utils.data import Sampler
from datasets.arrow_dataset import Dataset


class MultinomialSampler(Sampler):
    def __init__(
            self, 
            datasets: dict[str, Dataset], 
            S: float = 1 / 3
            ):

        self.datasets = datasets
        self.S = S
        
        # Compute the total counts from all datasets
        self.lg2count = {lg: len(ds) for lg, ds in datasets.items()}
        self.sample_size = sum(self.lg2count.values())  # Set sample_size to cover all instances
        self.tot_S = sum([count**self.S for count in self.lg2count.values()])
        self.resampling_probs = {
            lg: (count**self.S) / self.tot_S for lg, count in self.lg2count.items()
        }
        self.sampled_indices = self.calculate_sample_indices()

    def calculate_sample_indices(self):
        # Calculate the number of samples to draw from each dataset
        sampled_counts = {
            lg: int(self.sample_size * prob)
            for lg, prob in self.resampling_probs.items()
        }
        sampled_indices = {}

        for lg, count in sampled_counts.items():
            # Ensure sampling covers all instances if possible
            len_ = len(self.datasets[lg])
            upsample = len_ < count
            indices = np.random.choice(len_, size=count, replace=upsample)
            sampled_indices[lg] = indices
        return sampled_indices

    def __iter__(self):
        # Flatten and shuffle the list of tuples (lg, idx)
        all_samples = [
            (lg, idx) for lg, indices in self.sampled_indices.items() for idx in indices
        ]
        np.random.shuffle(all_samples)
        return iter(all_samples)

    def __len__(self):
        return self.sample_size
