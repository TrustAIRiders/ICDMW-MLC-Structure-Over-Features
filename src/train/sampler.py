from torch.utils.data import WeightedRandomSampler, Sampler
import numpy as np


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_batches=None):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_batches = num_batches

        self.class_indices = {}
        for c in np.unique(labels):
            self.class_indices[c] = np.where(self.labels == c)[0].tolist()

        self.num_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.num_classes

        if self.num_batches is None:
            max_class_size = max(len(idx) for idx in self.class_indices.values())
            self.num_batches = max_class_size // self.samples_per_class

    def __iter__(self):
        for c in self.class_indices:
            np.random.shuffle(self.class_indices[c])

        pointers = {c: 0 for c in self.class_indices}

        for _ in range(self.num_batches):
            batch = []
            for c in self.class_indices:
                indices = self.class_indices[c]
                n = len(indices)

                selected = []
                for _ in range(self.samples_per_class):
                    selected.append(indices[pointers[c] % n])
                    pointers[c] += 1

                batch.extend(selected)

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def make_weighted_sampler(dataset):
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)

    sample_weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler
