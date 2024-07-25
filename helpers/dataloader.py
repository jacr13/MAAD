import random


class DataloaderIterator:
    def __init__(self, dataset, indexes_batched) -> None:
        self.dataset = dataset
        self.indexes_batched = indexes_batched

        self._batch_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._batch_id < len(self.indexes_batched):
            batch = self.dataset.get_batch(self.indexes_batched[self._batch_id])
            self._batch_id += 1
            return batch
        else:
            raise StopIteration


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False) -> None:
        assert all(
            list(dataset.data.values())[0].size(0) == tensor.size(0)
            for tensor in dataset.data.values()
        ), "Size mismatch between tensors"
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.length = list(dataset.data.values())[0].size(0)

        self.indexes = list(range(self.length))

        if self.shuffle:
            random.shuffle(self.indexes)

        self.indexes_batched = [
            self.indexes[start_idx : start_idx + self.batch_size]
            for start_idx in range(0, self.length, self.batch_size)
        ]

        if (
            self.drop_last
            and len(self.indexes_batched) > 1
            and len(self.indexes_batched[-1]) != self.batch_size
        ):
            self.indexes_batched.pop()

    def to(self, device):
        self.dataset.to(device)

    def __iter__(self):
        return DataloaderIterator(self.dataset, self.indexes_batched)

    def __len__(self):
        return len(self.indexes_batched)
