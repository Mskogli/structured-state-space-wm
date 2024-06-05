import numpy as np

from jax.tree_util import tree_map
from torch.utils import data
from torch.utils.data import Dataset, random_split, DataLoader


from .depth_img_dataset import DepthImageDataset, split_dataset

from typing import Tuple


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ) -> None:
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def create_depth_dataset(
    batch_size: int = 128,
) -> Tuple[NumpyLoader, ...]:  # 2 tuple
    print("[*] Creating Dataset and Generating Dataloaders")

    dataset = DepthImageDataset(
        "/home/mathias/dev/datasets/quad_depth_imgs",
        "cuda:0",
        actions=True,
    )

    train_dataset, val_dataset = split_dataset(dataset, 0.1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


Dataloaders = {
    "depth_dataset": create_depth_dataset,
}
