import torch
import h5py
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, random_split, DataLoader
from typing import List, Tuple
from s4wm.utils.dlpack import from_torch_to_jax


class DepthImageDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        device: str,
        actions: bool = False,
    ) -> None:
        self.file = h5py.File(file_path, "r")
        self.device = device
        self.actions = actions
        self.num_trajs = 16500
        self.max_depth_value = 10
        self.min_depth_value = 0.0

    def __len__(self) -> int:
        return self.num_trajs

    def __getitem__(self, idx) -> torch.tensor:
        depth_images = []
        states = []
        actions = []

        for i in range(100):
            dataset = self.file[f"trajectory_{idx}/image_{i}"]
            img_data = dataset[:]
            depth_images.append(
                torch.from_numpy(img_data)
                .view(1, 135, 240, 1)
                .to(torch.device(self.device))
            )
            states.append(
                torch.from_numpy(dataset.attrs["states"])
                .view(1, 16)
                .to(torch.device(self.device))
            )
            actions.append(
                torch.from_numpy(dataset.attrs["actions"])
                .view(1, 4)
                .to(torch.device(self.device))
            )

        imgs = torch.cat(depth_images, dim=0)

        # Process images
        imgs[torch.isinf(imgs)] = self.max_depth_value
        imgs[imgs > self.max_depth_value] = self.max_depth_value
        imgs[imgs < self.min_depth_value] = self.min_depth_value
        imgs = (imgs - self.min_depth_value) / (
            self.max_depth_value - self.min_depth_value
        )

        acts = torch.cat(actions, dim=0)[1:]
        sts = torch.cat(states, dim=0)[:-1]
        extras = torch.cat([acts, sts], dim=-1)

        labels = imgs[1:, :].view(-1, 135 * 240)

        return (imgs, acts, labels)


def split_dataset(
    dataset: DepthImageDataset, val_split: float
) -> Tuple[DepthImageDataset, ...]:  # 2 tuple
    total_samples = dataset.num_trajs
    train_len = int(total_samples * (1 - val_split))
    val_len = total_samples - train_len

    return random_split(dataset, [train_len, val_len])


if __name__ == "__main__":

    dataset = DepthImageDataset(
        "/home/mathias/dev/datasets/quad_depth_imgs",
        "cuda:0",
        actions=True,
    )

    train_dataset, val_dataset = split_dataset(dataset, 0.1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    train_batch, _, _ = next(iter(train_loader))

    for idx, image in enumerate(train_batch[0], 1):
        print(image.size())
        print(image.requires_grad)
        print(image.device)
        plt.imsave(f"imgs/test{idx}.png", image.view(135, 240).cpu().numpy())
