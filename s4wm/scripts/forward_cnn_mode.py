import jax
import hydra
import os
import torch

import jax.numpy as jnp
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from s4wm.nn.s4_wm import S4WorldModel
from s4wm.data.dataloaders import create_depth_dataset
from s4wm.utils.dlpack import from_torch_to_jax


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    model = S4WorldModel(S4_config=cfg.model, training=False, **cfg.wm)
    torch.manual_seed(0)

    _, trainloader = create_depth_dataset(batch_size=1)
    test_depth_imgs, test_actions, _ = next(iter(trainloader))

    test_depth_imgs = from_torch_to_jax(test_depth_imgs)
    test_actions = from_torch_to_jax(test_actions)

    state = model.restore_checkpoint_state(
        "/home/mihir/dev-mathias/structured-state-space-wm/s4wm/nn/checkpoints/depth_dataset/d_model=1024-lr=0.0002-bsz=8-latent_type=Gaussian_12_blocks/checkpoint_88"
    )
    params = state["params"]

    test_depth_imgs = jnp.expand_dims(test_depth_imgs, axis=0)
    test_actions = jnp.expand_dims(test_actions, axis=0)

    init_depth = jnp.zeros((1, 1, 135, 240, 1))
    init_actions = jnp.zeros((1, 1, 4))
    model.init(
        jax.random.PRNGKey(0),
        init_depth,
        init_actions,
        jax.random.PRNGKey(2),
    )
    key = jax.random.PRNGKey(1)
    out = model.apply(
        {"params": params},
        test_depth_imgs,
        test_actions,
        key,
        reconstruct_priors=True,
    )

    pred_depth = out["depth"]["pred"].mean()
    recon_depth = out["depth"]["recon"].mean()
    batch = 2
    print(recon_depth[0, 0].shape)
    color = "magma"
    for i in range(99):
        plt.imsave(
            f"imgs/pred_{i}.png",
            pred_depth[batch, i, :].reshape(135, 240),
            cmap=color,
            vmin=0,
            vmax=1,
        )
        plt.imsave(
            f"imgs/recon_{i}.png",
            recon_depth[batch, i, :].reshape(135, 240),
            cmap=color,
            vmin=0,
            vmax=1,
        )
        plt.imsave(
            f"imgs/label_{i}.png",
            test_depth_imgs[batch, i + 1, :].reshape(135, 240),
            cmap=color,
            vmin=0,
            vmax=1,
        )


if __name__ == "__main__":
    main()
