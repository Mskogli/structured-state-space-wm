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
    torch.manual_seed(0)

    model = S4WorldModel(S4_config=cfg.model, training=False, **cfg.wm)
    _, trainloader = create_depth_dataset(batch_size=4)
    test_depth_imgs, test_actions, _ = next(iter(trainloader))

    test_depth_imgs = from_torch_to_jax(test_depth_imgs)
    test_actions = from_torch_to_jax(test_actions)

    params = model.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/s4wm/nn/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=8-latent_type=Categorical_12_blocks/checkpoint_99"
    )["params"]

    init_depth = jnp.zeros((4, 1, 135, 240, 1))
    init_actions = jnp.zeros((4, 1, 4))

    cache, prime = model.init_RNN_mode(params, init_depth, init_actions)

    ctx_l = 90
    dream_l = 9
    context_imgs = test_depth_imgs[:, :ctx_l, :]
    context_actions = test_actions[:, 1 : ctx_l + 1, :]
    dream_actions = test_actions[:, ctx_l + 1 : ctx_l + dream_l + 1, :]
    dream_actions = jnp.zeros_like(dream_actions)
    dream_actions = dream_actions.at[:, :, 2].set(-0.645)

    print("batman")

    out, _ = model.apply(
        {"params": params, "cache": cache, "prime": prime},
        context_imgs,
        context_actions,
        dream_actions,
        dream_l,
        mutable=["cache"],
        method="dream",
    )

    for i in range(dream_l):
        plt.imsave(f"imgs/draum_{i}.png", out[0][i][3, 0].reshape(135, 240))
        plt.imsave(
            f"imgs/gt_dream{i}.png", test_depth_imgs[3, ctx_l + i].reshape(135, 240)
        )


if __name__ == "__main__":
    main()
