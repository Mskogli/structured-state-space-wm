import hydra
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
import jax

from omegaconf import DictConfig
from s4wm.nn.s4_wm import S4WorldModel
from s4wm.data.dataloaders import create_depth_dataset
from s4wm.utils.dlpack import from_torch_to_jax
import numpy
from functools import partial

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(
    jax.lax.stop_gradient, x
)  # stop gradient - used for KL balancing


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(
    model, params, cache, prime, imgs: jax.Array, actions: jax.Array
) -> jax.Array:
    return model.apply(
        {
            "params": sg(params),
            "cache": sg(cache),
            "prime": sg(prime),
        },
        imgs,
        actions,
        single_step=False,
        mutable=["cache"],
        method="forward_RNN_mode",
    )


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    SEED = 29
    torch.manual_seed(SEED)
    numpy.random.seed(SEED)

    model = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=True, **cfg.wm)
    trainloader, _ = create_depth_dataset(batch_size=2)
    test_depth_imgs, test_actions, _ = next(iter(trainloader))

    init_depth = jnp.zeros((2, 4, 270, 480, 1))
    init_actions = jnp.zeros((2, 4, 4))

    params = model.restore_checkpoint_state(
        "/home/mathias/dev/rl_checkpoints/gaussian_128_2"
    )["params"]

    cache, prime = model.init_RNN_mode(params, init_depth, init_actions)

    for i in range(74):
        out, variables = _jitted_forward(
            model,
            params,
            cache,
            prime,
            from_torch_to_jax(test_depth_imgs),
            from_torch_to_jax(test_actions),
        )
        cache = variables["cache"]


if __name__ == "__main__":
    main()
