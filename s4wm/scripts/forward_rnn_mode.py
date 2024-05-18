import hydra
import os
import jax.numpy as jnp
import torch
import jax
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from s4wm.nn.s4_wm import S4WorldModel
from s4wm.data.dataloaders import create_depth_dataset
from s4wm.utils.dlpack import from_torch_to_jax
from functools import partial


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(
    model, params, cache, prime, imgs: jax.Array, actions: jax.Array, key
) -> jax.Array:
    out, vars = model.apply(
        {
            "params": params,
            "cache": cache,
            "prime": prime,
        },
        imgs,
        actions,
        key,
        True,
        mutable=["cache"],
    )
    return (
        out["depth"]["recon"].mean(),
        out["depth"]["pred"].mean(),
        out["z_prior"]["sample"],
        vars,
    )


@partial(jax.jit, static_argnums=(0))
def dream(model, params, cache, prime, pred_posterior, action, key) -> jax.Array:
    out, vars = model.apply(
        {
            "params": params,
            "cache": cache,
            "prime": prime,
        },
        pred_posterior,
        action,
        key,
        mutable=["cache"],
        method="open_loop_prediction",
    )
    return out["depth_pred"].mean(), out["z_post_pred"]["sample"], vars


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    context_length = 10
    dream_length = 20
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    key = jax.random.PRNGKey(0)
    model = S4WorldModel(S4_config=cfg.model, training=False, **cfg.wm)
    torch.manual_seed(0)

    _, val_loader = create_depth_dataset(batch_size=8)
    test_depth_imgs, test_actions, _ = next(iter(val_loader))

    test_depth_imgs = from_torch_to_jax(test_depth_imgs)
    test_actions = from_torch_to_jax(test_actions)

    init_depth = jnp.zeros((8, 1, 135, 240, 1))
    init_actions = jnp.zeros((8, 1, 4))

    state = model.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/s4wm/nn/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=8-latent_type=Categorical_12_blocks/checkpoint_99"
    )
    params = state["params"]

    cache, prime = model.init_RNN_mode(params, init_depth, init_actions)

    # Build context
    z_post = None

    batch = 3

    for i in range(context_length):
        sample_key, key = jax.random.split(key, num=2)
        depth = jnp.expand_dims(test_depth_imgs[:, i], axis=1)
        action = jnp.expand_dims(test_actions[:, i], axis=1)

        depth_recon, depth_pred, z_post, variables = _jitted_forward(
            model,
            params,
            cache,
            prime,
            depth,
            action,
            sample_key,
        )
        cache = variables["cache"]

        plt.imsave(
            f"imgs/recon_rnn_{i}.png",
            depth_recon[batch].reshape(135, 240),
            cmap="magma",
            vmin=0,
            vmax=1,
        )

        if i == context_length - 1:
            plt.imsave(
                f"imgs/dream_rnn_0.png",
                depth_pred[batch].reshape(135, 240),
                cmap="magma",
                vmin=0,
                vmax=1,
            )
            plt.imsave(
                f"imgs/dream_label_0.png",
                test_depth_imgs[batch, i + 1].reshape(135, 240),
                cmap="magma",
                vmin=0,
                vmax=1,
            )

    # Open loop predictions

    for i in range(dream_length):
        sample_key, key = jax.random.split(key, num=2)
        action = jnp.expand_dims(test_actions[:, i + context_length], axis=1)
        # action = action.at[:, :, 3].set(-1)
        # action = action.at[:, :, 0].set(0)
        # action = action.at[:, :, 1].set(0)
        # action = action.at[:, :, 2].set(1)
        depth_recon, z_post, variables = dream(
            model, params, cache, prime, z_post, action, key
        )
        cache = variables["cache"]
        plt.imsave(
            f"imgs/dream_rnn_{i+1}.png",
            depth_recon[batch].reshape(135, 240),
            cmap="magma",
            vmin=0,
            vmax=1,
        )

        plt.imsave(
            f"imgs/dream_label_{i+1}.png",
            test_depth_imgs[batch, i + context_length + 1].reshape(135, 240),
            cmap="magma",
            vmin=0,
            vmax=1,
        )


if __name__ == "__main__":
    main()
