import jax
import torch
import os
import time
import jax.numpy as jnp

from functools import partial
from omegaconf import DictConfig
from typing import Tuple, Sequence

from s4wm.utils.dlpack import from_jax_to_torch, from_torch_to_jax
from s4wm.nn.s4_wm import S4WorldModel
from s4wm.data.dataloaders import create_depth_dataset


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
        single_step=True,
        mutable=["cache"],
        method="forward_RNN_mode",
    )


class TorchWrapper:
    def __init__(
        self,
        batch_dim: int,
        ckpt_path: str,
        d_latent: int = 128,
        d_pssm_block: int = 512,
        d_ssm: int = 128,
    ) -> None:
        self.d_pssm_block = d_pssm_block
        self.d_ssm = d_ssm

        self.model = S4WorldModel(
            S4_config=DictConfig(
                {
                    "d_model": d_pssm_block,
                    "layer": {"l_max": 74, "N": d_ssm},
                }
            ),
            training=False,
            process_in_chunks=False,
            rnn_mode=True,
            **DictConfig(
                {
                    "latent_dim": d_latent,
                }
            ),
        )

        self.params = self.model.restore_checkpoint_state(ckpt_path)["params"]

        init_depth = jnp.zeros((batch_dim, 1, 270, 480, 1))
        init_actions = jnp.zeros((batch_dim, 1, 4))

        self.rnn_cache, self.prime = self.model.init_RNN_mode(
            self.params,
            init_depth,
            init_actions,
        )

        self.rnn_cache, self.prime, self.params = (
            sg(self.rnn_cache),
            sg(self.prime),
            sg(self.params),
        )

        # Force compilation
        _ = _jitted_forward(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            init_depth,
            init_actions,
        )
        return

    def forward(
        self, depth_imgs: torch.tensor, actions: torch.tensor
    ) -> Tuple[torch.tensor, ...]:  # 2 tuple

        jax_imgs, jax_actions = from_torch_to_jax(depth_imgs), from_torch_to_jax(
            actions
        )

        jax_preds, vars = _jitted_forward(
            self.model, self.params, self.rnn_cache, self.prime, jax_imgs, jax_actions
        )
        self.rnn_cache = vars["cache"]

        return (
            from_jax_to_torch(jax_preds["hidden"]),
            from_jax_to_torch(jax_preds["z_posterior"]["sample"]),
        )

    def reset_cache(self, batch_idx: Sequence) -> None:
        for i in range(4):
            for j in range(2):
                self.rnn_cache["PSSM_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                    "cache_x_k"
                ] = (
                    self.rnn_cache["PSSM_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                        "cache_x_k"
                    ]
                    .at[jnp.array([batch_idx])]
                    .set(jnp.ones((self.d_ssm, self.d_pssm_block), dtype=jnp.complex64))
                )
        return


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

    NUM_ENVS = 4

    torch_wm = TorchWrapper(
        NUM_ENVS,
        "/home/mathias/dev/structured-state-space-wm/s4wm/scripts/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=2/checkpoint_97",
        d_latent=1024,
        d_pssm_block=1024,
    )

    torch_wm.reset_cache(batch_idx=[0, 3])

    _, val_loader = create_depth_dataset(batch_size=1)
    test_depth_imgs, test_actions, _ = next(iter(val_loader))

    depth, actions = torch.unsqueeze(test_depth_imgs[:, 0], 1), torch.unsqueeze(
        test_actions[:, 0], 1
    )
    fwp_times = []
    for _ in range(200):
        start = time.time()
        _ = torch_wm.forward(depth, actions)
        end = time.time()
        print(end - start)
        fwp_times.append(end - start)
    fwp_times = jnp.array(fwp_times)

    print("Forward pass avg: ", jnp.mean(fwp_times))
    print("Forward pass std: ", jnp.std(fwp_times))
