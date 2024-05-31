import jax
import orbax
import jax.numpy as jnp
import orbax.checkpoint
import torch

from functools import partial
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from omegaconf import DictConfig

from .decoder import ImageDecoder, Decoder, ResNetDecoder, ResNetBlockDecoder
from .encoder import SimpleEncoder, ResNetEncoder, ResNetBlock
from .s4_nn import S4Block
from .dists import OneHotDist, MSEDist, sg, LogCoshDist

from s4wm.utils.dlpack import from_jax_to_torch, from_torch_to_jax
from typing import Dict, Union, Tuple, Sequence, Literal

tfd = tfp.distributions
f32 = jnp.float32

ImageDistribution = Literal["MSE", "LogCosh"]
LatentDistribution = Literal["Gaussian", "Categorical"]
LossReduction = Literal["sum", "mean"]


class S4WorldModel(nn.Module):
    S4_config: DictConfig

    latent_dim: int = 128
    num_classes: int = 32
    num_modes: int = 32

    alpha: float = 0.8
    beta_rec: float = 1.0
    beta_kl: float = 1.0
    kl_lower_bound: float = 1.0

    training: bool = True
    beta_warmup: bool = True

    rnn_mode: bool = False
    sample_mean: bool = False
    clip_kl_loss: bool = True

    image_dist_type: ImageDistribution = "MSE"
    latent_dist_type: LatentDistribution = "Categorical"
    loss_reduction: LossReduction = "mean"

    def setup(self) -> None:
        self.rng_post, self.rng_prior = jax.random.split(jax.random.PRNGKey(0), num=2)
        self.discrete_latent_state = self.latent_dist_type == "Categorical"

        self.encoder = ResNetEncoder(act_fn=nn.silu)
        self.decoder = ResNetDecoder(act_fn=nn.silu)

        self.S4_blocks = S4Block(
            **self.S4_config, rnn_mode=self.rnn_mode, training=self.training
        )

        self.statistic_heads = {
            "embedding": nn.Sequential(
                [
                    nn.Dense(features=self.latent_dim),
                    nn.silu,
                    nn.Dense(features=self.latent_dim),
                ]
            ),
            "hidden": nn.Sequential(
                [
                    nn.Dense(features=self.S4_config["d_model"]),
                    nn.silu,
                    nn.Dense(features=self.S4_config["d_model"]),  # 1x 2024
                    nn.silu,
                    nn.Dense(features=self.latent_dim),
                ]
            ),
        }

        self.input_head = nn.Sequential(
            [
                nn.Dense(features=self.S4_config["d_model"]),
                nn.silu,
                nn.Dense(features=self.S4_config["d_model"]),
            ]
        )

        self.beta_warmup_schedule = [
            self.calculate_cyclical_lr(i, 1857, 1, 0) for i in range(1700)
        ]
        self.beta_warmup_index = 0

    def compute_latent(
        self, statistics: jnp.ndarray, rng
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        dists = self.get_latent_distribution(statistics)

        if not self.sample_mean:
            sample = dists.sample(seed=rng)
        else:
            sample = dists.mode() if self.discrete_latent_state else dists.mean()

        if self.discrete_latent_state:
            sample = jax.lax.collapse(
                sample,
                start_dimension=2,
                stop_dimension=4,  # Flatten sample from a categorical with shape (batch, seq_l, modes, num_classes)
            )

        return sample, dists

    def compute_posteriors(
        self, embedding: jnp.ndarray, key
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        post_stats = self.get_statistics(embedding, statistics_head="embedding")
        return self.compute_latent(post_stats, key)

    def compute_priors(
        self, hidden: jnp.ndarray, key
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        prior_stats = self.get_statistics(hidden, statistics_head="hidden")
        return self.compute_latent(prior_stats, key)

    def reconstruct_depth(
        self, hidden: jnp.ndarray, latent_sample: jnp.ndarray
    ) -> tfd.Distribution:
        x = self.decoder(jnp.concatenate((hidden, latent_sample), axis=-1))
        return self.get_image_distribution(x)

    def compute_loss(
        self,
        img_prior_dist: tfd.Distribution,
        img_posterior: jnp.ndarray,
        z_posterior_dist: tfd.Distribution,
        z_prior_dist: tfd.Distribution,
        beta_rate: int = 1,
    ) -> jnp.ndarray:
        # Compute the KL loss with KL balancing https://arxiv.org/pdf/2010.02193.pdf
        # in order to focus on learning the posterior rather than the prior

        dynamics_loss = sg(z_posterior_dist).kl_divergence(z_prior_dist)
        representation_loss = z_posterior_dist.kl_divergence(sg(z_prior_dist))

        if self.clip_kl_loss:
            dynamics_loss = jnp.maximum(dynamics_loss, self.kl_lower_bound)
            representation_loss = jnp.maximum(representation_loss, self.kl_lower_bound)

        kl_loss = (
            beta_rate
            * self.beta_kl
            * jnp.sum(
                (self.alpha * dynamics_loss + (1 - self.alpha) * representation_loss),
                axis=-1,
            )
        )
        recon_loss = self.beta_rec * (
            -jnp.sum(img_prior_dist.log_prob(img_posterior.astype(f32)), axis=-1)
        )

        if self.loss_reduction == "mean":
            kl_loss = kl_loss / self.num_classes

        total_loss = recon_loss + kl_loss

        return total_loss, (
            recon_loss,
            kl_loss,
        )

    def calculate_cyclical_lr(
        self, iteration, total_steps, num_cycles, hold_fraction=0.5
    ):
        step_size, hold_steps = self.calculate_step_size_and_hold_steps(
            total_steps, num_cycles, hold_fraction
        )

        cycle = iteration // (step_size + hold_steps)

        cycle_pos = iteration - (cycle * (step_size + hold_steps))

        if cycle_pos < step_size:
            return cycle_pos / step_size
        elif cycle_pos < step_size + hold_steps:
            return 1.0
        else:
            return 0.0

    def calculate_step_size_and_hold_steps(
        self, total_steps, num_cycles, hold_fraction=0.5
    ):
        hold_fraction = min(max(hold_fraction, 0), 1)
        steps_per_cycle = total_steps / num_cycles

        hold_steps = int(steps_per_cycle * hold_fraction)
        step_size = steps_per_cycle - hold_steps

        return step_size, hold_steps

    def get_latent_distribution(
        self, statistics: Union[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> tfd.Distribution:
        if self.latent_dist_type == "Categorical":
            return tfd.Independent(OneHotDist(statistics["logits"].astype(f32)), 1)
        elif self.latent_dist_type == "Gaussian":
            mean = statistics["mean"]
            std = statistics["std"]
            return tfd.MultivariateNormalDiag(mean, std)
        else:
            raise NotImplementedError("Latent distribution type not defined")

    def get_image_distribution(self, statistics: jnp.ndarray) -> tfd.Distribution:
        mode = statistics.reshape(statistics.shape[0], statistics.shape[1], -1).astype(
            f32
        )
        if self.image_dist_type == "MSE":
            return MSEDist(mode, 1, agg=self.loss_reduction)
        elif self.image_dist_type == "LogCosh":
            return LogCoshDist(mode, 1, agg=self.loss_reduction)
        else:
            raise NotImplementedError("Image distribution type not defined")

    def get_statistics(
        self,
        x: jnp.ndarray,
        statistics_head: str,
        unimix: float = 0.01,
    ) -> Dict[str, jnp.ndarray]:
        if self.discrete_latent_state:
            logits = self.statistic_heads[statistics_head](x)
            logits = logits.reshape(
                logits.shape[0], logits.shape[1], self.num_modes, self.num_classes
            )
            if unimix:
                probs = jax.nn.softmax(logits, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - unimix) * probs + unimix * uniform
                logits = jnp.log(probs)
            return {"logits": logits}
        else:
            x = self.statistic_heads[statistics_head](x)
            mean, std = jnp.split(x, 2, -1)
            std = nn.softplus(std) + 0.1
            print(std)

            return {"mean": mean, "std": std}

    def __call__(
        self,
        depth_imgs: jnp.ndarray,
        actions: jnp.ndarray,
        key,
        reconstruct_priors: bool = False,
    ) -> Tuple[tfd.Distribution, ...]:  # 3 tuple
        out = {
            "z_post": {"dist": None, "sample": None},
            "z_prior": {"dist": None, "sample": None},
            "depth": {"recon": None, "pred": None},
            "hidden": None,
        }

        post_key, prior_key = jax.random.split(key)
        multi_step = depth_imgs.shape[1] > 1

        # Compute low dimensional embedding from depth images and the latent posteriors from the embeddings
        embeddings = self.encoder(depth_imgs)
        out["z_post"]["sample"], out["z_post"]["dist"] = self.compute_posteriors(
            embeddings, post_key
        )

        # Concatenate and mix the latent posteriors and actions before processing trough the S4 blocks
        g = self.input_head(
            jnp.concatenate(
                (
                    (
                        out["z_post"]["sample"][:, :-1]
                        if multi_step
                        else out["z_post"]["sample"]
                    ),
                    actions,
                ),
                axis=-1,
            )
        )
        out["hidden"] = self.S4_blocks(g)

        # Compute the latent priors from the final hidden state of the sequence model
        out["z_prior"]["sample"], out["z_prior"]["dist"] = self.compute_priors(
            out["hidden"], prior_key
        )

        # Reconstruct depth images from the latent posteriors and the hidden state -> the reconstruction is conditioned on the history
        out["depth"]["recon"] = self.reconstruct_depth(
            out["hidden"],
            (out["z_post"]["sample"][:, 1:] if multi_step else out["z_post"]["sample"]),
        )

        if reconstruct_priors:
            out["depth"]["pred"] = self.reconstruct_depth(
                out["hidden"], out["z_prior"]["sample"]
            )

        return out

    def encode_and_step(
        self, image: jnp.ndarray, action: jnp.ndarray, latent: jnp.ndarray, key
    ) -> Tuple[jnp.ndarray, ...]:  # 2 Tuple
        z, _ = self.compute_posteriors(self.encoder(image), key)
        h = self.S4_blocks(self.input_head(jnp.concatenate((latent, action), axis=-1)))
        return z, h

    def encode(self, image: jnp.ndarray, key) -> jnp.ndarray:
        z, _ = self.compute_posteriors(self.encoder(image), key)
        return z

    # TODO: everything below here needs a refactoring pass

    def init_RNN_mode(self, params, init_imgs, init_actions) -> None:
        assert self.rnn_mode
        variables = self.init(
            jax.random.PRNGKey(0), init_imgs, init_actions, jax.random.PRNGKey(1)
        )
        vars = {
            "params": params,
            "cache": variables["cache"],
            "prime": variables["prime"],
        }

        _, prime_vars = self.apply(
            vars,
            init_imgs,
            init_actions,
            jax.random.PRNGKey(2),
            mutable=["prime", "cache"],
        )
        return vars["cache"], prime_vars["prime"]

    def forward_RNN_mode(
        self,
        imgs,
        actions,
        compute_reconstructions: bool = False,
    ) -> Tuple[tfd.Distribution, ...]:  # 3 Tuple
        assert self.rnn_mode
        return self.__call__(
            imgs,
            actions,
            compute_reconstructions,
        )

    def restore_checkpoint_state(self, ckpt_dir: str) -> dict:
        ckptr = orbax.checkpoint.Checkpointer(
            orbax.checkpoint.PyTreeCheckpointHandler()
        )
        ckpt_state = ckptr.restore(ckpt_dir, item=None)

        return ckpt_state

    # Dreaming utils
    def _build_context(
        self, context_imgs: jnp.ndarray, context_actions: jnp.ndarray
    ) -> None:
        posterior, _ = self.get_latent_posteriors_from_images(
            context_imgs, sample_mean=False
        )
        g = self.input_head(jnp.concatenate((posterior, context_actions), axis=-1))
        hidden = jnp.expand_dims(self.S4_blocks(g)[:, -1, :], axis=1)
        prior, _ = self.get_latent_prior_from_hidden(hidden, sample_mean=False)
        return prior, hidden

    def _open_loop_prediction(
        self, predicted_posterior: jnp.ndarray, next_action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, ...]:  # 2 tuple
        g = self.input_head(
            jnp.concatenate(
                (
                    predicted_posterior.reshape((-1, 1, self.latent_dim)),
                    next_action.reshape((-1, 1, self.num_actions)),
                ),
                axis=-1,
            )
        )
        hidden = self.S4_blocks(g)
        prior, _ = self.get_latent_prior_from_hidden(hidden, sample_mean=True)
        return prior, hidden

    def open_loop_prediction(
        self, predicted_posterior: jnp.ndarray, action: jnp.ndarray, key
    ) -> Tuple[jnp.ndarray, ...]:  # 2 tuple
        out = {
            "z_post_pred": {"dist": None, "sample": None},
            "depth_pred": None,
            "hidden": None,
        }
        g = self.input_head(
            jnp.concatenate(
                (
                    predicted_posterior,
                    action,
                ),
                axis=-1,
            )
        )
        out["hidden"] = self.S4_blocks(g)
        out["z_post_pred"]["sample"], out["z_post_pred"]["dist"] = self.compute_priors(
            out["hidden"], key=key
        )
        out["depth_pred"] = self.reconstruct_depth(
            out["hidden"], out["z_post_pred"]["sample"]
        )
        return out

    def _decode_predictions(
        self, hidden: jnp.ndarray, prior: jnp.ndarray
    ) -> jnp.ndarray:
        img_post = self.reconstruct_depth(hidden, prior)
        return img_post.mean()

    def dream(
        self,
        context_imgs: jnp.ndarray,
        context_actions: jnp.ndarray,
        dream_actions: jnp.ndarray,
        dream_horizon: int = 10,
    ) -> Tuple[jnp.ndarray, ...]:  # 3 Tuple

        prior, hidden = self._build_context(context_imgs, context_actions)
        priors = [prior]
        hiddens = [hidden]
        pred_depths = []

        for i in range(dream_horizon):
            prior, hidden = self._open_loop_prediction(
                predicted_posterior=prior, next_action=dream_actions[:, i]
            )
            priors.append(prior)
            hiddens.append(hidden)

        for x in zip(hiddens, priors):
            pred_depth = self._decode_predictions(
                jnp.expand_dims(x[0], axis=1), jnp.expand_dims(x[1], axis=1)
            )
            pred_depths.append(pred_depth)

        return pred_depths, priors


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(
    model,
    params,
    cache,
    prime,
    image: jax.Array,
    action: jax.Array,
    latent: jax.Array,
    key,
) -> jax.Array:
    return model.apply(
        {
            "params": params,
            "cache": cache,  # The hidden states of the SSM operating across the sequence
            "prime": prime,  # The SSM matrices, lambda, P, Q ...
        },
        image,
        action,
        latent,
        key,
        mutable=["cache"],
        method="encode_and_step",
    )


@partial(jax.jit, static_argnums=(0))
def _jitted_encode(model, params, image: jax.Array, key) -> jax.Array:
    return model.apply(
        {
            "params": params,
        },
        image,
        key,
        method="encode",
    )


class S4WMTorchWrapper:
    def __init__(
        self,
        batch_dim: int,
        ckpt_path: str,
        d_latent: int = 128,
        d_pssm_blocks: int = 1024,
        d_ssm: int = 64,
        num_pssm_blocks: int = 4,
        l_max: int = 99,
        sample_mean: bool = False,
    ) -> None:
        self.d_pssm_block = d_pssm_blocks
        self.d_ssm = d_ssm
        self.num_pssm_blocks = num_pssm_blocks

        self.model = S4WorldModel(
            S4_config=DictConfig(
                {
                    "d_model": d_pssm_blocks,
                    "layer": {"l_max": l_max, "N": d_ssm},
                    "n_blocks": num_pssm_blocks,
                }
            ),
            training=False,
            rnn_mode=True,
            sample_mean=sample_mean,
            latent_dist_type="Gaussian",
            **DictConfig(
                {
                    "latent_dim": d_latent,
                }
            ),
        )

        self.params = self.model.restore_checkpoint_state(ckpt_path)["params"]

        init_depth = jnp.zeros((batch_dim, 1, 135, 240, 1))
        init_actions = jnp.zeros((batch_dim, 1, 4))
        init_latent = jnp.zeros((batch_dim, 1, 128))

        self.rnn_cache, self.prime = self.model.init_RNN_mode(
            self.params,
            init_depth,
            init_actions,
        )

        self.rnn_cache, self.prime, self.params = (
            self.rnn_cache,
            self.prime,
            self.params,
        )

        self.key = jax.random.PRNGKey(0)

        # Force compilation
        _ = _jitted_forward(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            init_depth,
            init_actions,
            init_latent,
            self.key,
        )

        _ = _jitted_encode(self.model, self.params, init_depth, self.key)

        return

    def forward(
        self, depth_imgs: torch.tensor, actions: torch.tensor, latent: torch.tensor
    ) -> Tuple[torch.tensor, ...]:  # 2 tuple

        self.key, subkey = jax.random.split(self.key)

        jax_imgs, jax_actions, jax_latent = (
            from_torch_to_jax(depth_imgs),
            from_torch_to_jax(actions),
            from_torch_to_jax(latent),
        )

        out, variables = _jitted_forward(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            jax_imgs,
            jax_actions,
            jax_latent,
            subkey,
        )

        self.rnn_cache = variables["cache"]

        return (
            from_jax_to_torch(out[0]),
            from_jax_to_torch(out[1]),
        )

    def encode(self, depth_imgs: torch.tensor) -> torch.tensor:
        self.key, subkey = jax.random.split(self.key)
        jax_depth_imgs = from_torch_to_jax(depth_imgs)
        z = _jitted_encode(self.model, self.params, jax_depth_imgs, subkey)
        return from_jax_to_torch(z)

    def reset_cache(self, batch_idx: Sequence) -> None:
        batch_idx = from_torch_to_jax(batch_idx)
        for i in range(self.num_pssm_blocks):
            for j in range(2):
                self.rnn_cache["S4_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                    "cache_x_k"
                ] = (
                    self.rnn_cache["S4_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                        "cache_x_k"
                    ]
                    .at[jnp.array([batch_idx])]
                    .set(
                        jnp.zeros((self.d_ssm, self.d_pssm_block), dtype=jnp.complex64)
                    )
                )
        return


if __name__ == "__main__":
    # TODO: Add init code

    pass
