import jax.numpy as jnp
import jax

from flax import linen as nn
from jax.nn.initializers import normal

from .s4_ssm import (
    hippo_initializer,
    log_step_initializer,
    kernel_DPLR,
    discrete_DPLR,
    causal_convolution,
    scan_SSM,
)


class S4Blocks(nn.Module):
    layer: dict  # Extra arguments to pass into layer constructor
    d_model: int = 512
    n_layers: int = 2
    n_blocks: int = 2
    dropout: float = 0.1
    training: bool = True
    embedding: bool = False
    rnn_mode: bool = False

    def setup(self) -> None:
        self.dense = nn.Dense(features=self.d_model)
        self.blocks = [
            S4Block(
                layer=self.layer,
                d_model=self.d_model,
                n_layers=self.n_layers,
                dropout=self.dropout,
                training=self.training,
                embedding=self.embedding,
                rnn_mode=self.rnn_mode,
            )
            for _ in range(self.n_blocks)
        ]

    def __call__(self, x: jnp.ndarray) -> None:
        for block in self.blocks:
            x = block(x)
        return x


class S4Block(nn.Module):
    layer: dict
    d_model: int
    n_layers: int
    dropout: float = 0.0
    training: bool = True
    embedding: bool = False
    rnn_mode: bool = False

    def setup(self) -> None:
        self.norm = nn.LayerNorm()
        self.drop = nn.Dropout(
            self.dropout, broadcast_dims=[0], deterministic=not self.training
        )
        self.dense_1 = nn.Dense(features=self.d_model)
        self.dense_2 = nn.Dense(features=self.d_model)

        self.layers = [
            SequenceBlock(
                layer=self.layer,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                rnn_mode=self.rnn_mode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)

        skip = x
        x = self.norm(x)
        x = self.dense_1(x)
        x = nn.gelu(x)
        x = self.drop(x)
        x = self.dense_2(x)
        x = self.drop(x)
        x = x + skip

        return x


class SequenceBlock(nn.Module):
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    training: bool = True
    rnn_mode: bool = False

    def setup(self) -> None:
        self.seq = S4Layer(**self.layer, rnn_mode=self.rnn_mode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        skip = x
        x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        x = skip + self.drop(x)
        return x


class S4Layer(nn.Module):
    N: int = 256
    l_max: int = 1
    rnn_mode: bool = False

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
    }

    def setup(self) -> None:
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))

        self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))

        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = jnp.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.rnn_mode:
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )
        else:
            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C,
                    self.step,
                    self.l_max,
                )

            self.x_k_1 = self.variable(
                "cache", "cache_x_k", lambda: jnp.zeros((self.N,), dtype=jnp.complex64)
            )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
                self.x_k_1.value = jnp.zeros((self.N,), dtype=jnp.complex64)
            self.ssm = ssm_var.value

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        if not self.rnn_mode:
            # CNN Mode - paralell forward pass
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode - sequential forward pass
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache") and not self.is_mutable_collection(
                "prime"
            ):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


S4Layer = cloneLayer(S4Layer)

S4Blocks = nn.vmap(
    S4Blocks,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)
