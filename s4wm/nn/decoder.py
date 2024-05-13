import jax.numpy as jnp
import os
import jax
import time

from jax import random
from flax import linen as nn
from jax.nn.initializers import glorot_uniform, zeros
from functools import partial


class ImageDecoder(nn.Module):
    act: str = "silu"
    process_in_chunks: bool = False

    def setup(self) -> None:

        if self.act == "elu":
            self.act_fn = nn.elu
        elif self.act == "gelu":
            self.act_fn = nn.gelu
        elif self.act == "silu":
            self.act_fn = nn.silu
        else:
            self.act_fn = lambda x: x

        glorot_init = (
            glorot_uniform()
        )  # Equivalent to Pytorch's Xavier Uniform https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.glorot_uniform.html

        self.dense_00 = nn.Dense(features=512, kernel_init=glorot_init, bias_init=zeros)
        self.dense_01 = nn.Dense(
            features=9 * 15 * 128, kernel_init=glorot_init, bias_init=zeros
        )

        self.deconv_1 = nn.ConvTranspose(
            features=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_2 = nn.ConvTranspose(
            features=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_3 = nn.ConvTranspose(
            features=32,
            kernel_size=(7, 6),
            strides=(4, 4),
            padding=(3, 4),
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_4 = nn.ConvTranspose(
            features=16,
            kernel_size=(3, 4),
            strides=(2, 2),
            padding=(0, 2),
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.deconv_5 = nn.ConvTranspose(
            features=1,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )

    def _upsample(self, latent: jnp.ndarray) -> jnp.ndarray:
        x = self.dense_00(latent)
        x = self.act_fn(x)
        x = self.dense_01(x)
        x = x.reshape(x.shape[0], x.shape[1], 9, 15, 128)

        x = self.deconv_1(x)
        x = self.act_fn(x)

        x = self.deconv_2(x)
        x = self.act_fn(x)

        x = self.deconv_3(x)
        x = self.act_fn(x)

        x = self.deconv_4(x)
        x = self.act_fn(x)

        x = self.deconv_5(x)
        x = nn.sigmoid(x)

        return jnp.squeeze(x, axis=-1)

    def __call__(self, latents: jnp.ndarray) -> jnp.ndarray:
        # Running the forward pass in chunks requires less contiguous memory

        if self.process_in_chunks:
            chunks = jnp.array_split(latents, 4, axis=1)
            downsampled_chunks = [self._upsample(chunk) for chunk in chunks]

            return jnp.concatenate(downsampled_chunks, axis=1)
        else:
            return self._upsample(latents)


class Decoder(nn.Module):
    c_out: int
    c_hid: int
    discrete_latent_state: bool

    @nn.compact
    def __call__(self, x):
        if self.discrete_latent_state:
            x = nn.Dense(features=4 * 8 * 2 * self.c_hid)(x)  # 1xc_hid for 2048 model
            x = nn.silu(x)
            x = x.reshape(x.shape[0], x.shape[1], 4, 8, -1)
        else:
            x = nn.Dense(features=4 * 8 * 2 * self.c_hid)(x)
            x = nn.silu(x)
            x = x.reshape(x.shape[0], x.shape[1], 4, 8, -1)

        x = nn.ConvTranspose(
            features=2 * self.c_hid,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=(2, 2),
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=2 * self.c_hid,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=(2, 1),
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=self.c_hid,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=(3, 2),
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=self.c_out,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=(2, 2),
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=self.c_out,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.sigmoid(jnp.squeeze(x[:, :, :-1, :], axis=-1))
        return x


resnet_kernel_init = nn.initializers.variance_scaling(
    2.0, mode="fan_out", distribution="normal"
)


class ResNetBlockDecoder(nn.Module):
    act_fn: callable  # Activation function
    c_out: int  # Output feature size
    subsample: bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x):
        # Network representing F
        z = nn.ConvTranspose(
            self.c_out,
            kernel_size=(2, 2),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        # z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(z)
        # z = nn.BatchNorm()(z, use_running_average=not train)
        if self.subsample:
            x = nn.ConvTranspose(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=resnet_kernel_init,
            )(x)

        x_out = self.act_fn(z + x)
        return x_out


class ResNetDecoder(nn.Module):
    act_fn: callable = nn.silu
    block_class: nn.Module = ResNetBlockDecoder
    num_blocks: tuple = (1, 1, 1)
    c_hidden: tuple = (64, 32, 16)

    @nn.compact
    def __call__(self, x):
        # A first convolution on the original image to scale up the channel size
        x = nn.Dense(features=5 * 8 * self.c_hidden[0])(x)
        x = self.act_fn(x)
        x = x.reshape(x.shape[0], x.shape[1], 5, 8, self.c_hidden[0])

        x = nn.ConvTranspose(
            self.c_hidden[0],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(1, 1),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        x = self.act_fn(x)
        x = nn.ConvTranspose(
            self.c_hidden[0],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(1, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)

        if (
            self.block_class == ResNetBlockDecoder
        ):  # If pre-activation block, we do not apply non-linearities yet
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = bc == 0 and block_idx > 0
                # ResNet block
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample,
                )(x)

        # Mapping to classification output
        x = nn.ConvTranspose(
            1,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)

        x = nn.sigmoid(jnp.squeeze(x[:, :, :-1, :-8], axis=-1))
        return x


@partial(jax.jit, static_argnums=(0))
def jitted_forward(model, params, latent):
    return model.apply(
        {
            "params": jax.lax.stop_gradient(params),
        },
        jax.lax.stop_gradient(latent),
    )


if __name__ == "__main__":
    # Test decoder implementation
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    key = random.PRNGKey(0)
    decoder = ResNetDecoder(act_fn=nn.silu, block_class=ResNetBlockDecoder)

    random_latent_batch = random.normal(key, (1, 2, 2048))

    params = decoder.init(key, random_latent_batch)["params"]
    output = decoder.apply({"params": params}, random_latent_batch)
    print("Output shape: ", output.shape)

    # _ = jitted_forward(decoder, params, random_latent_batch)

    # fnc = jax.vmap(jitted_forward)
    # fwp_times = []
    # for _ in range(200):
    #     start = time.time()
    #     _ = jitted_forward(decoder, params, random_latent_batch)
    #     end = time.time()
    #     print(end - start)
    #     fwp_times.append(end - start)
    # fwp_times = jnp.array(fwp_times)
