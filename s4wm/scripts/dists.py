import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
from s4wm.nn.dists import MSEDist

tfd = tfp.distributions

if __name__ == "__main__":

    key = jax.random.PRNGKey(0)
    key_1 = jax.random.PRNGKey(1)
    mean = jax.random.normal(key, (4, 45, 270 * 480))
    mean_2 = jax.random.normal(key_1, (4, 45, 270 * 480))

    mean_kl_1 = jax.random.normal(key, (4, 45, 512))
    mean_kl_2 = jax.random.normal(key, (4, 45, 512))

    dist_kl_1 = tfd.MultivariateNormalDiag(mean_kl_1, jnp.ones_like(mean_kl_1))
    dist_kl_2 = tfd.MultivariateNormalDiag(mean_kl_2 + 0.05, jnp.ones_like(mean_kl_2))

    dist_1 = MSEDist(mean, 1, agg="sum")

    log_probs = jnp.sum(dist_1.log_prob(mean_2), axis=-1)

    kl_div = dist_kl_1.kl_divergence(dist_kl_2)

    print(kl_div)
