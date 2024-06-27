import os
import hydra
import jax
import jax.numpy as np
import optax
import torch
import shutil

from functools import partial
from flax.training import checkpoints, train_state
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Any

from s4wm.nn.s4_wm import S4WorldModel
from s4wm.nn.s4_nn import S4Layer
from s4wm.data.dataloaders import Dataloaders
from s4wm.utils.dlpack import from_torch_to_jax

try:
    # Slightly nonstandard import name to make config easier - see example_train()
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None


class TrainState(train_state.TrainState):
    batch_stats: Any


def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(
    rng,
    model_cls,
    trainloader,
    lr=1e-3,
    lr_layer=None,
    lr_schedule=False,
    use_batchmean=False,
    weight_decay=0.0,
    total_steps=-1,
):
    model = model_cls(training=True)
    init_rng, dropout_rng, sample_rng = jax.random.split(rng, num=3)

    init_depth, init_actions, _ = next(iter(trainloader))
    init_depth, init_actions = from_torch_to_jax(init_depth), from_torch_to_jax(
        init_actions
    )

    parameters = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        init_depth,
        init_actions,
        sample_rng,
    )

    params = parameters["params"]

    if use_batchmean:
        batch_stats = parameters["batch_stats"]
    else:
        batch_stats = None

    if lr_schedule:
        schedule_fn = lambda lr: optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=lr,
            warmup_steps=1000,
            decay_steps=total_steps,
        )
    else:
        schedule_fn = lambda lr: lr
    # lr_layer is a dictionary from parameter name to LR multiplier
    if lr_layer is None:
        lr_layer = {}

    optimizers = {
        k: optax.chain(
            optax.clip_by_global_norm(1000),
            optax.adamw(learning_rate=schedule_fn(v * lr), weight_decay=weight_decay),
        )
        for k, v in lr_layer.items()
    }

    optimizers["__default__"] = optax.chain(
        optax.clip_by_global_norm(1000),
        optax.adamw(
            learning_rate=schedule_fn(lr),
            weight_decay=weight_decay,
        ),
    )
    name_map = map_nested_fn(lambda k, _: k if k in lr_layer else "__default__")
    tx = optax.multi_transform(optimizers, name_map)

    # Check that all special parameter names are actually parameters
    extra_keys = set(lr_layer.keys()) - set(jax.tree_leaves(name_map(params)))
    assert (
        len(extra_keys) == 0
    ), f"Special params {extra_keys} do not correspond to actual params"

    # Print parameter count
    _is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(
        lambda k, param: (
            param.size * (2 if _is_complex(param) else 1)
            if lr_layer.get(k, lr) > 0.0
            else 0
        )
    )(params)

    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Total training steps: {total_steps}")

    if use_batchmean:
        return TrainState.create(
            apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=tx
        )
    else:
        return TrainState.create(
            apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=tx
        )


def train_epoch(state, rng, model_cls, trainloader, beta_warmup=True):
    model = model_cls(training=True)
    batch_losses = []

    for batch_depth, batch_actions, batch_labels in tqdm(trainloader):
        rng, drop_rng, sample_rng = jax.random.split(rng, num=3)

        state, batch_loss, recon_loss, kld_loss = train_step(
            state,
            drop_rng,
            sample_rng,
            from_torch_to_jax(batch_depth),
            from_torch_to_jax(batch_actions),
            from_torch_to_jax(batch_labels),
            model,
            1,
        )
        batch_losses.append(batch_loss)

        wandb.log(
            {
                "train/recon_loss": recon_loss.tolist(),
                "train/kld_loss": kld_loss.tolist(),
            }
        )

    return (
        state,
        np.mean(np.array(batch_losses)),
    )


def validate(state, rng, model_cls, testloader):
    losses = []
    model = model_cls(training=False)

    for batch_depth, batch_actions, batch_labels in tqdm(testloader):

        loss = eval_step(
            state,
            rng,
            from_torch_to_jax(batch_depth),
            from_torch_to_jax(batch_actions),
            from_torch_to_jax(batch_labels),
            model,
        )

        losses.append(loss)
    return np.mean(np.array(losses))


@partial(jax.jit, static_argnums=6)
def train_step(
    state,
    drop_rng,
    sample_rng,
    batch_depth,
    batch_actions,
    batch_depth_labels,
    model,
    beta_rate=1,
):

    def loss_fn(params):
        out, updates = None, None

        if state.batch_stats is not None:
            out, updates = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                depth_imgs=batch_depth,
                actions=batch_actions,
                key=sample_rng,
                rngs={"dropout": drop_rng},
                mutable=["batch_stats"],
            )
        else:
            out = model.apply(
                {"params": params},
                depth_imgs=batch_depth,
                actions=batch_actions,
                key=sample_rng,
                rngs={"dropout": drop_rng},
            )

        loss, (recon_loss, kl_loss) = model.compute_loss(
            img_prior_dist=out["depth"]["recon"],
            img_posterior=batch_depth_labels,
            z_posterior_dist=out["z_post"]["dist"][:, 1:],
            z_prior_dist=out["z_prior"]["dist"],
            beta_rate=beta_rate,
        )

        return np.mean(loss), (
            np.mean(recon_loss),
            np.mean(kl_loss),
            updates,
        )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon_loss, kl_loss, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    if updates is not None:
        state = state.replace(batch_stats=updates["batch_stats"])

    return state, loss, recon_loss, kl_loss


@partial(jax.jit, static_argnums=5)
def eval_step(state, rng, batch_depth, batch_actions, batch_depth_labels, model):

    if state.batch_stats is not None:
        out = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats},
            batch_depth,
            batch_actions,
            rng,
        )
    else:
        out = model.apply({"params": state.params}, batch_depth, batch_actions, rng)

    loss, _ = model.compute_loss(
        img_prior_dist=out["depth"]["recon"],
        img_posterior=batch_depth_labels,
        z_posterior_dist=out["z_post"]["dist"][:, 1:],
        z_prior_dist=out["z_prior"]["dist"],
    )

    loss = np.mean(loss)

    return loss


def train(
    dataset: str,
    seed: int,
    wm: DictConfig,
    model: DictConfig,
    train: DictConfig,
):
    print("[*] Setting Randomness...")

    key = jax.random.PRNGKey(seed)
    key, rng, train_rng, val_rng = jax.random.split(key, num=4)
    torch.manual_seed(0)  # For torch dataloader order

    # Create dataset and data loaders
    create_dataloaders_fn = Dataloaders[dataset]
    trainloader, testloader = create_dataloaders_fn(batch_size=train.bsz)

    # Get model class and arguments
    layer_cls = S4Layer
    lr_layer = getattr(layer_cls, "lr", None)

    print(
        f"[*] Starting S4 World Model Training On Dataset: {dataset} =>> Initializing..."
    )

    model_cls = partial(S4WorldModel, S4_config=model, **wm)

    state = create_train_state(
        rng,
        model_cls,
        trainloader,
        lr=train.lr,
        lr_layer=lr_layer,
        lr_schedule=train.lr_schedule,
        weight_decay=train.weight_decay,
        total_steps=len(trainloader) * train.epochs,
    )

    # Loop over epochs
    best_loss, best_epoch = np.inf, 0
    for epoch in range(train.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        state, train_loss = train_epoch(
            state, train_rng, model_cls, trainloader, beta_warmup=True
        )

        print(f"[*] Running Epoch {epoch + 1} Validation...")

        val_loss = validate(state, val_rng, model_cls, testloader)

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(f"\tTrain Loss: {train_loss:.5f} -- Train Loss:")
        print(f"\tVal Loss: {val_loss:.5f} -- Train Loss:")

        if val_loss < best_loss:

            run_id = f"{os.path.dirname(os.path.realpath(__file__))}/checkpoints/{dataset}/d_model={model.d_model}-lr={train.lr}-bsz={train.bsz}-latent_type={wm.latent_dist_type}-num_blocks={model.n_blocks}-num_layers={model.n_layers}"
            if epoch > 0:
                shutil.rmtree(f"{run_id}/checkpoint_{best_epoch}")
            _ = checkpoints.save_checkpoint(
                run_id,
                state,
                epoch,
                keep=train.epochs,
            )

            best_loss, best_epoch = val_loss, epoch

        print(f"\tBest Test Loss: {best_loss:.5f}")

        if wandb is not None:
            wandb.run.summary["Best Test Loss"] = best_loss.tolist()
            wandb.run.summary["Best Epoch"] = best_epoch

        key, train_rng, val_rng = jax.random.split(key, num=3)


@hydra.main(version_base=None, config_path=".", config_name="train_cfg")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["XLA_CLIENT_PREALLOCATE"] = "True"

    print(OmegaConf.to_yaml(cfg))

    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    # Track with wandb
    if wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        wandb.init(**wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True))

    train(**cfg)


def calculate_cyclical_lr(iteration, total_steps, num_cycles, hold_fraction=0.5):
    step_size, hold_steps = calculate_step_size_and_hold_steps(
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


def calculate_step_size_and_hold_steps(total_steps, num_cycles, hold_fraction=0.5):
    hold_fraction = min(max(hold_fraction, 0), 1)
    steps_per_cycle = total_steps / num_cycles

    hold_steps = int(steps_per_cycle * hold_fraction)
    step_size = steps_per_cycle - hold_steps

    return step_size, hold_steps


if __name__ == "__main__":
    main()
