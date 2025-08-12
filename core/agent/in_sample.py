import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from core.agent import base
from core.network.network_architectures import (
    DoubleCriticDiscrete,
    DoubleCriticNetwork,
    FCNetwork,
)
from core.network.policy_factory import MLPCont, MLPDiscrete


@nnx.jit
def polyak_update(new, old, step_size):
    new_params = nnx.state(new)
    old_params = nnx.state(old)
    polyak_params = optax.incremental_update(new_params, old_params, step_size)
    nnx.update(old, polyak_params)
    return old


class InSampleAC(base.Agent):
    def __init__(
        self,
        discrete_control,
        state_dim,
        action_dim,
        hidden_units,
        learning_rate,
        tau,
        polyak,
        exp_path,
        seed,
        env_fn,
        timeout,
        gamma,
        offline_data,
        batch_size,
        use_target_network,
        target_network_update_freq,
        evaluation_criteria,
        logger,
    ):
        super(InSampleAC, self).__init__(
            exp_path=exp_path,
            seed=seed,
            env_fn=env_fn,
            timeout=timeout,
            gamma=gamma,
            offline_data=offline_data,
            action_dim=action_dim,
            batch_size=batch_size,
            use_target_network=use_target_network,
            target_network_update_freq=target_network_update_freq,
            evaluation_criteria=evaluation_criteria,
            logger=logger,
        )

        self.rng_key = jax.random.PRNGKey(seed)

        def get_policy_func(rngs):
            if discrete_control:
                return MLPDiscrete(state_dim, action_dim, [hidden_units] * 2, rngs=rngs)
            else:
                return MLPCont(state_dim, action_dim, [hidden_units] * 2, rngs=rngs)

        def get_critic_func(rngs):
            if discrete_control:
                return DoubleCriticDiscrete(
                    state_dim, [hidden_units] * 2, action_dim, rngs=rngs
                )
            else:
                return DoubleCriticNetwork(
                    state_dim, action_dim, [hidden_units] * 2, rngs=rngs
                )

        self.rng_key, pi_key, q_key, beh_pi_key, value_key = jax.random.split(
            self.rng_key, 5
        )

        pi_params_key, pi_sample_key = jax.random.split(pi_key)
        q_params_key, q_sample_key = jax.random.split(q_key)
        beh_pi_params_key, beh_pi_sample_key = jax.random.split(beh_pi_key)
        value_params_key, _ = jax.random.split(value_key)

        pi_rngs = nnx.Rngs(params=pi_params_key, sample=pi_sample_key)
        q_rngs = nnx.Rngs(params=q_params_key, sample=q_sample_key)
        beh_pi_rngs = nnx.Rngs(params=beh_pi_params_key, sample=beh_pi_sample_key)
        value_rngs = nnx.Rngs(params=value_params_key)

        self.pi = get_policy_func(pi_rngs)
        self.q = get_critic_func(q_rngs)
        self.beh_pi = get_policy_func(beh_pi_rngs)
        self.value_net = FCNetwork(
            jnp.prod(state_dim), [hidden_units] * 2, 1, rngs=value_rngs
        )

        self.pi_target = nnx.clone(self.pi)
        self.q_target = nnx.clone(self.q)

        self.pi_optimizer = nnx.Optimizer(self.pi, optax.adam(learning_rate))
        self.q_optimizer = nnx.Optimizer(self.q, optax.adam(learning_rate))
        self.value_optimizer = nnx.Optimizer(self.value_net, optax.adam(learning_rate))
        self.beh_pi_optimizer = nnx.Optimizer(self.beh_pi, optax.adam(learning_rate))

        self.exp_threshold = 10000
        self.tau = tau
        self.polyak = polyak
        self.fill_offline_data_to_buffer()
        self.offline_param_init()

        if discrete_control:
            self.get_q_value = self.get_q_value_discrete
        else:
            self.get_q_value = self.get_q_value_cont

    def get_q_value_discrete(self, q_net, o, a):
        q1_pi, q2_pi = q_net(o)
        q1_pi = jnp.take_along_axis(q1_pi, a[:, None], axis=1).squeeze(axis=1)
        q2_pi = jnp.take_along_axis(q2_pi, a[:, None], axis=1).squeeze(axis=1)
        q_pi = jnp.minimum(q1_pi, q2_pi)
        return q_pi, q1_pi, q2_pi

    def get_q_value_cont(self, q_net, o, a):
        q1_pi, q2_pi = q_net(o, a)
        q_pi = jnp.minimum(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    @partial(nnx.jit, static_argnums=(0,))
    def _update_beta(self, beh_pi, beh_pi_optimizer, data):
        def loss_fn(pi):
            log_probs = pi.get_logprob(data["obs"], data["act"])
            return -log_probs.mean()

        loss, grads = nnx.value_and_grad(loss_fn)(beh_pi)
        beh_pi_optimizer.update(grads)
        return loss

    @partial(nnx.jit, static_argnums=(0,))
    def _update_value(self, value_net, value_optimizer, pi, q_target, data, rngs):
        def loss_fn(value_net, rngs):
            v_phi = value_net(data["obs"]).squeeze(-1)
            actions, log_probs = pi(data["obs"], rngs=rngs)
            min_Q, _, _ = self.get_q_value(q_target, data["obs"], actions)
            target = min_Q - self.tau * log_probs
            value_loss = (0.5 * (v_phi - target) ** 2).mean()
            return value_loss, (v_phi, log_probs)

        (loss, (v_phi, log_probs)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            value_net, rngs
        )
        value_optimizer.update(grads)
        return loss, v_phi, log_probs

    @partial(nnx.jit, static_argnums=(0,))
    def _update_q(self, q_net, q_optimizer, pi, q_target, data, rngs):
        def loss_fn(q_net, rngs):
            next_actions, log_probs = pi(data["obs2"], rngs=rngs)
            min_Q, _, _ = self.get_q_value(q_target, data["obs2"], next_actions)
            q_target_values = data["reward"] + self.gamma * (1 - data["done"]) * (
                min_Q - self.tau * log_probs
            )

            min_q, q1, q2 = self.get_q_value(q_net, data["obs"], data["act"])

            critic1_loss = (0.5 * (q_target_values - q1) ** 2).mean()
            critic2_loss = (0.5 * (q_target_values - q2) ** 2).mean()
            loss_q = (critic1_loss + critic2_loss) * 0.5
            return loss_q, min_q

        (loss, q_info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(q_net, rngs)
        q_optimizer.update(grads)
        return loss, q_info

    @partial(nnx.jit, static_argnums=(0,))
    def _update_pi(self, pi, pi_optimizer, q, value_net, beh_pi, data):
        def loss_fn(pi):
            log_probs = pi.get_logprob(data["obs"], data["act"])
            min_Q, _, _ = self.get_q_value(q, data["obs"], data["act"])
            value = value_net(data["obs"]).squeeze(-1)
            beh_log_prob = beh_pi.get_logprob(data["obs"], data["act"])

            clipped = jnp.clip(
                jnp.exp((min_Q - value) / self.tau - beh_log_prob),
                self.eps,
                self.exp_threshold,
            )
            pi_loss = -(clipped * log_probs).mean()
            return pi_loss

        loss, grads = nnx.value_and_grad(loss_fn)(pi)
        pi_optimizer.update(grads)
        return loss

    @partial(nnx.jit, static_argnums=(0, 3))
    def _policy(self, pi, o, deterministic, rngs):
        o = self.state_normalizer(o)
        a, _ = pi(o, deterministic=deterministic, rngs=rngs)
        return a

    @partial(nnx.jit, static_argnums=(0,))
    def _sync_target(self, pi, pi_target, q, q_target, total_steps):
        def sync(pi, pi_target, q, q_target):
            pi_target = polyak_update(pi, pi_target, 1 - self.polyak)
            q_target = polyak_update(q, q_target, 1 - self.polyak)
            return pi_target, q_target

        def no_sync(pi, pi_target, q, q_target):
            return pi_target, q_target

        pi_target, q_target = nnx.cond(
            jnp.logical_and(
                self.use_target_network,
                total_steps % self.target_network_update_freq == 0,
            ),
            sync,
            no_sync,
            pi,
            pi_target,
            q,
            q_target,
        )
        return pi_target, q_target

    def update(self, data):
        self.rng_key, value_key, q_key = jax.random.split(self.rng_key, 3)
        value_rngs = nnx.Rngs(sample=value_key)
        q_rngs = nnx.Rngs(sample=q_key)

        data = jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)
        (
            loss_beta,
            loss_vs,
            v_info,
            logp_info,
            loss_q,
            qinfo,
            loss_pi,
            self.pi_target,
            self.q_target,
        ) = self._update(
            self.beh_pi,
            self.beh_pi_optimizer,
            self.value_net,
            self.value_optimizer,
            self.pi,
            self.pi_target,
            self.q_target,
            self.q,
            self.q_optimizer,
            self.pi_optimizer,
            data,
            value_rngs,
            q_rngs,
            self.total_steps,
        )

        return {
            "beta": loss_beta.item(),
            "actor": loss_pi.item(),
            "critic": loss_q.item(),
            "value": loss_vs.item(),
            "q_info": qinfo.mean().item(),
            "v_info": v_info.mean().item(),
            "logp_info": logp_info.mean().item(),
        }

    @partial(nnx.jit, static_argnums=(0,))
    def _update(
        self,
        beh_pi,
        beh_pi_optimizer,
        value_net,
        value_optimizer,
        pi,
        pi_target,
        q_target,
        q,
        q_optimizer,
        pi_optimizer,
        data,
        value_rngs,
        q_rngs,
        total_steps,
    ):
        loss_beta = self._update_beta(beh_pi, beh_pi_optimizer, data)
        loss_vs, v_info, logp_info = self._update_value(
            value_net, value_optimizer, pi, q_target, data, value_rngs
        )
        loss_q, qinfo = self._update_q(q, q_optimizer, pi, q_target, data, q_rngs)
        loss_pi = self._update_pi(pi, pi_optimizer, q, value_net, beh_pi, data)
        pi_target, q_target = self._sync_target(pi, pi_target, q, q_target, total_steps)
        return (
            loss_beta,
            loss_vs,
            v_info,
            logp_info,
            loss_q,
            qinfo,
            loss_pi,
            pi_target,
            q_target,
        )

    def policy(self, o, eval=False):
        self.rng_key, policy_key = jax.random.split(self.rng_key)
        policy_rngs = nnx.Rngs(sample=policy_key)
        a = self._policy(self.pi, o, deterministic=eval, rngs=policy_rngs)
        return np.asarray(a)

    def eval_step(self, state: np.ndarray):
        state = jnp.asarray(state)
        a = self.policy(state, eval=True)
        return a

    def save(self):
        _, pi_state = nnx.split(self.pi)
        _, q_state = nnx.split(self.q)
        _, value_net_state = nnx.split(self.value_net)
        _, beh_pi_state = nnx.split(self.beh_pi)
        _, pi_optimizer_state = nnx.split(self.pi_optimizer)
        _, q_optimizer_state = nnx.split(self.q_optimizer)
        _, value_optimizer_state = nnx.split(self.value_optimizer)
        _, beh_pi_optimizer_state = nnx.split(self.beh_pi_optimizer)

        ckpt = {
            "pi": pi_state,
            "q": q_state,
            "value_net": value_net_state,
            "beh_pi": beh_pi_state,
            "pi_optimizer": pi_optimizer_state,
            "q_optimizer": q_optimizer_state,
            "value_optimizer": value_optimizer_state,
            "beh_pi_optimizer": beh_pi_optimizer_state,
        }
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(
                os.path.join(self.parameters_dir, "default"), ckpt, force=True
            )

    def load(self):
        with ocp.StandardCheckpointer() as checkpointer:
            ckpt = checkpointer.restore(os.path.join(self.parameters_dir, "default"))
        nnx.merge(self.pi, ckpt["pi"])
        nnx.merge(self.q, ckpt["q"])
        nnx.merge(self.value_net, ckpt["value_net"])
        nnx.merge(self.beh_pi, ckpt["beh_pi"])
        nnx.merge(self.pi_optimizer, ckpt["pi_optimizer"])
        nnx.merge(self.q_optimizer, ckpt["q_optimizer"])
        nnx.merge(self.value_optimizer, ckpt["value_optimizer"])
        nnx.merge(self.beh_pi_optimizer, ckpt["beh_pi_optimizer"])
