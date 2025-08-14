import copy
import os

import flashbax as fbx
import jax
import numpy as np
import torch

from core.utils import torch_utils


class Agent:
    def __init__(
        self,
        exp_path,
        seed,
        env_fn,
        timeout,
        gamma,
        offline_data,
        action_dim,
        batch_size,
        use_target_network,
        target_network_update_freq,
        evaluation_criteria,
        logger,
    ):
        self.exp_path = exp_path
        self.seed = seed
        self.rng_key = jax.random.PRNGKey(seed)
        self.use_target_network = use_target_network
        self.target_network_update_freq = target_network_update_freq
        self.parameters_dir = self.get_parameters_dir()

        self.batch_size = batch_size
        self.env = env_fn()
        self.eval_env = copy.deepcopy(env_fn)()
        self.offline_data = offline_data
        self.replay = None
        self.state_normalizer = lambda x: x
        self.evaluation_criteria = evaluation_criteria
        self.logger = logger
        self.timeout = timeout
        self.action_dim = action_dim

        self.gamma = gamma
        self.device = 'cpu'
        self.stats_queue_size = 5
        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.reset = True
        self.ep_steps = 0
        self.num_episodes = 0
        self.ep_returns_queue_train = np.zeros(self.stats_queue_size)
        self.ep_returns_queue_test = np.zeros(self.stats_queue_size)
        self.train_stats_counter = 0
        self.test_stats_counter = 0
        self.agent_rng = np.random.RandomState(self.seed)

        self.populate_latest = False
        self.populate_states, self.populate_actions, self.populate_true_qs = None, None, None
        self.automatic_tmp_tuning = False

        self.state = None
        self.action = None
        self.next_state = None
        self.eps = 1e-8

    def get_parameters_dir(self):
        d = os.path.join(self.exp_path, "parameters")
        torch_utils.ensure_dir(d)
        return d

    def offline_param_init(self):
        self.trainset = self.training_set_construction(self.offline_data)
        self.training_size = len(self.trainset[0])
        self.training_indexs = np.arange(self.training_size)

        self.training_loss = []
        self.test_loss = []
        self.tloss_increase = 0
        self.tloss_rec = np.inf

    def get_data(self, rng_key):
        batch = self.replay.sample(self.replay_state, rng_key)
        states = batch.experience.first["s"]
        actions = batch.experience.first["a"]
        rewards = batch.experience.first["r"]
        next_states = batch.experience.second["s"]
        terminals = batch.experience.first["t"]
        in_ = self.state_normalizer(states)
        ns = self.state_normalizer(next_states)
        data = {
            "obs": in_,
            "act": actions,
            "reward": rewards,
            "obs2": ns,
            "done": terminals,
        }
        return data

    def fill_offline_data_to_buffer(self):
        self.trainset = self.training_set_construction(self.offline_data)
        train_s, train_a, train_r, train_ns, train_t = self.trainset
        dataset_size = len(train_s)
        dataset_transitions = {"s": train_s, "a": train_a, "r": train_r, "t": train_t}
        dummy_transition = jax.tree_util.tree_map(lambda x: x[0], dataset_transitions)
        self.replay = fbx.make_flat_buffer(
            max_length=dataset_size,
            min_length=self.batch_size,
            sample_batch_size=self.batch_size,
        )
        self.replay_state = self.replay.init(dummy_transition)
        add_fn = jax.jit(self.replay.add)

        def add_transition(carry, transition):
            replay_state = carry
            replay_state = add_fn(replay_state, transition)
            return replay_state, None

        self.replay_state, _ = jax.lax.scan(
            add_transition, self.replay_state, dataset_transitions
        )

    def step(self):
        # trans = self.feed_data()
        self.update_stats(0, None)
        self.rng_key, rng_key = jax.random.split(self.rng_key)
        data = self.get_data(rng_key)
        losses = self.update(data)
        return losses

    def update(self, data):
        raise NotImplementedError

    def update_stats(self, reward, done):
        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.num_episodes += 1
            if self.evaluation_criteria == "return":
                self.add_train_log(self.episode_reward)
            elif self.evaluation_criteria == "steps":
                self.add_train_log(self.ep_steps)
            else:
                raise NotImplementedError
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

    def add_train_log(self, ep_return):
        self.ep_returns_queue_train[self.train_stats_counter] = ep_return
        self.train_stats_counter += 1
        self.train_stats_counter = self.train_stats_counter % self.stats_queue_size

    def add_test_log(self, ep_return):
        self.ep_returns_queue_test[self.test_stats_counter] = ep_return
        self.test_stats_counter += 1
        self.test_stats_counter = self.test_stats_counter % self.stats_queue_size

    def populate_returns(self, log_traj=False, total_ep=None, initialize=False):
        total_ep = self.stats_queue_size if total_ep is None else total_ep
        total_steps = 0
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            ep_return, steps, traj = self.eval_episode(log_traj=log_traj)
            total_steps += steps
            total_states += traj[0]
            total_actions += traj[1]
            total_returns += traj[2]
            if self.evaluation_criteria == "return":
                self.add_test_log(ep_return)
                if initialize:
                    self.add_train_log(ep_return)
            elif self.evaluation_criteria == "steps":
                self.add_test_log(steps)
                if initialize:
                    self.add_train_log(steps)
            else:
                raise NotImplementedError
        return [total_states, total_actions, total_returns]

    def eval_episode(self, log_traj=False):
        ep_traj = []
        state = self.eval_env.reset()
        total_rewards = 0
        ep_steps = 0
        done = False
        while True:
            action = self.eval_step(state)
            last_state = state
            state, reward, done, _ = self.eval_env.step([action])
            # print(np.abs(state-last_state).sum(), "\n",action)
            if log_traj:
                ep_traj.append([last_state, action, reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.timeout:
                break

        states = []
        actions = []
        rets = []
        if log_traj:
            ret = 0
            for i in range(len(ep_traj)-1, -1, -1):
                s, a, r = ep_traj[i]
                ret = r + self.gamma * ret
                rets.insert(0, ret)
                actions.insert(0, a)
                states.insert(0, s)
        return total_rewards, ep_steps, [states, actions, rets]

    def log_return(self, log_ary, name, elapsed_time):
        rewards = log_ary
        total_episodes = len(self.episode_rewards)
        mean, median, min_, max_ = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)

        log_str = '%s LOG: steps %d, episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'

        self.logger.info(log_str % (name, self.total_steps, total_episodes, mean, median,
                                        min_, max_, len(rewards),
                                        elapsed_time))
        return mean, median, min_, max_

    def log_file(self, elapsed_time=-1, test=True):
        mean, median, min_, max_ = self.log_return(self.ep_returns_queue_train, "TRAIN", elapsed_time)
        if test:
            self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
            self.populate_latest = True
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            try:
                normalized = np.array([self.eval_env.env.unwrapped.get_normalized_score(ret_) for ret_ in self.ep_returns_queue_test])
                mean, median, min_, max_ = self.log_return(normalized, "Normalized", elapsed_time)
            except:
                pass
        return mean, median, min_, max_

    def policy(self, o, eval=False):
        o = torch_utils.tensor(self.state_normalizer(o), self.device)
        with torch.no_grad():
            a, _ = self.ac.pi(o, deterministic=eval)
        a = torch_utils.to_np(a)
        return a

    def eval_step(self, state):
        a = self.policy(state, eval=True)
        return a

    def training_set_construction(self, data_dict):
        assert len(list(data_dict.keys())) == 1
        data_dict = data_dict[list(data_dict.keys())[0]]
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        next_states = data_dict['next_states']
        terminations = data_dict['terminations']
        return [states, actions, rewards, next_states, terminations]
