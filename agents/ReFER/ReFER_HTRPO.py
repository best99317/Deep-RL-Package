import torch
import numpy as np
from collections import deque
from agents.VPG.run_vpg import explained_variance, adjust_learning_rate


def run_refer_htrpo_train(env, agent, max_time_steps, logger, eval_interval, num_evals):
    time_step_counter = 0
    total_updates = max_time_steps // agent.steps_per_iter
    epi_info_buf = deque(maxlen=100)
    # used for computing success rate
    success_history = deque(maxlen=100)  # TODO: ep_num = 0

    # if eval_interval:
    #     # eval_ret, eval_suc = agent.eval_brain(env, render=False, eval_num=1000)
    #     eval_ret = agent.eval_policy(env, eval_num=num_evals)
    #     print("evaluation_eprew:".ljust(20) + str(np.mean(eval_ret)))
    #     # print("evaluation_sucrate:".ljust(20) + str(np.mean(eval_suc)))
    #     logger.add_scalar("episode_reward/train", np.mean(eval_ret), time_step_counter)

    while True:
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_logpi_old, mb_obs_, mb_mus, mb_sigmas \
            , mb_distributions = [], [], [], [], [], [], [], [], []
        mb_dg, mb_ag = [], []
        epi_infos = []
        successes = []
        obs_dict = env.reset()
        ep_num = 0

        for i in range(0, agent.steps_per_iter, env.num_envs):
            for key in obs_dict.keys():
                obs_dict[key] = torch.Tensor(obs_dict[key])

            if not agent.discrete_action:
                # TODO: Ignore this for the moment
                actions, mus, logsigmas, sigmas = \
                    agent.choose_action(obs_dict['observation'], other_data=obs_dict["desired_goal"])
                logpi = agent.compute_logp(mus, logsigmas, sigmas, actions)
                mus = mus.cpu().numpy()
                sigmas = sigmas.cpu().numpy()
                mb_mus.append(mus)
                mb_sigmas.append(sigmas)
            else:
                actions, distributions = \
                    agent.choose_action(obs_dict['observation'], other_data=obs_dict["desired_goal"])
                logpi = agent.compute_logp(distributions, actions)
                distributions = distributions.cpu().numpy()
                mb_distributions.append(distributions)

            observations = obs_dict['observation'].cpu().numpy()
            actions = actions.cpu().numpy()
            logpi = logpi.cpu().numpy()

            if np.random.rand() < 0.0:
                actions = np.concatenate([np.expand_dims(env.action_space.sample(), axis=0)
                                          for i in range(env.num_envs)], axis=0)
                obs_dict_, rewards, dones, infos = env.step(actions)
            else:
                obs_dict_, rewards, dones, infos = env.step(actions)

            # if timestep_counter > 350000:
            # env.render()

            mb_obs.append(observations)
            mb_actions.append(actions)
            mb_logpi_old.append(logpi)
            mb_dones.append(dones.astype(np.uint8))
            mb_rewards.append(rewards)
            mb_obs_.append(obs_dict_['observation'].copy())
            mb_dg.append(obs_dict_['desired_goal'].copy())
            mb_ag.append(obs_dict_['achieved_goal'].copy())

            for e, info in enumerate(infos):
                if dones[e]:
                    if 'reward' not in info['episode']:
                        info['episode']['reward'] = info['episode'].pop('r')
                        info['episode']['length'] = info['episode'].pop('l')
                    epi_infos.append(info.get('episode'))
                    successes.append(info.get('is_success'))
                    for k in obs_dict_.keys():
                        obs_dict_[k][e] = info.get('new_obs')[k]
                    ep_num += 1

            obs_dict = obs_dict_

        epi_info_buf.extend(epi_infos)
        success_history.extend(successes)

        # make all final states marked by done, preventing wrong estimating of returns and advantages.
        # done flag:
        #      0: undone and not the final state
        #      1: realdone
        #      2: undone but the final state
        ep_num += (mb_dones[-1] == 0).sum()
        mb_dones[-1][np.where(mb_dones[-1] == 0)] = 2

        def reshape_data(arr):
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        mb_obs = reshape_data(np.asarray(mb_obs, dtype=np.float32))
        mb_rewards = reshape_data(np.asarray(mb_rewards, dtype=np.float32))
        mb_actions = reshape_data(np.asarray(mb_actions))
        mb_logpi_old = reshape_data(np.asarray(mb_logpi_old, dtype=np.float32))
        mb_dones = reshape_data(np.asarray(mb_dones, dtype=np.uint8))
        mb_obs_ = reshape_data(np.asarray(mb_obs_, dtype=np.float32))
        mb_ag = reshape_data(np.asarray(mb_ag, dtype=np.float32))
        mb_dg = reshape_data(np.asarray(mb_dg, dtype=np.float32))

        assert mb_rewards.ndim <= 2 and mb_actions.ndim <= 2 and \
               mb_logpi_old.ndim <= 2 and mb_dones.ndim <= 2, \
            "databuffer only supports 1-D data's batch."

        if not agent.discrete_action:
            mb_mus = reshape_data(np.asarray(mb_mus, dtype=np.float32))
            mb_sigmas = reshape_data(np.asarray(mb_sigmas, dtype=np.float32))
            assert mb_mus.ndim <= 2 and mb_sigmas.ndim <= 2, "databuffer only supports 1-D data's batch."
        else:
            mb_distributions = reshape_data(np.asarray(mb_distributions, dtype=np.float32))
            assert mb_distributions.ndim <= 2, "databuffer only supports 1-D data's batch."

        # store transition
        transition = {
            'state': mb_obs if mb_obs.ndim == 2 else np.expand_dims(mb_obs, 1),
            'action': mb_actions if mb_actions.ndim == 2 else np.expand_dims(mb_actions, 1),
            'reward': mb_rewards if mb_rewards.ndim == 2 else np.expand_dims(mb_rewards, 1),
            'next_state': mb_obs_ if mb_obs_.ndim == 2 else np.expand_dims(mb_obs_, 1),
            'done': mb_dones if mb_dones.ndim == 2 else np.expand_dims(mb_dones, 1),
            'logpi': mb_logpi_old if mb_logpi_old.ndim == 2 else np.expand_dims(mb_logpi_old, 1),
            'other_data': {
                'desired_goal': mb_dg if mb_dg.ndim == 2 else np.expand_dims(mb_dg, 1),
                'achieved_goal': mb_ag if mb_ag.ndim == 2 else np.expand_dims(mb_ag, 1),
            }
        }
        if not agent.discrete_action:
            transition['mu'] = mb_mus if mb_mus.ndim == 2 else np.expand_dims(mb_mus, 1)
            transition['sigma'] = mb_sigmas if mb_sigmas.ndim == 2 else np.expand_dims(mb_sigmas, 1)
        else:
            transition['distribution'] = \
                mb_distributions if mb_distributions.ndim == 2 else np.expand_dims(mb_distributions, 1)
        agent.store_transition(transition)

        # agent learning step
        agent.update_policy()

        # training controller
        time_step_counter += agent.steps_per_iter
        if time_step_counter >= max_time_steps:
            break

        print("------------------log information------------------")
        print("total_timesteps:".ljust(20) + str(time_step_counter))
        print("iterations:".ljust(20) + str(agent.learn_step_counter) + " / " + str(int(total_updates)))
        if agent.value_func_type is not None:
            explained_var = explained_variance(agent.V.cpu().numpy(), agent.estimated_R.cpu().numpy())
            print("explained_var:".ljust(20) + str(explained_var))
            logger.add_scalar("explained_var/train", explained_var, time_step_counter)
        print(
            "episode_len:".ljust(20) + "{:.1f}".format(np.mean([epi_info['length'] for epi_info in epi_info_buf])))
        print("episode_rew:".ljust(20) + str(np.mean([epi_info['reward'] for epi_info in epi_info_buf])))
        logger.add_scalar("episode_reward/train",
                          np.mean([epi_info['reward'] for epi_info in epi_info_buf]), time_step_counter)
        print("success_rate:".ljust(20) + "{:.3f}".format(100 * np.mean(success_history)) + "%")
        logger.add_scalar("success_rate/train", np.mean(success_history), time_step_counter)
        # print("mean_kl:".ljust(20) + str(agent.cur_kl))
        # logger.add_scalar("mean_kl/train", agent.cur_kl, time_step_counter)
        # print("policy_ent:".ljust(20) + str(agent.policy_ent))
        # logger.add_scalar("policy_ent/train", agent.policy_ent, time_step_counter)
        # print("policy_loss:".ljust(20)+ str(agent.policy_loss))
        # logger.add_scalar("policy_loss/train", agent.policy_loss, time_step_counter)
        print("value_loss:".ljust(20) + str(agent.value_loss))
        logger.add_scalar("value_loss/train", agent.value_loss, time_step_counter)
        print("valid_ep_ratio:".ljust(20) + "{:.3f}".format(agent.num_valid_ep / ep_num))
        logger.add_scalar("valid_ep_ratio/train", agent.num_valid_ep / ep_num, time_step_counter)
        # TODO: ep_num = 0
    return agent
