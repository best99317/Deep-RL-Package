import argparse
import os
import torch
import numpy as np
from gym import spaces
from tensorboardX import SummaryWriter

from utils.envbuilder import build_env
from utils.vecenv import space_dim
from agents.VPG.run_vpg import run_vpg_train
from agents.NPG.run_npg import run_npg_train
from agents.TRPO.run_trpo import run_trpo_train
from agents.HPG.run_hpg import run_hpg_train
from agents.HTRPO.run_htrpo import run_htrpo_train
from agents.ReFER.ReFER_HTRPO import run_refer_htrpo_train
# The following lines cannot be deleted for they are used in eval()
import agents
from agents.config import *
from configs.VPG_configs import VPG_FlipBit8, VPG_FlipBit4
from configs.TRPO_configs import TRPO_FlipBit8, TRPO_EmptyMaze, TRPO_FourRoomMaze, TRPO_FetchReachv1
from configs.HPG_configs import HPG_FetchPushv1
from configs.HTRPO_configs import HTRPO_FetchReachv1, HTRPO_FetchPushv1, HTRPO_EmptyMaze, HTRPO_FourRoomMaze
from configs.ReFER_configs import ReFER_FetchPushv1

torch.set_default_tensor_type(torch.FloatTensor)


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--alg', default='VPG',
                        help='algorithm to use: VPG | NPG | TRPO | PPO | HPG | HTRPO | ReFER')
    parser.add_argument('--env', default="Reacher-v1",
                        help='name of the environment to run')
    parser.add_argument('--reward', default='sparse',
                        help='reward type during training: dense | sparse, default: sparse. '
                             'NOTE: this argument will be used only when the environment supports sparse rewards. ')
    # parser.add_argument('--ou_noise', type=bool, default=True)
    # # TODO: SUPPORT PARAM NOISE
    # parser.add_argument('--param_noise', type=bool, default=False)
    # # TODO: SUPPORT NOISE END
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    # TODO: add '--num_steps' '--num_episodes' '--updates_per_step' '--snapshot_episode' '--render' into config file.
    parser.add_argument('--num_envs', type=int, default=1, metavar='N',
                        help='env numbers (default: 1)')
    parser.add_argument('--num_steps', type=int, default=1e6, metavar='N',
                        help='max episode length (default: 1e6)')
    parser.add_argument('--display', type=int, default=500, metavar='N',
                        help='episode interval for display (default: 5)')
    parser.add_argument('--eval_interval', type=int, default=0, metavar='N',
                        help='episode interval for evaluation (default: 0). 0 means no evaluation option is applied.')
    parser.add_argument('--num_evals', type=int, default=10, metavar='N',
                        help='evaluation episode number each time (default: 10)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether to resume training from a specific checkpoint')
    # TODO: Don't know what the following 3 parameters do
    parser.add_argument('--unnormobs', action='store_true', default=False,
                        help='whether to normalize inputs')
    parser.add_argument('--unnormret', action='store_true', default=False,
                        help='whether to normalize outputs')
    parser.add_argument('--unnormact', action='store_true', default=False,
                        help='whether to normalize outputs')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='resume from this checkpoint')
    parser.add_argument('--render', action='store_true', default=False,
                        help='whether to render GUI (default: False) during evaluation.')
    parser.add_argument('--cpu', help='whether use cpu to train', action='store_true', default = False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    configs = {}
    if args.alg in ("PG", "NPG", "TRPO", "PPO"):
        print("The chosen alg is on-policy. Stored transitions are not normalized.")
    # Do normalization for observations (and reward) for the following algorithms
    elif args.alg in ("TD3", "NAF", "DDPG", "DQN", "DDQN", "DuelingDQN", "HPG", "HTRPO", "ReFER"):
        print("The chosen alg is off-policy. Stored transitions are not normalized.")
        configs['norm_ob'] = not args.unnormobs
        configs['norm_rw'] = not args.unnormret
        args.unnormobs = True
        args.unnormret = True

    # build game environment
    env, env_type, env_id = build_env(args)
    env_obs_space = env.observation_space
    env_act_space = env.action_space
    # TODO: DEAL WITH DICT OBSERVATION SPACE, FOR EXAMPLE, IN TRPO OR DDPG.
    num_states = space_dim(env_obs_space)

    # 3 kinds of env_act_space
    if isinstance(env_act_space, spaces.Discrete):
        num_actions = env_act_space.n  # discrete action space, value based rl agent
        action_dims = 1
        DISCRETE_ACTION_SPACE = True
    elif isinstance(env_act_space, spaces.Box):
        num_actions = None
        action_dims = env_act_space.shape[0]
        DISCRETE_ACTION_SPACE = False
    elif isinstance(env_act_space, np.ndarray):
        num_actions = len(env_act_space)
        action_dims = 1
        DISCRETE_ACTION_SPACE = True
    else:
        assert 0, "Invalid Environment"

    # TODO: Don't know what this is for
    if env_type not in {"mujoco", "robotics", "robotsuite"}:
        print("The chosen env dose not support normalization. No normalization is applied.")
        configs['norm_ob'] = False
        configs['norm_rw'] = False

    # Define a logger
    logger = SummaryWriter(comment=args.alg + "-" + args.env)

    # create a directory to store the models
    output_dir = os.path.join("output", "models", args.alg)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initialize configurations
    if os.path.exists(os.path.join("configs", args.alg + "_configs", args.alg + "_" + "".join(args.env.split("-")) + '.py')):
        configs.update(eval(args.alg + "_" + "".join(args.env.split("-")) + "." + args.alg + "config"))
    configs['num_states'] = num_states
    configs['action_dims'] = action_dims
    configs['discrete_action'] = DISCRETE_ACTION_SPACE
    if num_actions:
        configs['num_actions'] = num_actions

    # for hindsight algorithms, init goal space of the environment.
    # if args.alg in {"HTRPO"}:
    #     configs['other_data'] = env.reset()
    #     assert isinstance(configs['other_data'], dict), \
    #         "Please check the environment settings, hindsight algorithms only support goal conditioned tasks."
    #     del configs['other_data']['observation']
    #     configs['goal_space'] = env_obs_space.spaces['desired_goal']
    #     configs['env'] = env

    # init agent
    if args.alg in ("VPG", "NPG", "TRPO", "PPO", "AdaptiveKLPPO", "HPG", "HTRPO", "ReFER"):
        configs['other_data'] = env.reset()
        assert isinstance(configs['other_data'], dict), \
            "Please check the environment settings, hindsight algorithms only support goal conditioned tasks."
        del configs['other_data']['observation']
        configs['goal_space'] = env_obs_space.spaces['desired_goal']
        configs['env'] = env
        if DISCRETE_ACTION_SPACE:
            RL_agent = eval("agents." + args.alg + "_Softmax(configs)")
        else:
            RL_agent = eval("agents." + args.alg + "_Gaussian(configs)")
    else:
        RL_agent = eval("agents." + args.alg + "(configs)")

    if not args.cpu:
        RL_agent.cuda()

    # resume networks
    if args.resume:
        RL_agent.load_model(load_path=output_dir, load_point=args.checkpoint)

    # training
    if args.alg == "VPG":
        trained_agent = run_vpg_train(env, RL_agent, args.num_steps, logger,
                                     eval_interval=args.eval_interval if args.eval_interval > 0 else None,
                                     num_evals=args.num_evals)
    elif args.alg == "NPG":
        trained_agent = run_npg_train(env, RL_agent, args.num_steps, logger,
                        eval_interval=args.eval_interval if args.eval_interval > 0 else None,
                        num_evals=args.num_evals)
    elif args.alg == "TRPO":
        trained_agent = run_trpo_train(env, RL_agent, args.num_steps, logger,
                        eval_interval=args.eval_interval if args.eval_interval > 0 else None,
                        num_evals=args.num_evals)
    elif args.alg == "HPG":
        trained_agent = run_hpg_train(env, RL_agent, args.num_steps, logger,
                        eval_interval=args.eval_interval if args.eval_interval > 0 else None,
                        num_evals=args.num_evals)
    elif args.alg == 'HTRPO':
        trained_agent = run_htrpo_train(env, RL_agent, args.num_steps, logger,
                        eval_interval=args.eval_interval if args.eval_interval > 0 else None,
                        num_evals=args.num_evals)
    elif args.alg == 'ReFER':
        trained_agent = run_refer_htrpo_train(env, RL_agent, args.num_steps, logger,
                        eval_interval=args.eval_interval if args.eval_interval > 0 else None,
                        num_evals=args.num_evals)
    # elif args.alg == "PPO" or args.alg == "AdaptiveKLPPO":
    #     trained_agent = run_ppo_train(env, RL_agent, args.num_steps, logger)
    # elif args.alg == "NAF":
    #     trained_agent = run_naf_train(env, RL_agent, args.num_steps, logger, args.display)
    # elif args.alg == "DDPG":
    #     trained_agent = run_ddpg_train(env, RL_agent, args.num_steps, logger, args.display)
    # elif args.alg == "TD3":
    #     trained_agent = run_td3_train(env, RL_agent, args.num_steps, logger, args.display)
    else:
        raise RuntimeError("Not an invalid algorithm.")

    logger.close()
