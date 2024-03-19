import argparse
import json
import torch
import torch.nn as nn
import math
import torch as ch
import random
import numpy as np
import os

def determinant(mat):
    '''
    Returns the determinant of a diagonal matrix
    Inputs:
    - mat, a diagonal matrix
    Returns:
    - The determinant of mat, aka product of the diagonal
    '''
    return ch.exp(ch.log(mat).sum())

def shape_equal_cmp(*args):
    '''
    Checks that the shapes of the passed arguments are equal
    Inputs:
    - All arguments should be tensors
    Returns:
    - True if all arguments have the same shape, else ValueError
    '''
    for i in range(len(args)-1):
        if args[i].shape != args[i+1].shape:
            s = "\n".join([str(x.shape) for x in args])
            raise ValueError("Expected equal shapes. Got:\n%s" % s)
    return True

def shape_equal(a, *args):
    '''
    Checks that a group of tensors has a required shape
    Inputs:
    - a, required shape for all the tensors
    - Rest of the arguments are tensors
    Returns:
    - True if all tensors are of shape a, otherwise ValueError
    '''
    for arg in args:
        if list(arg.shape) != list(a):
            if len(arg.shape) != len(a):
                raise ValueError("Expected shape: %s, Got shape %s" \
                                    % (str(a), str(arg.shape)))
            for i in range(len(arg.shape)):
                if a[i] == -1 or a[i] == arg.shape[i]:
                    continue
                raise ValueError("Expected shape: %s, Got shape %s" \
                                    % (str(a), str(arg.shape)))
    return shape_equal_cmp(*args)

class CtsPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    '''
    def __init__(self, state_dim, action_dim, init=None, hidden_sizes=(64, 64),
                 time_in_state=False, share_weights=False, activation=None, use_merged_bias=False):
        super().__init__()
        # if isinstance(activation, str):
        #     self.activation = activation_with_name(activation)()
        # else:
        #     # Default to tanh.
        #     self.activation = ACTIVATION()
        self.activation = nn.Tanh()
        print('Using activation function', self.activation)
        self.action_dim = action_dim
        self.discrete = False
        self.time_in_state = time_in_state
        self.use_merged_bias = use_merged_bias

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            if use_merged_bias:
                # Use an extra dimension for weight perturbation, simulating bias.
                lin = nn.Linear(prev_size + 1, i, bias=False)
            else:
                lin = nn.Linear(prev_size, i, bias=True)
            # initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        if use_merged_bias:
            self.final_mean = nn.Linear(prev_size + 1, action_dim, bias=False)
        else:
            self.final_mean = nn.Linear(prev_size, action_dim, bias=True)
        # initialize_weights(self.final_mean, init, scale=0.01)
        
        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            assert not use_merged_bias
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            # initialize_weights(self.final_value, init, scale=1.0)

        stdev_init = ch.zeros(action_dim)
        self.log_stdev = ch.nn.Parameter(stdev_init)

    def forward(self, x):
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            if self.use_merged_bias:
                # Generate an extra "one" for each element, which acts as a bias.
                bias_padding = ch.ones(x.size(0),1)
                x = ch.cat((x, bias_padding), dim=1)
            else:
                pass
            x = self.activation(affine(x))
        
        if self.use_merged_bias:
            bias_padding = ch.ones(x.size(0),1)
            x = ch.cat((x, bias_padding), dim=1)
        means = self.final_mean(x)
        std = ch.exp(self.log_stdev)

        return means, std 

    def get_value(self, x):
        assert self.share_weights, "Must be sharing weights to use get_value"

        # If the time is in the state, discard it
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)

    def sample(self, p):
        '''
        Given prob dist (mean, var), return: actions sampled from p_i, and their
        probabilities. p is tuple (means, var). means shape 
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        means, std = p
        return (means + ch.randn_like(means)*std).detach()

    def get_loglikelihood(self, p, actions):
        try:    
            mean, std = p
            nll =  0.5 * ((actions - mean) / std).pow(2).sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                   + self.log_stdev.sum(-1)
            return -nll
        except Exception as e:
            raise ValueError("Numerical error")

    def calc_kl(self, p, q):
        '''
        Get the expected KL distance between two sets of gaussians over states -
        gaussians p and q where p and q are each tuples (mean, var)
        - In other words calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))]
        - From https://stats.stackexchange.com/a/60699
        '''
        p_mean, p_std = p
        q_mean, q_std = q
        p_var, q_var = p_std.pow(2), q_std.pow(2)
        assert shape_equal([-1, self.action_dim], p_mean, q_mean)
        assert shape_equal([self.action_dim], p_var, q_var)

        d = q_mean.shape[1]
        diff = q_mean - p_mean

        log_quot_frac = ch.log(q_var).sum() - ch.log(p_var).sum()
        tr = (p_var / q_var).sum()
        quadratic = ((diff / q_var) * diff).sum(dim=1)

        kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
        assert kl_sum.shape == (p_mean.shape[0],)
        return kl_sum

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (mean, var), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        _, std = p
        detp = determinant(std)
        d = std.shape[0]
        entropies = ch.log(detp) + .5 * (d * (1. + math.log(2 * math.pi)))
        return entropies
    
    def reset(self):
        return

    def pause_history(self):
        return

    def continue_history(self):
        return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_common_parser_opts(parser):
    # Basic setup
    parser.add_argument('--game', type=str, help='gym game')
    parser.add_argument('--mode', type=str, choices=['ppo', 'trpo', 'robust_ppo', 'adv_ppo', 'adv_trpo', 'adv_sa_ppo'],
                        help='pg alg')
    parser.add_argument('--out-dir', type=str,
                        help='out dir for store + logging')
    parser.add_argument('--advanced-logging', type=str2bool, const=True, nargs='?')
    parser.add_argument('--kl-approximation-iters', type=int,
                        help='how often to do kl approx exps')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--policy-net-type', type=str)
    parser.add_argument('--value-net-type', type=str)
    parser.add_argument('--train-steps', type=int,
                        help='num agent training steps')
    parser.add_argument('--cpu', type=str2bool, const=True, nargs='?')

    # Which value loss to use
    parser.add_argument('--value-calc', type=str,
                        help='which value calculation to use')
    parser.add_argument('--initialization', type=str)

    # General Policy Gradient parameters
    parser.add_argument('--num-actors', type=int, help='num actors (serial)',
                        choices=[1])
    parser.add_argument('--t', type=int,
                        help='num timesteps to run each actor for')
    parser.add_argument('--gamma', type=float, help='discount on reward')
    parser.add_argument('--lambda', type=float, help='GAE hyperparameter')
    parser.add_argument('--val-lr', type=float, help='value fn learning rate')
    parser.add_argument('--val-epochs', type=int, help='value fn epochs')
    parser.add_argument('--initial-std', type=float, help='initial value of std for Gaussian policy. Default is 1.')

    # PPO parameters
    parser.add_argument('--adam-eps', type=float, choices=[0, 1e-5], help='adam eps parameter')

    parser.add_argument('--num-minibatches', type=int,
                        help='num minibatches in ppo per epoch')
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--ppo-lr', type=float,
                        help='if nonzero, use gradient descent w this lr')
    parser.add_argument('--ppo-lr-adam', type=float,
                        help='if nonzero, use adam with this lr')
    parser.add_argument('--anneal-lr', type=str2bool,
                        help='if we should anneal lr linearly from start to finish')
    parser.add_argument('--clip-eps', type=float, help='ppo clipping')
    parser.add_argument('--clip-val-eps', type=float, help='ppo clipping value')
    parser.add_argument('--entropy-coeff', type=float,
                        help='entropy weight hyperparam')
    parser.add_argument('--value-clipping', type=str2bool,
                        help='should clip values (w/ ppo eps)')
    parser.add_argument('--value-multiplier', type=float,
                        help='coeff for value loss in combined step ppo loss')
    parser.add_argument('--share-weights', type=str2bool,
                        help='share weights in valnet and polnet')
    parser.add_argument('--clip-grad-norm', type=float,
                        help='gradient norm clipping (-1 for no clipping)')
    parser.add_argument('--policy-activation', type=str,
                        help='activation function for countinous policy network')
    
    # TRPO parameters
    parser.add_argument('--max-kl', type=float, help='trpo max kl hparam')
    parser.add_argument('--max-kl-final', type=float, help='trpo max kl final')
    parser.add_argument('--fisher-frac-samples', type=float,
                        help='frac samples to use in fisher vp estimate')
    parser.add_argument('--cg-steps', type=int,
                        help='num cg steps in fisher vp estimate')
    parser.add_argument('--damping', type=float, help='damping to use in cg')
    parser.add_argument('--max-backtrack', type=int, help='max bt steps in fvp')
    parser.add_argument('--trpo-kl-reduce-func', type=str, help='reduce function for KL divergence used in line search. mean or max.')

    # Robust PPO parameters.
    parser.add_argument('--robust-ppo-eps', type=float, help='max eps for robust PPO training')
    parser.add_argument('--robust-ppo-method', type=str, choices=['convex-relax', 'sgld', 'pgd'], help='robustness regularization methods')
    parser.add_argument('--robust-ppo-pgd-steps', type=int, help='number of PGD optimization steps')
    parser.add_argument('--robust-ppo-detach-stdev', type=str2bool, help='detach gradient of standard deviation term')
    parser.add_argument('--robust-ppo-reg', type=float, help='robust PPO regularization')
    parser.add_argument('--robust-scheduler',  type=str, help='scheduler type')
    parser.add_argument('--robust-ppo-eps-scheduler-opts', type=str, help='options for epsilon scheduler for robust PPO training')
    parser.add_argument('--robust-ppo-beta', type=float, help='max beta (IBP mixing factor) for robust PPO training')
    parser.add_argument('--robust-ppo-beta-scheduler-opts', type=str, help='options for beta scheduler for robust PPO training')
    parser.add_argument('--compound_step', type=int, default=10, help='number of compound attack steps')

    # Adversarial PPO parameters.
    parser.add_argument('--adv-ppo-lr-adam', type=float,
                        help='if nonzero, use adam for adversary policy with this lr')
    parser.add_argument('--adv-entropy-coeff', type=float,
                        help='entropy weight hyperparam for adversary policy')
    parser.add_argument('--adv-eps', type=float, help='adversary perturbation eps')
    parser.add_argument('--adv-clip-eps', type=float, help='ppo clipping for adversary policy')
    parser.add_argument('--adv-val-lr', type=float, help='value fn learning rate for adversary policy')
    parser.add_argument('--adv-policy-steps', type=float, help='number of policy steps before adversary steps')
    parser.add_argument('--adv-adversary-steps', type=float, help='number of adversary steps before adversary steps')
    parser.add_argument('--adv-adversary-ratio', type=float, help='percentage of frames to attack for the adversary')

    # Adversarial attack parameters.
    parser.add_argument('--attack-method', type=str, choices=["none", "critic", "random", "action", "sarsa", "sarsa+action", "advpolicy", "action+imit","compound"], help='adversarial attack methods.')

    parser.add_argument('--attack-ratio', type=float, help='attack only a ratio of steps.')
    parser.add_argument('--attack-steps', type=int, help='number of PGD optimization steps.')
    parser.add_argument('--attack-eps', type=str, help='epsilon for attack. If set to "same", we will use value of robust-ppo-eps.')
    parser.add_argument('--attack-step-eps', type=str, help='step size for each iteration. If set to "auto", we will use attack-eps / attack-steps')
    parser.add_argument('--attack-sarsa-network', type=str, help='sarsa network to load for attack.')
    parser.add_argument('--attack-sarsa-action-ratio', type=float, help='When set to non-zero, enable sarsa-action attack.')
    parser.add_argument('--attack-advpolicy-network', type=str, help='adversarial policy network to load for attack.')
    parser.add_argument('--collect-perturbed-states', type=str2bool, help='collect perturbed states during training')

    # Normalization parameters
    parser.add_argument('--norm-rewards', type=str, help='type of rewards normalization', 
                        choices=['rewards', 'returns', 'none'])
    parser.add_argument('--norm-states', type=str2bool, help='should norm states')
    parser.add_argument('--clip-rewards', type=float, help='clip rews eps')
    parser.add_argument('--clip-observations', type=float, help='clips obs eps')

    # Sequence training parameters
    parser.add_argument('--history-length', type=int, help='length of history to use for LSTM. If <= 1, we do not use LSTM.')
    parser.add_argument('--use-lstm-val', type=str2bool, help='use a lstm for value function')

    # Saving
    parser.add_argument('--save-iters', type=int, help='how often to save model (0 = no saving)')
    parser.add_argument('--force-stop-step', type=int, help='forcibly terminate after a given number of steps. Useful for debugging and tuning.')

    # Visualization
    parser.add_argument('--show-env', type=str2bool, help='Show environment visualization')
    parser.add_argument('--save-frames', type=str2bool, help='Save environment frames')
    parser.add_argument('--save-frames-path', type=str, help='Path to save environment frames')

    # For grid searches only
    # parser.add_argument('--cox-experiment-path', type=str, default='')
    return parser

def override_json_params(params, json_params, excluding_params):
    # Override the JSON config with the argparse config
    missing_keys = []
    for key in json_params:
        if key not in params:
            missing_keys.append(key)
    # assert not missing_keys, "Following keys not in args: " + str(missing_keys)

    missing_keys = []
    for key in params:
        if key not in json_params and key not in excluding_params:
            missing_keys.append(key)
    # assert not missing_keys, "Following keys not in JSON: " + str(missing_keys)

    json_params.update({k: params[k] for k in params if params[k] is not None})
    return json_params

def return_params_radial(config_path):
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str, default=config_path, required=True,
                        help='json for this config')
    parser.add_argument('--out-dir-prefix', type=str, default="", required=False,
                        help='prefix for output log path')
    parser.add_argument('--load-model', type=str, default=None, required=False, help='load pretrained model and optimizer states before training')
    parser.add_argument('--no-load-adv-policy', action='store_true', required=False, help='Do not load adversary policy and value network from pretrained model.')
    parser.add_argument('--adv-policy-only', action='store_true', required=False, help='Run adversary only, by setting main agent learning rate to 0')
    parser.add_argument('--deterministic', action='store_true', help='disable Gaussian noise in action for --adv-policy-only mode')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser = add_common_parser_opts(parser)
    
    args = parser.parse_args()

    params = vars(args)
    seed = params['seed']
    json_params = json.load(open(args.config_path))

    extra_params = ['config_path', 'out_dir_prefix', 'load_model', 'no_load_adv_policy', 'adv_policy_only', 'deterministic', 'seed','compound_step']
    params = override_json_params(params, json_params, extra_params)

    if params['adv_policy_only']:
        if params['adv_ppo_lr_adam'] == 'same':
            params['adv_ppo_lr_adam'] = params['ppo_lr_adam']
            print(f"automatically setting adv_ppo_lr_adam to {params['adv_ppo_lr_adam']}")
        print('disabling policy training (train adversary only)')
        params['ppo_lr_adam'] = 0.0 * params['ppo_lr_adam']
    else:
        # deterministic mode only valid when --adv-policy-only is set
        assert not params['deterministic']

    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.set_printoptions(threshold=5000, linewidth=120)

    # Append a prefix for output path.
    if args.out_dir_prefix:
        params['out_dir'] = os.path.join(args.out_dir_prefix, params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")
    
    return params