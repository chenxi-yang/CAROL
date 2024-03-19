import hydra
import numpy as np
import omegaconf
import torch

import carol.util.env
from carol.algorithms.core import CAROLalgorithm

@hydra.main(config_path="conf", config_name="main")
def main(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = carol.util.env.EnvHandler.make_env(cfg)
    extra_env, _, _ = carol.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    test_env, *_ = carol.util.env.EnvHandler.make_env(cfg)
    best_response_algorithm = CAROLalgorithm(
        env=env,
        test_env=test_env,
        extra_env=extra_env,
        termination_fn=term_fn,
        cfg=cfg,
    )
    best_response_algorithm.train()


if __name__ == "__main__":
    main()