import hydra
import omegaconf

import config
from verifier import Verifier

@hydra.main(config_path="../conf", config_name="main")
def main(cfg: omegaconf.DictConfig):
    benchmark_name = cfg.evaluation.benchmark_name
    assert(benchmark_name == cfg.overrides.env)
    model_name = cfg.evaluation.model_name
    horizon_steps_list = cfg.evaluation.horizon_steps_list
    print(f"Evaluate Provability.")

    config_dict = config.test_config(config_type='provability')
    expr_list = config_dict[benchmark_name][model_name]

    if benchmark_name == 'hopper':
        model_error = 0.002
    
    for horizon_steps in horizon_steps_list:
        verifier = Verifier(
            cfg=cfg,
            horizon_steps=horizon_steps,
        )
        for expr, expr_dict in expr_list.items():
            policy_dir = expr_dict['policy_dir']
            model_dir = expr_dict['model_dir']
            worst_reward  = verifier.extract_symbolic_reward(policy_dir, model_dir, model_error=model_error)
            print(f"horizon step: {horizon_steps}, experiment name: {expr}, worst reward: {worst_reward}")

if __name__ == "__main__":
    main()