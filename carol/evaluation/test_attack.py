import hydra
import omegaconf

from mad_attack import MADAttack
import config

@hydra.main(config_path="../conf", config_name="main")
def main(cfg: omegaconf.DictConfig):
    attack_times = 5
    benchmark_name = cfg.evaluation.benchmark_name
    assert(benchmark_name == cfg.overrides.env)
    attacker_name = cfg.evaluation.attacker_name
    model_name = cfg.evaluation.model_name
    attack_steps_list = cfg.evaluation.attack_steps_list
    print(f"Attack target: {model_name}.")

    config_dict = config.test_config(config_type='attack')
    disturbance_list = config_dict[benchmark_name]['disturbance_list']
    model_path_dict = config_dict[benchmark_name]['model_path_dict']

    for attack_idx in range(attack_times):
        reward_dict = dict()
        for attack_steps in attack_steps_list:
            for disturbance in disturbance_list:
                if attacker_name == "MAD":
                    attacker = MADAttack(
                        disturbance=disturbance,
                        cfg=cfg,
                        attack_steps=attack_steps, 
                    )
                model_dict = model_path_dict[model_name]
                for model_idx, model_path in model_dict.items():
                    reward = attacker.reward_evaluate(model_path, model_name, expr_name=model_path)
                    if model_name not in reward_dict:
                        reward_dict[model_name] = dict()
                    if disturbance not in reward_dict[model_name]:
                        reward_dict[model_name][disturbance] = list()
                        reward_dict[model_name][disturbance].append(reward)
                    else:
                        reward_dict[model_name][disturbance].append(reward)
        print(reward_dict)

if __name__ == "__main__":
    main()