PATH_CAROL_hopper = f"../../../../../example_model/hopper"

def test_config(config_type):
    attack_config_dict = {
        'hopper': { 
            'disturbance_list': [0.0, 0.075],
            'model_path_dict': {
                'CAROL': {
                    'policy': f"{PATH_CAROL_hopper}/overall",
                },
            },
        },
    }

    provability_config_dict = {
        'hopper': {
            'CAROL': {
                'Overall':{
                    'policy_dir': f"{PATH_CAROL_hopper}/overall",
                    'model_dir': f"{PATH_CAROL_hopper}/overall",
                },
                'Separate 1':{
                    'policy_dir': f"{PATH_CAROL_hopper}/overall",
                    'model_dir': f"{PATH_CAROL_hopper}/separate_1",
                },
                'Separate 2':{
                    'policy_dir': f"{PATH_CAROL_hopper}/overall",
                    'model_dir': f"{PATH_CAROL_hopper}/separate_1",
                },
                'Separate 3':{
                    'policy_dir': f"{PATH_CAROL_hopper}/overall",
                    'model_dir': f"{PATH_CAROL_hopper}/separate_1",
                },
            },
        },
    }


    if config_type == "attack":
        config_dict = attack_config_dict
    elif config_type == "provability":
        config_dict = provability_config_dict
    else:
        NotImplementedError(f"Please define the config_type.")
    return config_dict
