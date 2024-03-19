from carol.domain.box import Box
import itertools
import random
import torch

def partition_and_sample(s, partition_per_dim, sample_size, all_steps):
    # get the lower bound of s
    base_c = s.inf()
    # get the parititoned delta
    base_delta = s.delta / partition_per_dim

    if len(s.shape) == 2:
        B = s.shape[0]
    else:
        B = 1
    if sample_size == len(all_steps):
        sampled_steps = all_steps
    else:
        sampled_steps = random.sample(all_steps, sample_size)
    data_c = None
    data_delta = None

    # Option 1: repeat steps and delta first, do multiplication once
    base_c_repeated = base_c.repeat(sample_size, 1)
    base_delta_repeated = base_delta.repeat(sample_size, 1)
    # print(f"{base_delta.shape}, {base_delta_repeated.shape}")
    # step_start = time.time()
    steps = torch.IntTensor(sampled_steps).to(base_delta.device)
    steps_repeated = torch.repeat_interleave(steps, repeats=B, dim=0)
    # print(f"{steps.shape}, {steps_repeated.shape}")
    
    # print(f"repeat all steps: {time.time() - step_start}")
    bias_repeated = torch.mul(steps_repeated, base_delta_repeated) * 2
    new_data_inf_repeated = base_c_repeated + bias_repeated
    new_data_c_repeated = new_data_inf_repeated + base_delta_repeated
    # print(f"time: {time.time() - start_time}")
    return Box(new_data_c_repeated, base_delta_repeated)
        
    # # Option 2: do the multiplication for every step
    # for step in sampled_steps:
    #     step = torch.IntTensor(step).to(base_delta.device)
    #     bias = torch.mul(step, base_delta) * 2
    #     new_data_inf = base_c + bias
    #     new_data_c = new_data_inf + base_delta
    #     if data_c is None:
    #         data_c = new_data_c
    #         data_delta = base_delta
    #     else:
    #         data_c = torch.cat((data_c, new_data_c), 0)
    #         data_delta = torch.cat((data_delta, base_delta), 0)
    #     i += 1
    # print(f"step: {i}, time: {time.time() - start_time}")
    # return Box(data_c, data_delta)

if __name__ == "__main__":
    c = torch.Tensor(
        [[1, 0, 2, 3],
        [5, 4, 3, 1]]
    )
    delta = torch.Tensor(
        [[2,2,2,2],
        [2,1,1,2]]
    )
    s = Box(c, delta)

    res_s = partition_and_sample(s, 2, 1)
    print(res_s.c)
    print(res_s.delta)



        

