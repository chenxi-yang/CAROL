import math



class BaseScheduler():
    def __init__(self, 
        target_eps, 
        start_steps, 
        end_steps):
        self.target_eps = target_eps
        self.start_steps = start_steps
        self.end_steps = end_steps
    
    def step_eps(self, env_steps, cur_eps):
        NotImplementedError(f"Not Implemented.")


class ConstantScheduler(BaseScheduler):
    def __init__(self, 
        target_eps, 
        start_steps, 
        end_steps):
        super(ConstantScheduler, self).__init__(target_eps, start_steps, end_steps)

        self.target_eps = target_eps
        self.start_steps = start_steps
        self.end_steps = end_steps
    
    def step_eps(self, env_steps, cur_steps):
        return cur_steps 


class LinearScheduler(BaseScheduler):
    def __init__(self, 
        target_eps, 
        start_steps, 
        end_steps):
        super(LinearScheduler, self).__init__(target_eps, start_steps, end_steps)

        self.target_eps = target_eps
        self.start_steps = start_steps
        self.end_steps = end_steps
    
    def step_eps(self, env_steps, cur_eps):
        if env_steps <= self.start_steps:
            eps = cur_eps # not reach the eschedule step
        elif env_steps >= self.end_steps:
            eps = self.target_eps # the scheduler process is limited to the 
        else:
            target_eps = self.target_eps * (env_steps - self.start_steps)/(self.end_steps - self.start_steps)
            eps = min(target_eps, self.target_eps)
        return eps


class ExpScheduler(BaseScheduler):
    def __init__(self, 
        target_eps, 
        start_steps, 
        end_steps,
        exp_scale=10,):
        super(ExpScheduler, self).__init__(target_eps, start_steps, end_steps)

        self.target_eps = target_eps
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.exp_scale = exp_scale
    
    def step_eps(self, env_steps, cur_eps):
        if env_steps <= self.start_steps:
            eps = cur_eps # not reach the eschedule step
        elif env_steps >= self.end_steps:
            eps = self.target_eps # the scheduler process is limited to the 
        else:
            # update the ratio
            ratio = (env_steps - self.end_steps)/(self.end_steps) * self.exp_scale
            # ratio = (env_steps - self.end_steps)/(self.start_steps) * self.exp_scale # 20
            multiple_ratio = math.e**ratio
            target_eps = self.target_eps * multiple_ratio
            eps = min(target_eps, self.target_eps)
                    
        return eps


class SmoothScheduler(BaseScheduler):
    '''
    Exp scheduler + Linear Scheduler.
    Starting from start_steps, using the exp to increase the eps first.
    When the derivative of exp > linear: use linear scheduler.
    '''
    def __init__(self, 
        target_eps, 
        start_steps, 
        end_steps,
        exp_scale=10,
        ):
        super(SmoothScheduler, self).__init__(target_eps, start_steps, end_steps)

        self.target_eps = target_eps
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.exp_scale = exp_scale
        self.turning = None
        self.A, self.B = None, None

    def step_eps(self, env_steps, cur_eps):
        # print(f"step: {env_steps}; eps: {cur_eps}")
        if env_steps <= self.start_steps:
            eps = cur_eps # not reach the eschedule step
        elif env_steps >= self.end_steps + self.start_steps:
            eps = self.target_eps # the scheduler process is limited to the 
        else:
            # print(f"env_steps: {env_steps}; cur_eps: {cur_eps}; turning: {self.turning}")
            # print(f"A: {self.A}; B: {self.B}")
            if not self.turning: # exp part
                # ratio = (env_steps - self.end_steps)/(self.start_steps) * self.exp_scale
                ratio = (env_steps - self.end_steps)/(self.end_steps) * self.exp_scale
                multiple_ratio = math.e**ratio
                # get the eps from exp scheduling
                # print(f"ratio: {ratio}, multiple_ratio: {multiple_ratio}")
                exp_eps = self.target_eps * multiple_ratio
                # calculate the derivative of exp: eps = target_eps * e**(exp_scale * (env_steps-end)/start)
                d_exp = exp_eps * self.exp_scale / self.start_steps
                # d_linear = (self.target_eps - cur_eps) / (self.end_steps - env_steps)
                
                # In the linear in smooth, turn the real end_steps to end_steps + start_steps
                # d_linear = (self.target_eps - cur_eps) / (self.end_steps + self.start_steps - env_steps)
                d_linear = (self.target_eps - cur_eps) / (self.end_steps + self.start_steps - env_steps)
                # print(f"d_exp: {d_exp}; d_linear: {d_linear}")
                if d_exp < d_linear: # when exp is not steep
                    eps = exp_eps
                else: # use linear scheduling
                    # linear scheduler: eps = A * step + B
                    self.A = d_linear
                    self.B = cur_eps - self.A * env_steps
                    eps = self.A * env_steps + self.B
                    self.turning = True
            else:
                eps = self.A * env_steps + self.B
            eps = min(eps, self.target_eps)
                    
        return eps


class RefSmoothedScheduler(BaseScheduler):
    def __init__(self, 
        target_eps, 
        start_steps, 
        end_steps,
        exp_scale=10,
        ):
        super(RefSmoothedScheduler, self).__init__(target_eps, start_steps, end_steps)

        self.target_eps = target_eps
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.exp_scale = exp_scale
        self.beta = 4.0
        self.mid_point = 0.25
            
    # Smooth schedule that slowly morphs into a linear schedule.
    # Code is based on DeepMind's IBP implementation:
    # https://github.com/deepmind/interval-bound-propagation/blob/2c1a56cb0497d6f34514044877a8507c22c1bd85/interval_bound_propagation/src/utils.py#L84
    def step_eps(self, env_steps, cur_eps=None):
        init_value = 0.0
        final_value = self.target_eps
        beta = self.beta
        step = env_steps
        # Batch number for schedule start
        init_step = self.start_steps
        # Batch number for schedule end
        final_step = self.end_steps
        # Batch number for switching from exponential to linear schedule
        mid_step = int((final_step - init_step) * self.mid_point) + init_step
        t = (mid_step - init_step) ** (beta - 1.)
        # find coefficient for exponential growth, such that at mid point the gradient is the same as a linear ramp to final value
        alpha = (final_value - init_value) / ((final_step - mid_step) * beta * t + (mid_step - init_step) * t)
        # value at switching point
        mid_value = init_value + alpha * (mid_step - init_step) ** beta
        # return init_value when we have not started
        is_ramp = float(step > init_step)
        # linear schedule after mid step
        is_linear = float(step >= mid_step)
        exp_value = init_value + alpha * float(step - init_step) ** beta
        linear_value = min(mid_value + (final_value - mid_value) * (step - mid_step) / (final_step - mid_step), final_value)
        eps = is_ramp * ((1.0 - is_linear) * exp_value + is_linear * linear_value) + (1.0 - is_ramp) * init_value
        return eps




