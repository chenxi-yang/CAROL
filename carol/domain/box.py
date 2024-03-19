"""
Definition of different domains
1. interval
2. disjunction of intervalss
3. octagon
4. zonotope
5. polyhedra
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def var(i, requires_grad=True):
    if torch.cuda.is_available():
        return Variable(torch.tensor(i, dtype=torch.float).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.tensor(i, dtype=torch.float), requires_grad=requires_grad)

EPSILON = var(1e-6)


class Box(list):
    def __init__(self, c, delta):
        self.c = c
        self.delta = delta
    
    def new(self, c, delta):
        res = self.__class__(c, delta)
        return res

    def widen(self, width):
        # increase the delta by width
        # assert(width >= 0)
        self.delta = self.delta + width
    
    def clone(self):
        return self.new(self.c.clone(), self.delta.clone())
    
    def check_in(self, other):
        # check: other in self (other.left >= self.left and other.right <= self.right)
        self_left = self.c - self.delta
        self_right = self.c + self.delta
        other_left = other.c - other.delta
        other_right = other.c + other.delta
        
        left_cmp = torch.ge(other_left, self_left)
        if False in left_cmp:
            return False
        right_cmp = torch.ge(self_right, other_right)
        if False in right_cmp:
            return False
        return True
    
    def select_from_index(self, dim, idx):
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx)
            if torch.cuda.is_available():
                idx = idx.cuda()
        return self.new(torch.index_select(self.c, dim, idx), torch.index_select(self.delta, dim, idx))

    def set_from_index(self, idx, other):
        self.c[idx] = other.c
        self.delta[idx] = other.delta
        return 
    
    def set_value(self, value):
        return self.new(value, var(0.0))
    
    def concatenate(self, other):
        return self.new(torch.cat((self.c, other.c), 1), torch.cat((self.delta, other.delta), 1))
    
    def sound_join(self, other):
        l1, r1 = self.c - self.delta, self.c + self.delta
        l2, r2 = other.c - other.delta, other.c + other.delta
        l = torch.min(l1, l2)
        r = torch.max(r1, r2)
        res = self.new((r + l) / 2, (r - l) / 2)
        return res
        
    def getRight(self):
        return self.c.add(self.delta)
    
    def getLeft(self):
        return self.c.sub(self.delta)
    
    def getInterval(self):
        res = Interval(self.c.sub(self.delta), self.c.add(self.delta))
        return res
    
    def matmul(self, other):
        if len(self.c.shape) == 3:
            self.c, self.delta = torch.squeeze(self.c, 1), torch.squeeze(self.delta, 1)
        return self.new(self.c.matmul(other), self.delta.matmul(other.abs()))
    
    def clamp(self, min, max):
        l, r = self.c - self.delta, self.c + self.delta
        if isinstance(l, torch.Tensor):
            l_clamped, r_clamped = torch.clamp(l, min, max), torch.clamp(r, min, max)
        elif isinstance(l, np.ndarray):
            l_clamped, r_clamped = np.clip(l, min, max), np.clip(r, min, max)
        return self.new((l_clamped + r_clamped) / 2, (r_clamped - l_clamped) / 2)
    
    def cut(self, eps):
        sat_idx = self.delta >= eps
        self.delta[sat_idx] = self.delta[sat_idx] - eps
        
    def abs(self):
        # self.c, self.delta
        
        l, r = self.c - self.delta, self.c + self.delta
        new_l = torch.zeros(l.shape)
        new_r = torch.zeros(l.shape)
        if torch.cuda.is_available():
            new_l = new_l.cuda()
            new_r = new_r.cuda()
        all_neg_idx = torch.logical_and(l < 0, r<=0)
        mid_idx = torch.logical_and(l<0, r>0)
        all_pos_idx = torch.logical_and(l>=0, r>0)

        new_l[all_neg_idx], new_r[all_neg_idx] = r[all_neg_idx].abs(), l[all_neg_idx].abs()
        new_l[mid_idx], new_r[mid_idx] = F.relu(l[mid_idx]), r[mid_idx]
        new_l[all_pos_idx], new_r[all_pos_idx] = l[all_pos_idx], r[all_pos_idx]

        return self.new((new_r + new_l) / 2, (new_r - new_l) / 2)
 

    def conv(self, weight, bias, padding):
        if len(self.c.shape) == 2:
            c, delta = self.c[:, None, :], self.delta[:, None, :]
        else:
            c, delta = self.c, self.delta
        new_c = F.conv1d(c, weight, bias=bias, padding=padding)
        new_delta = F.conv1d(c, weight.abs(), bias=bias, padding=padding)
        return self.new(new_c, new_delta)

    def add(self, other):
        if isinstance(other, float):
            other = torch.tensor(other)
        if isinstance(other, torch.Tensor):
            c, d = self.c.add(other), self.delta
            res = self.new(c, d)
        else:
            c, d = self.c.add(other.c), self.delta + other.delta
            res = self.new(c, d)
        return res
            
    def sub_l(self, other): # self - other
        if isinstance(other, torch.Tensor):
            return self.new(self.c.sub(other), self.delta)
        else:
            return self.new(self.c.sub(other.c), self.delta + other.delta)
    
    def sub_r(self, other): # other - self
        if isinstance(other, torch.Tensor):
            return self.new(other.sub(self.c), self.delta)
        else:
            return self.new(other.c.sub(self.c), self.delta + other.delta)
    
    def mul(self, other):
        interval = self.getInterval()
        if isinstance(other, torch.Tensor):
            pass
        elif isinstance(other, float):
            other = torch.tensor(other)
        else:
            other = other.getInterval()
        res_interval = interval.mul(other)
        res = res_interval.getBox()
        return res
        
    def exp(self):
        a = self.delta.exp()
        b = (-self.delta).exp()
        return self.new(self.c.exp().mul((a+b)/2), self.c.exp().mul((a-b)/2))
    
    def div(self, other): # other / self
        if isinstance(other, torch.Tensor):
            l, r = self.c - self.delta, self.c+self.delta
            updated_l, updated_r = other / r, other / l
            return self.new((updated_r + updated_l)/2, (updated_r - updated_l)/2)
        else:
            print(f"Currently Not Implemented.")
            exit(0)
    
    def sigmoid(self): # monotonic function
        tp = torch.sigmoid(self.c + self.delta)
        bt = torch.sigmoid(self.c - self.delta)
        # print(f"in sigmoid, tp: {tp}, bt: {bt}")
        return self.new((tp + bt)/2, (tp - bt)/2)
    
    def tanh(self): # monotonic function
        tp = torch.tanh(self.c + self.delta)
        bt = torch.tanh(self.c - self.delta)
        return self.new((tp + bt)/2, (tp - bt)/2)
    
    def relu(self): # monotonic function
        tp = F.relu(self.c + self.delta)
        bt = F.relu(self.c - self.delta)        
        return self.new((tp + bt)/2, (tp - bt)/2)
    
    def softplus(self):
        tp = F.softplus(self.c + self.delta)
        bt = F.softplus(self.c - self.delta)
        return self.new((tp + bt)/2, (tp - bt)/2)
    
    def sqrt(self):
        tp = torch.sqrt(self.c + self.delta)
        bt = torch.sqrt(self.c - self.delta)
        return self.new((tp + bt)/2, (tp - bt)/2)
    
    def torch_from_numpy(self):
        new_c = torch.from_numpy(self.c)
        new_delta = torch.from_numpy(self.delta)
        return self.new(new_c, new_delta)
    
    def float(self):
        return self.new(self.c.float(), self.delta.float())

    def to(self, device):
        new_c, new_delta = self.c.to(device), self.delta.to(device)
        return self.new(new_c, new_delta)
    
    @property
    def device(self):
        return self.c.device
    
    @property
    def shape(self):
        assert(self.c.shape==self.delta.shape)
        return self.c.shape
    
    def inf(self):
        return self.c - self.delta
    
    def sup(self):
        return self.c + self.delta
    
    def width(self):
        return self.delta * 2
    
    @property
    def ndim(self):
        return self.c.ndim
    
    def dim(self):
        assert(self.c.dim == self.delta.dim)
        return self.c.dim
    
    def detach(self):
        return self.new(self.c.detach(), self.delta.detach())
    
    def cpu(self):
        return self.new(self.c.cpu(), self.delta.cpu())
    
    def numpy(self):
        return self.new(self.c.numpy(), self.delta.numpy())
    
    def to_tensor(self):
        if isinstance(self.c, torch.Tensor):
            return self
        if isinstance(self.c, np.ndarray):
            return self.new(torch.from_numpy(self.c), torch.from_numpy(self.delta))
        raise ValueError("Input must be torch.Tensor or np.ndarray.")
    
    def astype(self, target_type):
        return self.new(self.c.astype(target_type), self.delta.astype(target_type))

    def to_float_tensor(self):
        return self.new(torch.FloatTensor(self.c), torch.FloatTensor(self.delta))
    
    def unsqueeze(self, dim):
        return self.new(self.c.unsqueeze(dim), self.delta.unsqueeze(dim))

    # Hook Methods
    def __setitem__(self, index, value):
        assert(isinstance(value, Box))
        self.c[index] = value.c
        self.delta[index] = value.delta
    
    def __getitem__(self, indices):
        return self.new(self.c[indices], self.delta[indices])
    
    def __len__(self):
        assert(len(self.c) == len(self.delta))
        return len(self.c)



class Interval:
    def __init__(self, left=var(0.0), right=var(0.0)):
        self.left = left
        self.right = right
    
    # for the same api
    def getInterval(self):
        res = Interval()
        res.left = self.left
        res.right = self.right
        return res
    
    def setInterval(self, l, r):
        res = Interval()
        res.left = l
        res.right = r
        return res
    
    def new(self, left, right):
        return self.__class__(left, right)
    
    def in_other(self, other):
        return torch.logical_and(self.left >= other.left, self.right <= other.right)
    
    def clone(self):
        return self.new(self.left.clone(), self.right.clone())

    def getBox(self):
        return Box(self.getCenter(), self.getDelta())
    
    def getLength(self):
        if self.right.data.item() < self.left.data.item():
            return var(0.0)
        else:
            return self.right.sub(self.left)
        
    def getVolumn(self):
        if self.right.data.item() < self.left.data.item():
            return var(0.0)
        else:
            return torch.max(EPSILON, (self.right.sub(self.left)))
    
    def split(self, partition):
        domain_list = list()
        unit = self.getVolumn().div(var(partition))
        for i in range(partition):
            new_domain = Interval()
            new_domain.left = self.left.add(var(i).mul(unit))
            new_domain.right = self.left.add(var(i + 1).mul(unit))
            domain_list.append(new_domain)
        return domain_list

    def getCenter(self):
        # C = var(2.0)
        return (self.left.add(self.right)).div(2.0)
    
    def getDelta(self):
        return (self.right.sub(self.left)).div(2.0)

    def equal(self, interval_2):
        if interval_2 is None:
            return False
        if interval_2.left.data.item() == self.left.data.item() and interval_2.right.data.item() == self.right.data.item():
            return True
        else:
            return False

    def isEmpty(self):
        if self.right.data.item() < self.left.data.item():
            return True
        else:
            return False
    
    def isPoint(self):
        if float(self.right) == float(self.left): # or abs(self.right.data.item() - self.left.data.item()) < EPSILON.data.item():
            return True
        else:
            return False

    def setValue(self, x):
        res = Interval()
        res.left = x
        res.right = x
        return res
    
    def soundJoin(self, other):
        res = self.new(torch.min(self.left, other.left), torch.max(self.right, other.right))
        return res
    
    def smoothJoin(self, other, alpha_prime_1, alpha_prime_2, alpha_1, alpha_2):
        c1, c2 = self.getCenter(), other.getCenter()
        delta1, delta2 = self.getDelta(), other.getDelta()
        c_out = (alpha_1 * c1 + alpha_2 * c2) / (alpha_1 + alpha_2)
        new_c1, new_c2 = alpha_prime_1 * c1 + (1 - alpha_prime_1) * c_out, alpha_prime_2 * c2 + (1 - alpha_prime_2) * c_out
        new_delta1, new_delta2 = alpha_prime_1 * delta1, alpha_prime_2 * delta2
        new_left = torch.min(new_c1 - new_delta1, new_c2 - new_delta2)
        new_right = torch.max(new_c1 + new_delta1, new_c1 + new_delta2)
        res = self.new(new_left, new_right)

        return res
    
    # arithmetic
    def add(self, y):

        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = self.left.add(y)
            res.right = self.right.add(y)
        else:
            res.left = self.left.add(y.left)
            res.right = self.right.add(y.right)

        return res

    def sub_l(self, y):

        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = self.left.sub(y)
            res.right = self.right.sub(y)
        else:
            res.left = self.left.sub(y.right)
            res.right = self.right.sub(y.left)
        return res

    def sub_r(self, y):
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = y.sub(var(1.0).mul(self.right))
            res.right = y.sub(var(1.0).mul(self.left))
        else:
            res.left = y.left.sub(self.right)
            res.right = y.right.sub(self.left)
        return res

    def mul(self, y):
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = torch.min(self.right.mul(y), self.left.mul(y))
            res.right = torch.max(self.right.mul(y), self.left.mul(y))
        else:
            res.left = torch.min(torch.min(y.left.mul(self.left), y.left.mul(self.right)), torch.min(y.right.mul(self.left), y.right.mul(self.right)))
            res.right = torch.max(torch.max(y.left.mul(self.left), y.left.mul(self.right)), torch.max(y.right.mul(self.left), y.right.mul(self.right)))
        return res

    def div(self, y):
        res = Interval()
        tmp_interval = Interval()
        tmp_interval.left = var(1.0).div(self.right)
        tmp_interval.right = var(1.0).div(self.left)
        res = tmp_interval.mul(y)
        return res

    def exp(self):
        res = Interval()
        res.left = torch.exp(self.left)
        res.right = torch.exp(self.right)
        return res

    # def cos(self):
    #     cache = Interval(self.left, self.right)

    #     cache = handleNegative(cache)
        
    #     t = cache.fmod(PI_TWICE)
    #     del cache
    #     torch.cuda.empty_cache()
    #     if float(t.getVolumn()) >= float(PI_TWICE):
    #         res = Interval(var_list([-1.0]), var_list([1.0]))
    #     elif float(t.left) >= float(PI):
    #         cosv = (t.sub_l(PI)).cos()
    #         res = cosv.mul(var_list([-1.0]))
    #     else:
    #         tl = torch.cos(t.right)
    #         tr = torch.cos(t.left)
    #         if float(t.right) <= float(PI.data.item()):
    #             res = Interval(tl, tr)
    #         elif float(t.right) <= float(PI_TWICE):
    #             res = Interval(var_list([-1.0]), torch.max(tl, tr))
    #         else:
    #             res = Interval(var_list([-1.0]), var_list([1.0]))
    #     del t
    #     torch.cuda.empty_cache()

    #     return res

    # def sin(self):
    #     return self.sub_l(PI_HALF).cos()

    def max(self, y):
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = torch.max(self.left, y)
            res.right = torch.max(self.right, y)
        else:
            res.left = torch.max(self.left, y.left)
            res.right = torch.max(self.right, y.right)
        return res
    
    def min(self, y):
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = torch.min(self.left, y)
            res.right = torch.min(self.right, y)
        else:
            res.left = torch.min(self.left, y.left)
            res.right = torch.min(self.right, y.right)
        return res
    
    def sqrt(self):
        res = Interval()
        res.left = torch.sqrt(self.left)
        res.right = torch.sqrt(self.right)
        return res
    
    def fmod(self, y):
        if isinstance(y, torch.Tensor):
            y_interval = Interval()
            y_interval = y_interval.setValue(y)
        else:
            y_interval = y
        
        if self.left.data.item() < 0.0:
            yb = y_interval.left
        else:
            yb = y_interval.right
        n = self.left.div(yb)
        if(n.data.item() <= 0.0): 
            n = torch.ceil(n)
        else:
            n = torch.floor(n)
        tmp_1 = y_interval.mul(n)

        res = self.sub_l(tmp_1)
        
        return res