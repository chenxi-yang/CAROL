import torch

from carol.util.common import var

# interval domain
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