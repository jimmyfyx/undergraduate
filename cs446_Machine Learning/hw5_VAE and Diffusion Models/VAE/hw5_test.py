import torch
import numpy as np


# z = torch.tensor([[1, 1], [3, 2]])
x = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
# mu = torch.tensor([[1, 2], [2, 3]])
p = torch.tensor([[0.5, 0.6, 0.5], [0.4, 0.5, 0.6]])
sample = torch.normal(mean=x[0], std=p[0])


def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]

        # TODO: implement the log likelihood of a bernoulli distribution p(x)
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         if x[i][j].item() == 1:
        #             x[i][j] = p[i][j]
                    
        #         else:
        #             x[i][j] = 1 - p[i][j]
                
        #         print(x)
        
    
        prob = (1 - x) * torch.log(1 - p) +  x * torch.log(p)
        print(prob)
        prob = torch.sum(prob, dim=1)
        print(prob)
        1/0

        
        return torch.log(prob)

logpdf_bernoulli(x, p)