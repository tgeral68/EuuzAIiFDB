import tqdm
import torch
from torch import nn

from function_tools import poincare_function, poincare_module
from embedding_tools import losses
from optim_tools import optimizer



class RiemannianEmbedding(nn.Module):
    def __init__(self, n_exemple, size=10,  cuda=False, lr=1e-2, verbose=True, negative_distribution=None,
                optimizer_method=optimizer.PoincareBallSGDAdd):
        super(RiemannianEmbedding, self).__init__()
        self.cuda = cuda
        self.N = n_exemple
        self.W = poincare_module.PoincareEmbedding(n_exemple, size)
        if(self.cuda):
            self.W.cuda()
        self.optimizer = optimizer_method(self.W.parameters(), lr=lr)
        self.verbose = verbose
        self.d = poincare_function.distance
        if(negative_distribution is None):
            self.n_dist = torch.distributions.Categorical(torch.ones(self.N)/self.N)
        else:
            self.n_dist = negative_distribution

    def forward(self, x):
        return self.W(x)

    def get_PoincareEmbeddings(self):
        return self.W.l_embed(torch.arange(0, self.N, device=self.W.l_embed.weight.data.device)).detach()

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def fit(self, dataloader, alpha=1.0, beta=1.0, max_iter=100,
            negative_sampling=5):
        
        progress_bar = tqdm.trange(max_iter) if(self.verbose) else range(max_iter)
        for i in progress_bar:
            loss_value1, loss_value2, = 0,0
            for example, neigbhors, walks in dataloader:
                self.optimizer.zero_grad()
                # obtain negatives examples sampled according to the given distribution
                with torch.no_grad():
                    negative = self.n_dist.sample(sample_shape=(walks.size(0), walks.size(1), negative_sampling))
                # set variables to cuda device
                if(self.cuda):
                    example = example.cuda()
                    neigbhors = neigbhors.cuda()
                    walks = walks.cuda()
                    negative = negative.cuda()
                # get the needed embeddings
                r_example = example.unsqueeze(1).expand_as(neigbhors)
                embed_source, embed_neigbhor = self.W(r_example), self.W(neigbhors)
                embed_source_rw  = self.W(walks)
                embed_source_rw, embed_context_rw = embed_source_rw[:,:,0], embed_source_rw[:,:,1]
                embed_negative = self.W(negative)
                # computing O1 loss
                loss_o1 = losses.SGDLoss.O1(embed_source, embed_neigbhor)
                # computing O2 loss
                loss_o2 = losses.SGDLoss.O2(embed_source_rw, embed_context_rw, embed_negative)
                # computing total loss
                loss = alpha * loss_o1.mean() + beta * loss_o2.mean() 
                # if we want to use the prior loss
                loss_value1 = loss_o1.mean().item()
                loss_value2 = loss_o2.mean().item()
                loss.backward()
                self.optimizer.step()
            if(self.verbose):
                progress_bar.set_postfix({"O1":loss_value1, "O2":loss_value2})