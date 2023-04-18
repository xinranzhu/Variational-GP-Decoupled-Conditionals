
import sys
import time
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader

class SVGP(ApproximateGP):
    def __init__(self, inducing_points, kernel_type='se', ard_num_dims=None):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution
        )
        super(SVGP, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if kernel_type == 'se':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern1/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern3/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern5/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=ard_num_dims)
         
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
