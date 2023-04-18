import sys
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from .mlls.variational_elbo import VariationalELBO as DCSVGP_ELBO
from .mlls.predictive_log_likelihood import PredictiveLogLikelihood as DCSVGP_PLL
from gpytorch.mlls.variational_elbo import VariationalELBO as SVGP_ELBO
from gpytorch.mlls.predictive_log_likelihood import PredictiveLogLikelihood as SVGP_PLL

def train_gp(
    model, train_x, train_y, 
    num_epochs=100, 
    train_batch_size=1024,
    lr=0.001, gamma=0.2,
    model_name="DCSVGP",
    mll_type="ELBO",
    device="cpu", 
    beta1=1.0, beta2=0.001,
    ):
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    optimizer = torch.optim.Adam([ 
        {'params': model.parameters()},   
    ], lr=lr)
    milestones = [int(num_epochs*len(train_loader)/3), int(2*num_epochs*len(train_loader)/3)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    
    mll_name = model_name + "_" + mll_type
    mll = eval(mll_name)(model.likelihood, model, num_data=train_y.size(0))
    
    model.train()
    kwargs = {'beta1': beta1, "beta2": beta2}
    for i in range(num_epochs):
        for _, (x_batch, y_batch) in enumerate(train_loader):
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            optimizer.zero_grad()
            kwargs['x'] = x_batch
            output = model.likelihood(model(x_batch))
            if model_name.startswith("DC"):
                loss = -mll(output, y_batch, **kwargs)
            else:
                loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        means = output.mean.cpu()
        stds  = output.variance.sqrt().cpu()
        rmse = torch.mean((means - y_batch.cpu())**2).sqrt()
        nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch.cpu()).mean()
        
        if i % 100 == 0:
            if model_name.startswith("DC"):
                print(f"\n\nEpoch {i}, loss: {loss.item():.3f}, ls: {model.covar_module.lengthscale.mean().item():.2f}, ls_mean: {model.variational_strategy.covar_module_mean.lengthscale.mean().item():.2f}, nll: {nll:.3f}, rmse: {rmse:.3e}")
            else:
                print(f"\n\nEpoch {i}, loss: {loss.item():.3f}, ls: {model.covar_module.lengthscale.mean().item():.2f}, nll: {nll:.3f}, rmse: {rmse:.3e}")
            sys.stdout.flush()

    return model


def eval_gp(
    model, test_x, test_y,
    test_batch_size=1024, 
    device="cpu"):

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    model.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, _ in test_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
            preds = model.likelihood(model(x_batch))
            if device == "cuda":
                means = torch.cat([means, preds.mean.cpu()])
                variances = torch.cat([variances, preds.variance.cpu()])
            else:
                means = torch.cat([means, preds.mean])
                variances = torch.cat([variances, preds.variance])

    means = means[1:]
    variances = variances[1:]
    
    return means, variances



