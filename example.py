import argparse
import random
import torch
from src.utils import load_data_1d, load_data, str2bool
from src.svgp import SVGP
from src.dcsvgp import DCSVGP
from src.train_eval import train_gp, eval_gp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="DCSVGP", help='type of model, choice of SVGP, DCSVGP')
    parser.add_argument('--beta2', type=float, default=0, help='value of beta2 for DCSVGP and DCPPGPR')
    parser.add_argument('--mll_type', type=str, default="ELBO", help='type of mll, choices of ELBO or PLL')
    parser.add_argument('--kernel_type', type=str, default="se", help='type of kernels, choice of se, matern1/2, matern3/2, and matern5/2')
    parser.add_argument('--ARD', type=str, default='false', help='name of dataset, choice of 1D and pol, choice of true of false')
    parser.add_argument('--dataset', type=str, default="1D", help='name of dataset, choice of 1D and pol')
    parser.add_argument('--seed', type=int, default=0, help='torch random seed')
    parser.add_argument('--num_inducing', type=int, default=10, help='number of inducing points')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='size of training minibatch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    return parser.parse_args()

def main(args):
    model_name = args.model_name
    dataset = args.dataset
    seed = args.seed
    num_inducing = args.num_inducing
    kernel_type = args.kernel_type
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    lr = args.learning_rate
    mll_type = args.mll_type
    beta2 = args.beta2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ARD = str2bool(args.ARD)

    torch.set_default_dtype(torch.double)
    torch.manual_seed(seed)

    if dataset == "1D":
        train_x, train_y, test_x, test_y = load_data_1d()
    elif dataset == "pol":
        train_x, train_y, _, _, test_x, test_y = load_data(data_dir='./data/uci/', dataset=dataset, seed=seed)
    dim = train_x.shape[1]
    train_n = train_x.shape[0]
    ard_num_dims = dim if ARD else None
    rand_index = random.sample(range(train_n), num_inducing)
    inducing_points_init = train_x[rand_index, :]
    
    model = eval(model_name)(
        inducing_points=inducing_points_init, 
        ard_num_dims=ard_num_dims,
        kernel_type=kernel_type)
    
    model = train_gp(
        model, 
        train_x, train_y, 
        num_epochs=num_epochs, 
        model_name=model_name,
        train_batch_size=train_batch_size,
        lr=lr,
        mll_type=mll_type,
        device=device,
        beta2=beta2
    )

    means, variances = eval_gp(
        model, test_x, test_y, device=device
    )

    rmse = torch.mean((means - test_y.cpu())**2).sqrt()
    nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y.cpu()).mean()
    print(f"Testing RMSE: {rmse:.3e}, Testing NLL: {nll:.3e}.")

if __name__ == "__main__":
    main(parse_args())


# DCSVGP on 1D
# python example.py --model DCSVGP --mll_type ELBO --dataset 1D --learning_rate 0.01 --num_epochs 1000 --num_inducing 10
# SVGP on 1D
# python example.py --model SVGP --mll_type ELBO --dataset 1D --learning_rate 0.01 --num_epochs 1000 --num_inducing 10 
# DCPPGPR on 1D
# python example.py --model DCSVGP --mll_type PLL --dataset 1D --learning_rate 0.01 --num_epochs 1000 --num_inducing 10
# PPGPR on 1D
# python example.py --model SVGP --mll_type PLL --dataset 1D --learning_rate 0.01 --num_epochs 1000 --num_inducing 10 

# DCSVGP on pol
# python example.py --model DCSVGP --mll_type ELBO --dataset pol --learning_rate 0.005 --num_epochs 300 --num_inducing 500
# SVGP on pol
# python example.py --model SVGP --mll_type ELBO --dataset pol --learning_rate 0.005 --num_epochs 300 --num_inducing 500 
# DCPPGPR on pol
# python example.py --model DCSVGP --mll_type PLL --dataset pol --learning_rate 0.005 --num_epochs 300 --num_inducing 500
# PPGPR on pol
# python example.py --model SVGP --mll_type PLL --dataset pol --learning_rate 0.005 --num_epochs 300 --num_inducing 500 

