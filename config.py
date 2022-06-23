import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot_save_directory', type=str, default='./plot')
    parser.add_argument('--PDE', type=str, default='2D-Navier-Stokes', choices=['2D-Navier-Stokes',
                                                                                '2D-Fokker-Planck',
                                                                                '2D-McKean-Vlasov'])
    parser.add_argument('--model_save_directory', type=str, default='./model')
    parser.add_argument('--diffusion_coefficient', type=float, default=0.)
    parser.add_argument('--plot_domain_size', type=float, default=1.)
    parser.add_argument('--test_domain_size', type=float, default=10.)
    parser.add_argument('--number_of_iterations', type=int, default=5000)
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--refer_batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--ODE_tolerance', type=float, default=1e-7)
    parser.add_argument('--total_evolving_time', type=float, default=2.)
    parser.add_argument('--reg_f', type=float, default=0.)
    parser.add_argument('--test_frequency', type=int, default=100)
    parser.add_argument('--save_frequency', type=int, default=1000)
    parser.add_argument('--plot_frequency', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dim', type=int, default=2, choices=[2], help='only 2D problems are supported now')


    args = parser.parse_args()
    return args
