import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', default='00', type=str, help='ID for logging.')
    parser.add_argument('--gpu', default='0', type=str, help='Visible GPU.')
    parser.add_argument('--data', default='Digit-5', type=str, help='Digit-5 or DomainNet')
    parser.add_argument('--n_training_runs', default=5, type=str, help='How many times we train for given subset (excl. random sampling)')
    parser.add_argument('--in_class_ranking', default=False, type=bool, help='Whether the subsets are ranked in class or global')
    parser.add_argument('--random', default=True, type=bool, help='Whether to use random sampling')
    parser.add_argument('--n_random_subsets', default=12, type=int, help='Number of how many random samples for given subset size.')
    parser.add_argument('--ssim', default=True, type=bool, help='Whether to use SSIM sampling')
    parser.add_argument('--feature', default=True, type=bool, help='Whether to use Learned Feature Space Distance sampling')
    parser.add_argument('--dist_measure', default='cos', type=str, help='What feature distance measure to use, either "cos" or "euclid"')
    parser.add_argument('--autoencoder', default=True, type=bool, help='Whether to use Autoencoder sampling')

    args = parser.parse_args()
    return args
