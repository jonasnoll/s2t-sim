import argparse


# Phase one / baseline related options

def parse_opts():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='./data/train', type=str, help='Input file path')
    # parser.add_argument('--normal_class', default='0', type=str, help='normal_class_folder_name')
    # parser.add_argument('--g_learning_rate', default='0.001', type=float, help='g_learning_rate')
    # parser.add_argument('--d_learning_rate', default='0.0001', type=float, help='d_learning_rate')
    # parser.add_argument('--adversarial_training_factor', default='0.5', type=float, help='loss parameter for generator (reconstruction and adversaria)')
    # parser.add_argument('--sigma_noise', default='0.9', type=float, help='sigma of noise added to the iamges')
    # parser.add_argument('--epoch', default=4, type=int, help='Epoch for training') # Was 100
    # parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    # parser.add_argument('--n_threads', default=8, type=int, help='Number of threads for multi-thread loading')
    # parser.add_argument('--batch_shuffle', default=True, type=bool, help='shuffle input batch or not')
    # parser.add_argument('--drop_last', default=True, type=bool, help='drop the remaining of the batch if the size doesnt match minimum batch size')
    # parser.add_argument('--image_grids_numbers', default=64, type=int, help='total number of grid squares to be saved every / few epochs')
    # parser.add_argument('--n_row_in_grid', default=10, type=int, help=' Number of images displayed in each row of the grid images.')
    # parser.add_argument('--frame_size', default=45, type=int, help='one side size of the square patch to be extracted from each frame')
    # parser.add_argument('--final_d_path', default='', type=str, help='final d model save path')
    # parser.add_argument('--final_g_path', default='', type=str, help='final g model save path')

    parser.add_argument('--exp_id', default='00', type=str, help='ID for logging.')
    parser.add_argument('--gpu', default='0', type=str, help='Visible GPU.')
    parser.add_argument('--data', default='Digit-5', type=str, help='Digit-5 or DomainNet')
    parser.add_argument('--n_training_runs', default=2, type=str, help='How many times we train for given subset')  # TODO: Change back to 5
    parser.add_argument('--in_class_ranking', default=False, type=bool, help='Whether the subsets are ranked in class or global')  # TODO: Change back to 5
    parser.add_argument('--random', default=False, type=bool, help='Whether to use random sampling')
    parser.add_argument('--n_random_subsets', default=2, type=int, help='Number of how many random samples for given subset size.')  # TODO: Change back to 12
    parser.add_argument('--ssim', default=True, type=bool, help='Whether to use SSIM sampling')

    args = parser.parse_args()
    return args
