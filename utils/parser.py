import argparse

parser = argparse.ArgumentParser(description='Arguments for running the scripts')

parser.add_argument('--data_path', type=str, default='./', help='Path to the unziped dataset directories(H36m/AMASS/3DPW)')
parser.add_argument('--missing_mode', type=str, default='random', choices=['random'], help="Choose which part is missing")
parser.add_argument('--total_frame', type=int, default=25, help="Number of total input frames")
parser.add_argument('--missing_frame', type=int, default=20, help="Number of total missing frames")
parser.add_argument('--missing_joint', type=int, default=5, help="Number of total missing joints")
parser.add_argument('--total_joint', type=int, default=22, help="Number of total input joints")

parser.add_argument('--n_epoch', type=int, default=30, help= 'Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256,help= 'Batch size')
parser.add_argument('--base_lr', type=int, default=1e-02,help= 'Learning rate')
parser.add_argument('--seed', type=int, default=1234567890,help= 'Seed')

parser.add_argument('--model_name', type=str, default='tawls', help= 'model')
parser.add_argument('--wls_hidden_scale', type=int, default=1, help= 'wls_hidden_scale')
parser.add_argument('--wls_n_hidden', type=int, default=1,help= 'wls_n_hidden')
parser.add_argument('--ta_n_heads', type=int, default=1,help= 'ta_heads')
parser.add_argument('--dropout', type=int, default=0.0,help= 'Seed')
parser.add_argument('--model_path', type=str, default='', help= 'Directory with the models checkpoints ')

config = parser.parse_args()