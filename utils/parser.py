import argparse

parser = argparse.ArgumentParser(description='Arguments for running the scripts')

parser.add_argument('--data_dir', type=str, default='../datasets/', help='Path to the unziped dataset directories(H36m/AMASS/3DPW)')
parser.add_argument('--total_frame', type=int, default=25, help="Number of total input frames")
parser.add_argument('--missing_frame', type=int, default=20, help="Number of total missing frames")
parser.add_argument('--missing_joint', type=int, default=5, help="Number of total missing joints")
parser.add_argument('--total_joint', type=int, default=18, help="Number of total input joints")

parser.add_argument('--mode', type=str, default='train', choices=['train','test'], help= 'Choose to train or test')
parser.add_argument('--n_epochs', type=int, default=50, help= 'Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256,help= 'Batch size')
parser.add_argument('--lr', type=int, default=1e-02,help= 'Learning rate')
parser.add_argument('--use_scheduler', type=bool, default=True, help= 'MultiStepLR scheduler')
parser.add_argument('--milestones', type=list, default=[15, 25, 35, 40], help= 'The epochs after which the learning rate is adjusted by gamma')
parser.add_argument('--gamma', type=float, default=0.1, help= 'Gamma correction to the learning rate, after reaching the milestone epochs')
parser.add_argument('--clip_grad', type=float, default=None, help= 'Select max norm to clip gradients')
parser.add_argument('--model_path', type=str, default='./checkpoints/CKPT_3D_H36M', help= 'Directory with the models checkpoints ')

args = parser.parse_args()




