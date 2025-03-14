import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='../data')
parser.add_argument('--dataset', type=str, choices=['rdvs', 'vidsod_100', 'dvisal'])
parser.add_argument('--epoch',       type=int,   default=15,   help='epoch number')
parser.add_argument('--lr',          type=float, default=2e-4,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=5,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=352,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=15,    help='every n epochs decay learning rate')
parser.add_argument('--load',        type=str,   default=None,  help='train from checkpoints')
parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')

parser.add_argument('--rgb_label_root',      type=str, default='/data3/zl/salient_object_detection/rgbd_videos/train',           help='the training rgb images root')
parser.add_argument('--depth_label_root',    type=str, default='./Data/TrainDataset/depth/',         help='the training depth images root')
parser.add_argument('--gt_label_root',       type=str, default='./Data/TrainDataset/GT/',            help='the training gt images root')
parser.add_argument('--val_rgb_root',        type=str, default='/data3/zl/salient_object_detection/rgbd_videos/test_refine',      help='the test rgb images root')
parser.add_argument('--val_depth_root',      type=str, default='./Data/NJU2K/depth/',    help='the test depth images root')
parser.add_argument('--val_gt_root',         type=str, default='./Data/NJU2K/GT/',       help='the test gt images root')
parser.add_argument('--save_path',           type=str, default='./cache/flow_1e5_exp2/',    help='the path to save models and logs')
parser.add_argument('--resume', default=False, action="store_true")


opt = parser.parse_args()

 