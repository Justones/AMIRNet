import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=10000, help='maximum number of epochs to train the total model.')

parser.add_argument('--lr', type=float, default=5e-4, help='learning rate of encoder.')

parser.add_argument('--patch_size', type=int, default=256, help='patcphsize of input.')
parser.add_argument('--batch_size', type=int, default=32, help='batchsize of input.')

parser.add_argument('--num_workers', type=int, default=3, help='number of workers.')

# path
parser.add_argument('--data_path', type=str, default='./',  help='where clean images of denoising saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')

# setting

parser.add_argument('--resume',type=int, default=False, help='reload pretrained weights')
parser.add_argument('--resume_path',type=str, default='./', help='reload weights from this path')

parser.add_argument('--save_frequency',type=int, default=100, help='frequency of save the model')
parser.add_argument('--test_frequency',type=int, default=20, help='frequency of test')

parser.add_argument('--frequency_clustering',type=int, default=200, help='this should be consistent with the hierarchical layers')
options = parser.parse_args()
