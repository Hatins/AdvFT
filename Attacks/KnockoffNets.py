import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import argparse
from Utility.utilities_AdvFT import *
from torch.utils.data.dataloader import DataLoader
from Utility.utilities_KnockoffNets import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knockoff net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='Mnist', choices=['Mnist', 'Fashion_mnist', 'Cifar10', ], help='choosing the dataset for victim model')
    parser.add_argument('--model', type=str, default='Lenet', choices=['resnet18', 'Lenet'],help='choosing the model for victim model')
    parser.add_argument('--epoch', type=int, default=100, help='setting the epoch for victim model training')
    parser.add_argument('--lr', type=float, default=0.1, help='setting the learning rate for victim model training')
    parser.add_argument('--pt_file',type=str,default='../Trained/Mnist_Lenet_epoch_59_accuracy_99.44%.pt')
    parser.add_argument('--batch', type=int, default=64, help='setting the batch_size for victim model training')
    parser.add_argument('--num_workers', type=int, default=0, help='setting the num_worker for victim model training')
    parser.add_argument('--query_set', type=str, default='F:/Python_Project/Data-Agnostic_DNN_Model_Protection/Knockoff/dataset/Imagenet_100K', help='setting the path of ImageNet')
    parser.add_argument('--poison', type=bool, default=False, help='whether poisoning with DP')
    parser.add_argument('--query_times', type=int, default=50000, help='setting the query times')
    args = parser.parse_args()
    args.device = torch.device('cuda:0')
    transform_train,transform_test = transform_setting(args)
    Victim_model = setting_model(args)
    Surrogate_model = setting_model(args)

    Victim_model.load_state_dict(torch.load(args.pt_file, map_location='cuda:0'))
    Victim_model = Victim_model.to(args.device)

    test_path = r'../Dataset/' + args.dataset + '/' + '/test'
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    print('setting the transfer set')
    Dataset = estabilish_surrogate_dataset(args,Victim_model,transform_train)

    train_loader = DataLoader(Dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    train_surrogate_model(args, Surrogate_model, train_loader, test_set, test_loader)