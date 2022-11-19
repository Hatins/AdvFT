import argparse
from Utility.utilities_AdvFT import *
from torch.utils.data.dataloader import DataLoader
from Model.gan import GeneratorA

parser = argparse.ArgumentParser(description='Training the Victim model for surrogate attack',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',type=str,default='Mnist',choices=['Mnist','Fashion_mnist','Cifar10'], help='choosing the dataset for victim model')
parser.add_argument('--model',type=str,default='Lenet',choices=['Lenet','resnet18','resnet_draw'],help = 'choosing the model for victim model')
parser.add_argument('--margin',type=int,default=100)
parser.add_argument('--training_epoch',type=int,default=100,help='setting the epoch for victim model training')
parser.add_argument('--fine_tune_epoch',type=int,default=100,help='setting the epoch for victim model training')
parser.add_argument('--batch',type=int,default=64,help='setting the batch_size for victim model training')
parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
parser.add_argument('--num_worker',type=int,default=0,help='setting the num_worker for victim model training')
parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
parser.add_argument('--optimizer',type=str,default='Adam',choices=['Adam','SGD'], help='setting of the training of model_R')
parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")
parser.add_argument('--steps', nargs='+', default=[], type=float,help="Percentage epochs at which to take next step")
parser.add_argument('--I_g', type=int, default=500, help="Number of generator iterations per epoch_iter")
parser.add_argument('--I_f', type=int, default=5, help="Number of discriminator iterations per epoch_iter")
parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
args = parser.parse_args()

if __name__ == '__main__':
    args.device = ('cuda:0')
    transform_train,transform_test = transform_setting(args)
    Victim_model = setting_model(args)

    train_path = r'./Dataset/' + '/'+ args.dataset + '/train'
    test_path = r'./Dataset/' + '/' + args.dataset + '/test'

    train_set = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_worker)

    #TODO: training model_R, maybe do not need

    train_model_R(args, Victim_model, train_set,test_set, train_loader, test_loader)

    Generator = GeneratorA(nz=256, nc=3, img_size=args.size, activation=torch.tanh)
    ref_model = setting_model(args)
    fin_model = setting_model(args)

    pretrained_file_path = './Trained'
    pretrained_file_list = os.listdir(pretrained_file_path)
    for i in range(len(pretrained_file_list)):
        if (args.dataset in pretrained_file_list[i]):
            pretrained_file = pretrained_file_path + '/' + pretrained_file_list[i]
            print('loading pretrained file ', pretrained_file)

    ref_model.load_state_dict(torch.load(pretrained_file, map_location='cuda:0'))
    fin_model.load_state_dict(torch.load(pretrained_file, map_location='cuda:0'))

    #TODO: setting of optimizer_F

    # MNIST FASHIONMNIST GAN Adam lr = 0.0001
    # CIFAR10 GAN SGD lr = 0.0001
    optimizer_F = optim.Adam(fin_model.parameters(), lr=0.0001)

    optimizer_G = optim.Adam(Generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * args.fine_tune_epoch) for step in args.steps])
    scheduler_F = optim.lr_scheduler.MultiStepLR(optimizer_F, steps, args.scale)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)

    optimizer = [optimizer_F, optimizer_G]
    scheduler = [scheduler_F,scheduler_G]

    fine_tune_model_F(args,Generator,ref_model,fin_model,train_set,test_set,train_loader,test_loader,optimizer,scheduler)










