import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
sys.path.append(os.path.abspath(os.path.join(__file__,"..")))
from Utility.utilities_DFME import *
import argparse
from Utility.utilities_AdvFT import *
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from Model.gan import GeneratorA


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Datafree attack')
    parser.add_argument('--query_times',type=int,default=6*(10**6),help = 'setting the query_times')
    parser.add_argument('--model',type=str,default='Lenet',choices=['resnet18','Lenet'],help='setting the surrogate model')
    parser.add_argument('--seed',type=int,default=1,metavar='S',help='setting the random seed')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--num_classes',type=int,default=10,help='setting the number of classes')
    parser.add_argument('--dataset', type=str, default='Mnist', choices=['Mnist', 'Cifar10', 'Gtsrb', 'Fashion_mnist'], help='choosing the dataset')
    parser.add_argument('--pt_file',type=str,default='../Trained/Mnist_Lenet_epoch_59_accuracy_99.44%.pt')
    parser.add_argument('--batch',type=int,default=128,help='setting the batch_size')
    parser.add_argument('--num_workers',type=int,default=0,help='setting the num_workers')
    parser.add_argument('--size',type=int,default=28,help='setting the image size')
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--poison',type=bool,default=True,help='whether poisoning with DP')

    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'] )
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])
    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--grad_epsilon', type=float, default=1e-3)
    '''GAN'''
    parser.add_argument('--noise_size',type=int,default=256,help='size of random noise input to generator')
    parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
    parser.add_argument('--g_iter', type=int, default=1, help="Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help="Number of discriminator iterations per epoch_iter")
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
    args = parser.parse_args()

    args.device = torch.device('cuda:0')
    args.G_activation = torch.tanh
    '''setting the random seed'''
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform_train,transform_test = transform_setting(args)
    Victim_model = setting_model(args)

    Victim_model.load_state_dict(torch.load(args.pt_file, map_location=args.device))
    Victim_model.eval()

    Victim_model = Victim_model.to(args.device)
    Surrogate_model = setting_model(args)
    Surrogate_model = Surrogate_model.to(args.device)

    print('the victim model is loading '+ args.pt_file)
    print('the Victim model is ' + args.model)

    Generator = GeneratorA(nz=args.noise_size, nc=3, img_size=args.size, activation=torch.tanh)
    Generator = Generator.to(args.device)

    args.Generator = Generator
    args.Surrogate_model = Surrogate_model
    args.Victim_model = Victim_model

    args.cost_per_iteration = args.batch * (args.g_iter * (args.grad_m + 1) + args.d_iter)

    number_epochs = args.query_times // (args.cost_per_iteration * args.epoch_itrs) + 1

    print('Total query times is {}'.format(args.query_times))
    print('cost per iterations: ', args.cost_per_iteration)
    print('Total number of epoch: ', number_epochs)

    optimizer_S = optim.SGD(Surrogate_model.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(Generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    best_acc = 0
    acc_list = []

    test_path = r'../Dataset/' + args.dataset + '/' + '/test'
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)


    for epoch in range(1, number_epochs + 1):
        # Train
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        train(args, Victim_model=Victim_model, Surrogate_model=Surrogate_model,Generator=Generator,
              optimizer=[optimizer_S,optimizer_G],epoch=epoch,device=args.device)

        acc = a_test(args, Surrogate_model=Surrogate_model, Generator=Generator, device=args.device, test_loader=test_loader, epoch=epoch)

        if acc>best_acc:
            best_acc = acc

        print('the best_acc is {}'.format(best_acc))
