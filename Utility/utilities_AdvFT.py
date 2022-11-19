import torch
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import random
import torchvision
from Model.Lenet import lenet
from Model.resnet import resnet18
from Model.resnet_draw import resnet18_draw
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

#TODO: Setting random seeds
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#TODO: Freezing the parameters of the final layer
def freeze_bn(model):
    for p in model.modules():
        if isinstance(p, torch.nn.BatchNorm2d):
            p.eval()  # freeze running mean and var
            p.weight.requires_grad = False
            p.bias.requires_grad = False
    return model

#TODO: Loss of generator
def loss_computed(args, fake, ref_model, fin_model):
    ref_model.eval()
    fin_model.eval()
    ref_feature, ref_logit = ref_model(fake)
    fin_feature, fin_logit = fin_model(fake)
    pdist = nn.PairwiseDistance(p=2)
    loss_values = torch.mean(pdist(fin_feature, ref_feature))
    return loss_values

#TODO: Training of M_R
def train_model_R(args, model, train_set, test_set, train_loader, test_loader):
    print('Victim model is training')
    num_params = sum(i.numel() for i in model.parameters() if i.requires_grad)
    print('the number of parameters is {}'.format(num_params))
    Victim_model = model.to(args.device)
    accuracy_test_list = []
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(Victim_model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = MultiStepLR(optimizer, milestones=[args.training_epoch * 0.3, args.training_epoch * 0.6,
                                                   args.training_epoch * 0.9], gamma=0.2)

    for each_epoch in range(args.training_epoch):
        Victim_model = Victim_model.to(args.device)
        Victim_model.train()

        # training
        correct_train_number = 0
        loop_train = tqdm(enumerate(train_loader), total=len(train_loader))
        for index, (inputs, labels) in loop_train:
            inputs, lables = inputs.to(args.device), labels.to(args.device)
            _, outputs = Victim_model(inputs)
            pred = outputs.argmax(dim=1)
            correct_train_number += pred.eq(labels.view_as(pred).to(args.device)).sum().item()
            loss = criterion(outputs, lables)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy_train_show = correct_train_number / len(train_set)
            loop_train.set_description(f'training_Epoch [{each_epoch + 1}/{args.training_epoch}]')
            loop_train.set_postfix(loss=loss.item(), acc_train=accuracy_train_show)
        scheduler.step()

        # testing
        loop_test = tqdm(test_loader, total=len(test_loader))
        correct_test_number = 0
        Victim_model.eval()
        with torch.no_grad():
            for inputs, labels in loop_test:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                _, outputs = Victim_model(inputs)
                pred = outputs.argmax(dim=1)
                correct_test_number += pred.eq(labels.view_as(pred)).sum().item()
                accuracy_test_show = correct_test_number / len(test_set)
                loop_test.set_description(f'testing__Epoch [{each_epoch + 1}/{args.fine_tune_epoch}]')
                loop_test.set_postfix(acc_test=accuracy_test_show)
            accuracy_test = correct_test_number / len(test_set)
        save_name = './Trained/' + args.dataset + '_' + args.model + '_epoch_{}_accuracy_{:.2f}%.pt'.format(
            each_epoch, accuracy_test * 100)
        accuracy_test_list.append(accuracy_test * 100)
        torch.save(Victim_model.state_dict(), save_name)
    accuracy_test_list = np.array(accuracy_test_list)

    print('the best accurary is {:.2f}'.format(accuracy_test_list.max()))
    save_best_name = './Trained//' + args.dataset + '_' + args.model + '_epoch_{}_accuracy_{:.2f}%.pt'.format(
        accuracy_test_list.argmax(), accuracy_test_list.max())
    save_path = os.listdir(r'./Trained/')
    for file_name in save_path:
        file_name_need_remove = r'./Trained/' + '/' + file_name
        if file_name_need_remove != save_best_name and  args.dataset in file_name_need_remove:
            os.remove(file_name_need_remove)

#TODO: Fine-tuning of M_F
def fine_tune_model_F(args, Generator, ref_model, fin_model, train_set, test_set, train_loader, test_loader, optimizer, scheduler):
    optimizer_F, optimizer_G = optimizer
    scheduler_F, scheduler_G = scheduler
    fin_model.eval()
    ref_model = ref_model.to(args.device)
    ref_model.eval()
    fin_model = freeze_bn(fin_model)
    fin_model = fin_model.to(args.device)
    Generator = Generator.to(args.device)

    for each_epoch in range(args.fine_tune_epoch):
        scheduler_F.step()
        scheduler_G.step()
        for _ in range(args.I_g):
            noise = torch.randn((args.batch, 256)).to(args.device)
            optimizer_G.zero_grad()  # set gradient to 0 in each epoch
            Generator.train()
            fake = Generator(noise)
            Loss_G = loss_computed(args, fake, ref_model, fin_model)
            Loss_G.backward()
            optimizer_G.step()
            print('loss_G = {}'.format(Loss_G))
        print('the {}/{} cycle of gan has finished'.format(each_epoch + 1, args.fine_tune_epoch))
        for each_f_iter in range(args.I_f):
            loop_train = tqdm(enumerate(train_loader), total=len(train_loader), position=0)
            freeze_bn(fin_model)
            for index, (inputs, labels) in loop_train:
                inputs, output = inputs.to(args.device), labels.to(args.device)
                noise = torch.randn((args.batch, 256)).to(args.device)
                Generator.eval()
                fake = Generator(noise).detach()
                optimizer_F.zero_grad()

                train_fin_feature, _ = fin_model(inputs)
                train_ref_feature, _ = ref_model(inputs)

                fake_fin_feature, _ = fin_model(fake)
                fake_ref_feature, _ = ref_model(fake)

                pdist = nn.PairwiseDistance(p=2)
                loss_F = max([torch.mean(pdist(train_fin_feature, train_ref_feature)) -
                              torch.median(pdist(fake_fin_feature, fake_ref_feature)) + args.margin,
                              torch.tensor(0.0, requires_grad = True).to(args.device)]) + torch.mean(pdist(train_fin_feature, train_ref_feature))

                #the results of each losses
                print('the overall loss is ', loss_F)
                print('l2 distance of train_x is ', torch.mean(pdist(train_fin_feature, train_ref_feature)))
                print('l2 distance of gan_x is ', torch.median(pdist(fake_fin_feature, fake_ref_feature)))

                loss_F.backward()
                optimizer_F.step()
                loop_train.set_description(f'Fine_Epoch [{each_epoch/args.fine_tune_epoch}：{each_f_iter + 1}/{args.I_f}]')

            loop_test = tqdm(test_loader, total=len(test_loader), position=0)
            correct_test_number = 0
            fin_model.eval()
            with torch.no_grad():
                for inputs, labels in loop_test:
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    _, outputs = fin_model(inputs)
                    pred = outputs.argmax(dim=1)
                    correct_test_number += pred.eq(labels.view_as(pred)).sum().item()
                    accuracy_test_show = correct_test_number / len(test_set)
                    loop_test.set_description(f'testing__Epoch [{each_epoch/args.fine_tune_epoch}：{each_f_iter + 1}/{args.I_f}]')
                    loop_test.set_postfix(acc_test=accuracy_test_show)
                accuracy_test = correct_test_number / len(test_set)

            save_name = './Trained/' + 'after_fine_tune' + '/' + args.dataset + '_' + args.model + '_epoch_{}_accuracy_{:.2f}%.pt'.format(
                    each_epoch, accuracy_test * 100)
        torch.save(fin_model.state_dict(), save_name)

# TODO: 2D visualization
def visualize(model, ref_model, dataloader, device):
    model.to(device)
    ref_model.to(device)
    model.eval()
    ref_model.eval()

    for n1, p1 in ref_model.named_parameters():
        for n2, p2 in model.named_parameters():
            if n1 == n2:
                print('Name:{}, Diff: {:.4f}.'.format(n1, (p1 - p2).norm()))

    features = torch.empty(0, 3).to(device)
    features_ref = torch.empty(0, 3).to(device)

    prototype_ref = copy.copy(ref_model.out.weight.detach())
    prototype_v = copy.copy(model.out.weight.detach())

    correct = 0
    correct_ref = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            feature, pred = model(data)
            feature_ref, pred_ref = ref_model(data)

            correct += pred.argmax(dim=1).eq(target).sum().item()
            correct_ref += pred_ref.argmax(dim=1).eq(target).sum().item()

            t1 = torch.cat((feature, target.unsqueeze(-1)), dim=1)
            features = torch.cat((features, t1), dim=0)

            t2 = torch.cat((feature_ref, target.unsqueeze(-1)), dim=1)
            features_ref = torch.cat((features_ref, t2), dim=0)

    features = features.to('cpu')
    features_ref = features_ref.to('cpu')
    all_labels = features[:, 2]
    indices = []
    for i in range(10):
        indices.append(torch.nonzero(all_labels.eq(i)).squeeze(-1))

    color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(1, figsize=(6, 6))
    for i in range(10):
        plt.plot(features[indices[i], 0], features[indices[i], 1], 'o', markersize=1, label=str(i), color=color_cycle[i])
        plt.plot([0, prototype_v[i, 0], 1000 * prototype_v[i, 0]], [0, prototype_v[i, 1], 1000 * prototype_v[i, 1]], '--', color=color_cycle[i])
    plt.legend(loc="lower left", markerscale=8., fontsize=10)
    plt.xlabel('$z(0)$', fontsize=14)
    plt.ylabel('$z(1)$', fontsize=14)
    plt.axis('square')
    plt.xlim([-200, 200])
    plt.ylim([-200, 200])
    #should be adjusted based on the datasets (xlim,ylim)
    plt.tight_layout()
    plt.grid()

    plt.figure(2, figsize=(6, 6))
    for i in range(10):
        plt.plot(features_ref[indices[i], 0], features_ref[indices[i], 1], 'o', markersize=1, label=str(i), color=color_cycle[i])
        plt.plot([0, prototype_ref[i, 0], 1000 * prototype_ref[i, 0]], [0, prototype_ref[i, 1], 1000 * prototype_ref[i, 1]], '--', color=color_cycle[i])
    plt.legend(loc="lower left", markerscale=8., fontsize=10)
    plt.xlabel('$z(0)$', fontsize=14)
    plt.ylabel('$z(1)$', fontsize=14)
    plt.axis('square')
    plt.xlim([-200, 200])
    plt.ylim([-200, 200])
    plt.tight_layout()
    plt.grid()

    plt.figure(3, figsize=(6, 6))
    for i in range(10):
        plt.plot([0, prototype_v[i, 0]], [0, prototype_v[i, 1]], '-o', color=color_cycle[i], label=str(i))
        plt.plot([0, prototype_ref[i, 0]], [0, prototype_ref[i, 1]], '--o', color=color_cycle[i], linewidth=3)
    plt.legend(loc="lower left", markerscale=1., fontsize=10)
    plt.xlabel('$w_i(0)$', fontsize=14)
    plt.ylabel('$w_i(1)$', fontsize=14)
    plt.axis('square')
    plt.xlim([-1., 1.])
    plt.ylim([-1., 1.])
    plt.tight_layout()
    plt.grid()
    plt.show()

#TODO :Setting the transform
def transform_setting(args):
    if args.dataset == 'Mnist':
        args.num_classes = 10
        args.size = 28
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif args.dataset == 'Fashion_mnist':
        args.num_classes = 10
        args.size = 28
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif args.dataset == 'Cifar10':
        args.num_classes = 10
        args.size = 32
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    return transform_train,transform_test

#TODO :Setting model
def setting_model(args):
    if args.model == 'Lenet':
        model_R = lenet(num_classes=args.num_classes)
    elif args.model == 'resnet18':
        model_R = resnet18(num_classes=args.num_classes, penultimate_2d=False)
    elif args.model == 'resnet_draw':
        model_R = resnet18_draw(num_classes=args.num_classes)
    return model_R

#TODO :DP poisoning
def sigmoid(z):
    return torch.sigmoid(z)
def inv_sigmoid(p):
    assert (p >= 0.).any()
    assert (p <= 1.).any()
    return torch.log(p / (1 - p))
def reverse_sigmoid(y, beta, gamma):
    return beta * (sigmoid(gamma * inv_sigmoid(y)) - 0.5)
def DP_poison(args,y_v):
    beta = 0.7
    gamma = 0.5
    y_prime = y_v - reverse_sigmoid(y_v, beta, gamma)
    repeat_prime = ((y_prime.sum(dim=len(y_prime.shape)-1)).unsqueeze(1)).repeat(1,args.num_classes)
    y_prime /= repeat_prime
    return y_prime

#TODO :Soft cross entropy
def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))
