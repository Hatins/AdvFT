from tqdm import tqdm
import os
from PIL import Image
import torch
from torch.utils.data import TensorDataset
from Utility.utilities_AdvFT import DP_poison,soft_cross_entropy
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR



#TODO :Estabilish the dataset of surrogate attack
def estabilish_surrogate_dataset(args,Victim_model,transform_train):
    query_set_path = args.query_set
    filelist = os.listdir(query_set_path)
    Victim_model.eval()
    loop_save = tqdm(enumerate(filelist), total=args.query_times)
    loop_save.set_description(f'transfer set establishing...')
    image_list = []
    probability_list = []
    query_times = args.query_times
    for index, data in loop_save:
        if query_times > 0:
            image_name = query_set_path + '/' + data  #reading the images
            image_PIL = Image.open(image_name).convert('RGB')
            image_save = transform_train(image_PIL)
            image_save = image_save.unsqueeze(0)
            image_save = image_save.to(args.device)
            with torch.no_grad():
                _, probability_save = Victim_model(image_save)
                if args.poison == True:
                    probability_save = DP_poison(args,F.softmax(probability_save,dim=1))
                else:
                    probability_save = F.softmax(probability_save,dim=1)
            image_save = image_save.squeeze(0)
            image_list.append(image_save)
            probability_save = probability_save.squeeze(0)
            probability_list.append(probability_save)
            query_times = query_times - 1

    image_tensor = torch.stack([i for i in image_list], dim=0)
    probability_tensor = torch.stack([i for i in probability_list], dim=0)
    Dataset = TensorDataset(image_tensor, probability_tensor)
    return Dataset

#TODO :Training surrogate model
def train_surrogate_model(args,Surrogate_model,train_loader,test_set, test_loader):
    Surrogate_model = Surrogate_model.to(args.device)
    optimizer = optim.SGD(Surrogate_model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    accuracy_test_list = []
    accuracy_best = 0

    for each_epoch in range(args.epoch):
        Surrogate_model.train()
        Surrogate_model = Surrogate_model.to(args.device)
        correct_train_number = 0
        loop_train = tqdm(enumerate(train_loader), total=len(train_loader), position=0)
        loop_test = tqdm(test_loader, total=len(test_loader), position=0)
        for index, (inputs, probability) in loop_train:
            inputs, probability = inputs.to(args.device), probability.to(args.device)
            _, outputs = Surrogate_model(inputs)
            optimizer.zero_grad()
            loss = soft_cross_entropy(outputs, probability)
            loss.backward()
            optimizer.step()
            pred = outputs.argmax(dim=1)
            labels_item = probability.argmax(dim=1)

            correct_train_number += pred.eq(labels_item.view_as(pred).to(args.device)).sum().item()
            accuracy_train_show = correct_train_number / args.query_times
            loop_train.set_description(f'training_Epoch [{each_epoch + 1}/{args.epoch}]')
            loop_train.set_postfix(loss=loss.item(), acc_train=accuracy_train_show)

        scheduler.step()

        # testing
        correct_test_number = 0
        Surrogate_model.eval()

        with torch.no_grad():
            for data, target in loop_test:
                data, target = data.to(args.device), target.to(args.device)
                _, outputs = Surrogate_model(data)
                pred = outputs.argmax(dim=1)
                correct_test_number += pred.eq(target.view_as(pred)).sum().item()
                accuracy_test_show = correct_test_number / len(test_set)
                loop_test.set_description(f'testing__Epoch [{each_epoch + 1}/{args.epoch}]')
                loop_test.set_postfix(acc_test=accuracy_test_show)
            accuracy_test = correct_test_number / len(test_set)
            accuracy_test_list.append(accuracy_test * 100)


        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test

    print('the best accuracy is {}'.format(accuracy_best, '.2f'))