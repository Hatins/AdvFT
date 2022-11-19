import torch
import torchvision
from Utility.utilities_AdvFT import visualize
from torch.utils.data.dataloader import DataLoader
from Model.resnet_draw import resnet18_draw

device = torch.device('cpu')

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize([28, 28]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
])
# transform_train.transforms.append(Cutout(n_holes=1, length=16))
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize([28, 28]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
])

ref_model = resnet18_draw(num_classes=10,penultimate_2d=True)
fin_model = resnet18_draw(num_classes=10,penultimate_2d=True)


pretrained_file_ref = '../Trained/Mnist_resnet_draw_epoch_55_accuracy_99.46%.pt'
pretrained_file_fin = '../Trained/after_fine_tune/Mnist_resnet_draw_epoch_73_accuracy_99.47%.pt'

ref_model.load_state_dict(torch.load(pretrained_file_ref, map_location='cuda:0'))
fin_model.load_state_dict(torch.load(pretrained_file_fin, map_location='cuda:0'))

train_path = r'../Dataset/' + '/'+ 'Mnist' + '/train'
test_path = r'../Dataset/' + '/' + 'Mnist' + '/test'

train_set = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

visualize(fin_model, ref_model, test_loader, device)