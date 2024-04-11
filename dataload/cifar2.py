import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import easydict 
from .preprocessing import get_target_label_idx, global_contrast_normalization

class CIFAR_loader(data.Dataset):
    """Preprocessing을 포함한 dataloader를 구성"""
    def __init__(self, data, target, transform, target_transform):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            # x = Image.fromarray(x, mode='L')
            x = self.transform(x)
            
        if self.target_transform is not None:
            y = self.target_transform(y)
            
        return x, y

    def __len__(self):
        return len(self.data)

def get_cifa10(args, normal_class):
    
    n_classes = 2  # 0: normal, 1: outlier
    normal_classes = tuple([normal_class])
    outlier_classes = list(range(0, 10))
    outlier_classes.remove(normal_class)

    # Pre-computed min and max values (after applying GCN) from train data per class
    min_max = [(-28.94083453598571, 13.802961825439636),
                (-6.681770233365245, 9.158067708230273),
                (-34.924463588638204, 14.419298165027628),
                (-10.599172931391799, 11.093187820377565),
                (-11.945022995801637, 10.628045447867583),
                (-9.691969487694928, 8.948326776180823),
                (-9.174940012342555, 13.847014686472365),
                (-6.876682005899029, 12.282371383343161),
                (-15.603507135507172, 15.2464923804279),
                (-6.132882973622672, 8.046098172351265)]

    # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                    transforms.Normalize([min_max[normal_class][0]] * 3,
                                                            [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

    # outlier_classes들은 1 normal은 0
    target_transform = transforms.Lambda(lambda x: int(x in outlier_classes))

    #데이터 불러오기, 학습여부  o
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    
    
    target_normal_index = trainset.classes.index(args.normal_class)
    x_train = trainset.data
    y_train = np.array(trainset.targets)
    # x_train = x_train[np.where(np.array(y_train)==target_normal_index)]
    # y_train = y_train[np.where(y_train==target_normal_index)]
    data_train = CIFAR_loader(x_train, y_train, transform=transform, target_transform=target_transform)
    

    #데이터 불러오기, 학습여부  x
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    x_test = testset.data
    y_test = np.array(testset.targets)
    y_test = np.where(y_test==target_normal_index, 0, 1)
    test_dataset = CIFAR_loader(x_test, y_test, vaild_transform)
    
    
    dataset_size = len(data_train)
    train_size = int(dataset_size * 0.9)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset = random_split(data_train, [train_size, validation_size])
    #학습용 셋은 섞어서 뽑기
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.BATCH_SIZE,
                                            shuffle=True, 
                                            num_workers=0
                                            )
    
    vaildloader = torch.utils.data.DataLoader(validation_dataset, 
                                            batch_size=args.BATCH_SIZE,
                                            shuffle=True, 
                                            num_workers=0
                                            )

    #테스트 셋은 normal과 abnormal이 함께 들어가 있음
    testloader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=args.TEST_BATCH_SIZE,
                                            shuffle=False, 
                                            num_workers=0
                                            )
    #클래스들
    classes = trainset.classes
    
    return trainloader, vaildloader, testloader, classes

if __name__=="__main__":
    
    args = easydict.EasyDict({
       'num_epochs':50,
       'num_epochs_ae':50,
       'lr':1e-3,
       'lr_ae':1e-3,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'BATCH_SIZE':8,
       'TEST_BATCH_SIZE':4,
       'pretrain':False,
       'latent_dim':32,
       'normal_class':'airplane',        # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                })
    dataloader_train, dataloader_vaild, dataloader_test, classes = get_cifa10(args)
    
    #이미지 확인하기

    def imshow(img):
        img = img / 2 + 0.5     # 정규화 해제
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # 학습용 이미지 뽑기
    dataiter = iter(dataloader_train)
    images, labels = next(dataiter)

    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))

    # 이미지별 라벨 (클래스) 보여주기
    print(' '.join('%5s' % classes[labels[j]] for j in range(args.BATCH_SIZE)))