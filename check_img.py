from dataload.cifar import get_cifa10
import easydict 

import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img):
    img = img / 2 + 0.5     # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


args = easydict.EasyDict({
       'BATCH_SIZE':8,
       'TEST_BATCH_SIZE':4,
       'pretrain':False,
       'latent_dim':32,
       'normal_class':'airplane',        # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                })
dataloader_train, dataloader_vaild, dataloader_test, classes = get_cifa10(args)



# 학습용 이미지 뽑기
dataiter = iter(dataloader_vaild)
images, labels = next(dataiter)


# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))

# 이미지별 라벨 (클래스) 보여주기
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
