import os

import torch
import torch.nn as nn
import torch.optim as optim
import easydict 

# from Networks.model import BasicNet, MyCNNNet, TransModel
from process import TrainerDeepSVDD
from process import train, validation, eval, model_test
from visualization import draw_training_result
from dataload.cifar import get_cifa10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LR = 0.001

args = easydict.EasyDict({
       'num_epochs_ae':150,
       'num_epochs':150,
       'lr':1e-3,
       'lr_ae':1e-3,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'ae_lr_milestones': [100],
       'lr_milestones':[100],
       'BATCH_SIZE':1024,
       'TEST_BATCH_SIZE':4,
       'pretrain':True,         # 항상 True
       'load_AE':False,     # False
       'latent_dim':128,
       'normal_class':'truck',        
       'AutoEncoder_path':'./weights/',
       'save_path':'./weights/'
                })

'''
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
AUC
airplane : 67.72%
automobile : 51.84%
bird : 63.19% 
cat : 52.94%
deer : 73.46%
dog : 46.85%
frog : 76.55%
horse : 54.11%
ship : 46.80%
truck : 49.64%
'''

if __name__=="__main__":
    args.AutoEncoder_path = args.AutoEncoder_path + '{}_pretrained_AE.pth'.format(args.normal_class)
    args.save_path = args.save_path + '{}_best_model.pth'.format(args.normal_class)
    
    weight_root = './weights'
    os.makedirs(weight_root, exist_ok=True)
    dataloader_train, dataloader_vaild, dataloader_test, classes = get_cifa10(args)
    epoch_start = 0
    
    deep_SVDD = TrainerDeepSVDD(args, dataloader_train, device)
    

    if not args.load_AE:        # AE load 안하고 훈련시키면 
        deep_SVDD.AE_train()
    
    model, c, optimizer, scheduler = deep_SVDD.model_set()
    model = model.to(device)


    # 훈련 
    train_epochs, val_epochs = [], []
    train_loss_list, val_loss_list = [], []

    loss_score = 1000
    for epoch in range(epoch_start, args.num_epochs):

        train_loss= train(model, c, optimizer, scheduler, dataloader_train, device) 
        print('Model training... Epoch: {}, Loss: {:.3f}'.format(epoch, train_loss))

        if (epoch+1)%5==0:
            valid_loss = validation(model, c, dataloader_vaild, device)
            print('Model vaildation ... Loss: {:.3f}'.format(valid_loss))
            if loss_score>valid_loss:
                loss_score = valid_loss
                model = model.cpu()
                print('---'*20)
                print('lowest valid loss: {:.3f} And SAVE model'.format(valid_loss))
                torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': model.state_dict()}, args.save_path)
                model.to(device)
                
            val_loss_list.append(valid_loss)
            val_epochs.append(epoch)
            
        train_loss_list.append(train_loss)
        train_epochs.append(epoch)
    
    draw_training_result(train_loss_list, train_epochs, val_loss_list, val_epochs)           
    print('Finished Training')
    print()
    
    print('Eval')
    # eval(model, c, dataloader_test, device)
    labels, scores, normal_threshold, abnormal_threshold = eval(model, c, dataloader_test, device)
    print('normal_threshold & abnormal_threshold =>',normal_threshold, abnormal_threshold)
    # print('Test')
    # model_test(model, c, dataloader_test, normal_threshold, abnormal_threshold, device)
