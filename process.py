
from tqdm import tqdm
import torch 
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from Networks.model import pretrain_autoencoder, DeepSVDD_network
import time

class TrainerDeepSVDD:
    def __init__(self, args, data_loader, device):
        self.args = args
        self.train_loader = data_loader
        self.device = device
        self.save_path = args.AutoEncoder_path

    def AE_train(self):
        """ DeepSVDD 모델에서 사용할 가중치를 학습시키는 AutoEncoder 학습 단계"""
        ae = pretrain_autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)           # torch.Size([1024, 1, 28, 28])
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        """학습된 AutoEncoder 가중치를 DeepSVDD모델에 Initialize해주는 함수"""
        c = self.set_c(model, dataloader)
        net = DeepSVDD_network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, self.save_path)
    

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c
    
    def model_set(self):
        """Deep SVDD model 학습"""
        net = DeepSVDD_network(self.args.latent_dim).to(self.device)
        
        if self.args.pretrain==True:
            state_dict = torch.load(self.save_path)
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        
        return net, c, optimizer, scheduler


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def train(model, c, optimizer, scheduler, dataloader, device):
    model.train()
    total_loss = 0

    for i, (inputs, label) in enumerate(tqdm(dataloader), 0):
        # 입력 받기 (데이터 [입력, 라벨(정답)]으로 이루어짐)
        inputs = inputs.to(device)
        label = label.to(device)
        #학습
        optimizer.zero_grad()
        z = model(inputs)   # ([1024, 3, 32, 32]) -> ([1024, 128])
        loss = torch.mean(torch.sum((z - c) ** 2, dim=1))   # mean([1024, 128] - [128]) = 777
        loss.backward()
        optimizer.step()

        # 결과 출력
        total_loss += loss.item()
        
    scheduler.step()    
    
    train_avg_loss = total_loss / len(dataloader)
    
    return train_avg_loss

def validation(model, c, dataloader, device):
    
    model.eval()
    with torch.no_grad():
        total_loss = 0.0

        for i, (inputs, label) in enumerate(tqdm(dataloader), 0):
            # 입력 받기 (데이터 [입력, 라벨(정답)]으로 이루어짐)
            inputs = inputs.to(device)
            label = label.to(device)
            
            # vaild
            z = model(inputs)
            loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
            total_loss += loss.item()

        val_avg_loss = total_loss / len(dataloader)
        
    return val_avg_loss


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
  
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('./result.png')
  
def vis_result(df, normal_threshold, abnormal_threshold):
    labels = ['normal', 'outlier']
    plt.figure(figsize=(8,6))
    plt.scatter(df[df.Label == 0].index, df[df.Label == 0].Scores, c='indigo', marker='o', label='normal')
    plt.scatter(df[df.Label == 1].index, df[df.Label == 1].Scores, alpha=0.5,c='gold', marker='o', label='abnormal')
    
    plt.title('normal and abnormal DeepSVDD scores', fontsize = 12)
    plt.xlabel('Sepal.Length', fontsize = 10)
    plt.ylabel('Scores', fontsize = 10)
    plt.ylim([0, abnormal_threshold + normal_threshold])
    
    plt.legend()
    plt.axhline(normal_threshold, 0, len(df), color='blue', linestyle='--', linewidth=2)
    plt.axhline(abnormal_threshold, 0, len(df), color='red', linestyle='solid', linewidth=2)
    plt.savefig('./scores.png')

def eval(net, c, dataloader, device):
   # ROC AUC score 계산
    net.to(device)
    start_time = time.time()
    idx_label_score = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            # if self.objective == 'soft-boundary':
            #     scores = dist - self.R ** 2
            # else:
            #     scores = dist
            scores = dist
            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(
                                        labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

    
    
    test_time = time.time() - start_time
    print(test_time)
    idx_label_score

    # Compute AUC
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)

    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))
    
    roc_curve_plot(labels , scores)
    
    # threshold를 정해주어야함 
    scores_df = pd.Series(scores, name='Scores').astype(float)
    labels_df = pd.Series(labels, name='Label')
    df = pd.concat([scores_df, labels_df], axis=1)
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    normal_threshold = df.loc[df['Label']==0]['Scores'].mean()
    abnormal_threshold = df.loc[df['Label']==1]['Scores'].mean()
    
    # scores = np.where(scores>normal_threshold, 1, 0)
    vis_result(df, normal_threshold, abnormal_threshold)
    
    return labels, scores, normal_threshold, abnormal_threshold


def eval1(net, c, dataloader, device):
    net.to(device)
    start_time = time.time()
    idx_label_score = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            # if self.objective == 'soft-boundary':
            #     scores = dist - self.R ** 2
            # else:
            #     scores = dist
            scores = dist
            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(
                                        labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

    test_time = time.time() - start_time
    print(test_time)
    idx_label_score

    # Compute AUC
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)

    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))
 
def model_test(model, c, dataloader, normal_threshold, abnormal_threshold, device):

    model.eval()

    with torch.no_grad():

        test_loss_sum = 0
        test_correct, f1 = 0,0
        test_total = 0
        total_predicted, total_labels = [], []
        for images, label in dataloader:

            x_test = images.to(device)
            y_test = label.to(device)

            outputs = model(x_test)
            loss = torch.mean(torch.sum((outputs - c) ** 2, dim=1))

            test_loss_sum += loss

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
            predicted = np.where(predicted>abnormal_threshold, 0, 1)
            test_total += y_test.size(0)
            test_correct += (predicted == label.cpu().numpy()).sum().item()
            f1 += f1_score(label.cpu().numpy() , predicted) # average='macro'
            total_predicted += list(predicted)
            total_labels += list(label.cpu().numpy())

        test_avg_loss = test_loss_sum / len(dataloader)
        test_avg_accuracy = 100* (test_correct / test_total)
        test_avg_f1 = (f1 / len(dataloader))

        print('loss:', test_avg_loss)
        print('accuracy:', test_avg_accuracy)
        print('F1-score:', test_avg_f1)
        print(classification_report(total_labels, total_predicted))
        