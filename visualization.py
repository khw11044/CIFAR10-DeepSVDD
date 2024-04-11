import matplotlib.pyplot as plt 



def draw_training_result(train_loss_list, train_epochs, val_loss_list, val_epochs):
    plt.figure(figsize=(10,6))
    
    plt.plot(train_epochs, train_loss_list, label='train_err', marker = '.')
    plt.plot(val_epochs, val_loss_list, label='val_err', marker = '.')
    
    plt.grid()
    plt.legend()
    plt.savefig('./train_result')
    
    #plt.show()