import numpy as np
from optimizer import *
from matplotlib import pyplot as plt
import datetime
import time


class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test, x_val, t_val,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.x_val = x_val
        self.t_val = t_val
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.val_size = x_val.shape[0]
        self.iter_per_epoch = int(max(self.train_size / mini_batch_size, 1))
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.val_loss_list = []

        # set time
        self.now = str(datetime.datetime.now().month) + str(datetime.datetime.now().day) \
              + '_' + str(datetime.datetime.now().hour).zfill(2) + str(datetime.datetime.now().minute).zfill(2)

        self.w_file = open("./outputs/output_" + self.now + ".txt", "w")
        #self.w_file.write("learning rate: " + str(optimizer_param['lr']) + '\n')


    def train_step(self):
        batch_mask_train = np.random.choice(self.train_size, self.batch_size)
        x_batch_train = self.x_train[batch_mask_train]
        t_batch_train = self.t_train[batch_mask_train]


        batch_mask_val = np.random.choice(self.val_size, self.batch_size)
        x_batch_val = self.x_val[batch_mask_val]
        t_batch_val = self.t_val[batch_mask_val]
        
        grads = self.network.gradient(x_batch_train, t_batch_train)
        self.optimizer.update(self.network.params, grads)
        
        train_loss = self.network.loss(x_batch_train, t_batch_train)
        self.train_loss_list.append(train_loss)
        val_loss = self.network.loss(x_batch_val, t_batch_val)
        self.val_loss_list.append(val_loss)
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_val_sample, t_val_sample = self.x_val, self.t_val
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_val_sample, t_val_sample = self.x_val[:t], self.t_val[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            val_acc = self.network.accuracy(x_val_sample, t_val_sample)
            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)

            if self.verbose:
                print("=== epoch:" + str(self.current_epoch) + ", validation acc:" + str(val_acc) + ", traning loss:" + str(train_loss) + " ===")
                self.w_file.write("=== epoch:" + str(self.current_epoch) + ", validation acc:" + str(val_acc) + ", traning loss:" + str(train_loss) + " ===\n")
        self.current_iter += 1

    def train(self):
        start = time.time()

        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)
        test_loss = self.network.loss(self.x_test, self.t_test)

        if self.verbose:
            self.w_file.write("=============== Final Test Accuracy & Loss ===============\n")
            self.w_file.write("test acc:" + str(test_acc) + ", test loss:" + str(test_loss) + '\n')
            print("=============== Final Test Accuracy & Loss ===============")
            print("test acc:" + str(test_acc) + ", test loss:" + str(test_loss))

            x = np.arange(1, self.epochs + 1)

            y1 = self.train_acc_list
            y2 = self.val_acc_list

            y3 = [self.train_loss_list[i] for i in range(len(self.train_loss_list) // self.iter_per_epoch)]
            y4 = [self.val_loss_list[i] for i in range(len(self.val_loss_list) // self.iter_per_epoch)]

            # Plot for Accuracy
            plt.figure(figsize=(12, 5))  # Set the figure size
            plt.subplot(1, 2, 1)
            plt.plot(x, y1, label='Training Accuracy', color='blue')
            plt.plot(x, y2, label='Validation Accuracy', color='orange')
            plt.title('Training/Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Plot for Loss
            plt.subplot(1, 2, 2)
            plt.plot(x, y3, label='Training Loss', color='blue')
            plt.plot(x, y4, label='Validation Loss', color='orange')
            plt.title('Training/Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.savefig('./outputs/output_' + self.now + '.png', dpi=200)
            end = time.time()
            elapsed_time = end - start

            self.w_file.write(
                f"경과 시간: {int(elapsed_time // 3600)} 시간 {int((elapsed_time % 3600) // 60)} 분 {int(elapsed_time % 60)} 초\n")
            self.w_file.close()