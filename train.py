import numpy as np
from data import load_data
from cnn import ConvNet
from trainer import Trainer
import time

from PIL import Image

# 데이터 읽기
img_pixel = 16

(x_train, t_train), (x_test, t_test), (x_val, t_val) = load_data(img_pixel)

def show_first_image():
    # Data Visualization
    image = x_train[0]

    image = image.reshape(img_pixel,img_pixel)

    pil_img = Image.fromarray((image * 255).astype(np.uint8), 'L')
    pil_img.show()

# show_first_image()

max_epochs = 20

network = ConvNet(input_dim=(1,img_pixel,img_pixel),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=4, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test, x_val, t_val,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.01},
                  evaluate_sample_num_per_epoch=1000)

start = time.time()
trainer.train()
end = time.time()

elapsed_time = end - start

print(f"경과 시간: {int(elapsed_time // 3600)} 시간 {int((elapsed_time % 3600) // 60)} 분 {int(elapsed_time % 60)} 초")
