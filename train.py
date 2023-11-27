import numpy as np
from data import load_data
from cnn import ConvNet
from trainer import Trainer

from PIL import Image

# 데이터 읽기
img_pixel = 128

(x_train, t_train), (x_test, t_test), (x_val, t_val) = load_data(img_pixel)

def show_first_image():
    # Data Visualization
    image = x_train[0]

    image = image.reshape(img_pixel,img_pixel)

    pil_img = Image.fromarray((image * 255).astype(np.uint8), 'L')
    pil_img.show()

# show_first_image()

max_epochs = 50

count_for_lr = 0
while True:
    if count_for_lr < 12:
        if count_for_lr % 3 == 0:
            lr = 0.02
        elif count_for_lr % 3 == 1:
            lr = 0.005
        else:
            lr = 0.01
    else:
        lr = 0.01
    network = ConvNet(input_dim=(1, img_pixel, img_pixel),
                      conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                      hidden_size=100, output_size=4, weight_init_std=0.01)

    trainer = Trainer(network, x_train, t_train, x_test, t_test, x_val, t_val,
                      epochs=max_epochs, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr': lr},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()
    count_for_lr += 1

