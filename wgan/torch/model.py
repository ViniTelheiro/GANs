import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.ConvTranspose2d(100, 256, 4, stride=2, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU()
    )
    self.conv2 = nn.Sequential(
        nn.ConvTranspose2d(256, 128, 3, stride=2,padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU()
    )
    
    self.conv3 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 4, stride=2,padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU()
    )

    self.conv4 = nn.Sequential(
        nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
        nn.Tanh()
    )


  def __random_noise(self, b_size:int):
    return torch.randn((b_size, 100, 1, 1))

  def forward(self,b_size:int):
    x = self.__random_noise(b_size)#.cuda(0)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    return x


  def generate_img(self,path:str):
    generated_images = self.forward(b_size=16)#.detach().to('cpu')
    generated_images = np.transpose(generated_images.detach().numpy(),axes=(0,2,3,1))
    generated_images = generated_images * 127.5 + 127.5
    fig = plt.figure(figsize=(10,10))
    for i in range(generated_images.shape[0]):
      plt.subplot(4,4,i+1)
      plt.imshow(generated_images[i,:,:,0] , cmap='gray')
      plt.axis('off')
    plt.savefig(path)


class Discriminator(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 64, 5, padding='same'),
        nn.LeakyReLU(),
        nn.Dropout(.3)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(64, 128, 5, padding='same'),
        nn.LeakyReLU(),
        nn.Dropout(.3)
    )
    self.linear = nn.Sequential(
        nn.Linear(100352,1),
        nn.Sigmoid()
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = torch.flatten(x,1)
    x = self.linear(x)
    return x

