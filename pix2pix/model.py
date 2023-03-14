import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def encode(in_channels:int, out_channels:int ,k_size:int, apply_batchnorm:bool=True):
    output = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, k_size, stride=2, padding=1, bias=False)
    )

    if apply_batchnorm:
        output.append(nn.BatchNorm2d(out_channels))

    output.append(nn.LeakyReLU())
    
    return output


def decode(in_channels:int, out_channels:int, k_size:int, apply_dropout:bool=False):
    output = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, k_size, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels)
    )
    if apply_dropout:
        output.append(nn.Dropout(.5))
    output.append(nn.ReLU())
    
    return output


def desnormalize(img):
    if img.dim() == 3:
        img = img.unsqueeze(0)
        
    img = np.transpose(img.detach().numpy(),axes=(0,2,3,1))
    img = img * 127.5 + 127.5
    
    return img.astype('uint8')




class Generator(nn.Module):
    def __init__(self) -> None: 
        super().__init__()
        
        self.downsamplig = nn.Sequential(
            encode(3, 64, 4, apply_batchnorm=False),
            encode(64, 128, 4),
            encode(128, 256, 4),
            encode(256, 512, 4),
            encode(512, 512, 4),
            encode(512, 512, 4),
            encode(512, 512, 4),
            encode(512, 512, 4, apply_batchnorm=False)
        )

        self.upsamplig = nn.Sequential(
            decode(512, 512, 4, apply_dropout=True),
            decode(512*2, 512, 4, apply_dropout=True),
            decode(512*2, 512, 4, apply_dropout=True),
            decode(512*2, 512, 4),
            decode(512*2, 256, 4),
            decode(256*2, 128, 4),
            decode(128*2, 64, 4)
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
            )    

    def forward(self, x):
        acts = []

        for layer in self.downsamplig:
            x = layer(x)
            acts.append(x[:])
        
        
        for a, layer in zip(acts[::-1][1:], self.upsamplig):
            x = layer(x)
            print(f'shape of x: {x.shape}')
            print(f'shape of a: {a.shape}')
            print()

            x = torch.cat([x.clone(), a], dim=1)
        
        x = self.last(x)
        return x
    

    def generate_img(self, test_input, real, file_path:str):
        generated = self.forward(test_input.unsqueeze(0))

        generated = desnormalize(generated)
        test_input = desnormalize(test_input)
        real = desnormalize(real)
        
        plt.figure(figsize=(12, 8))
        img_list = [test_input[0], real[0], generated[0]]
        title = ['Input image', 'Real (ground truth)', 'Generated image (fake)']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            plt.imshow(img_list[i])
            plt.axis('off')
        plt.savefig(file_path, bbox_inches='tight')
        plt.clf()


        

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = [
            encode(6, 64,4,False),
            encode(64, 128, 4),
            encode(128, 256, 4),
            nn.ZeroPad2d(1)
        ]


        self.l2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ZeroPad2d(1)
            
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(512, 1, 4, stride=1),
            nn.Sigmoid()
        )



    def forward(self, x, y):
        input = torch.concat([x, y], 1)

        for layer in self.l1:
            input = layer(input)
        input = self.l2(input)
        input = self.l3(input)
        
        return input
    

