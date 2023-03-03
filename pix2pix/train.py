from dataset import get_dataset
from model import Generator, Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from random import randint


if __name__ == '__main__':
    dataset = get_dataset(train=True)
    train_dataloader = DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=0)
    
    test_dataset = get_dataset(train=False)

    criterion = nn.BCELoss()
    l1 = nn.L1Loss()

    generator = Generator()
    discriminator = Discriminator()

    generator_optim = optim.Adam(generator.parameters(), lr=2e-4, betas=[.5, .999])
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=2e-4, betas=[.5, .999])

    gen_losses = []
    disc_losses = []

    save_path = './log'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)   
    
    epochs = 40000//len(train_dataloader)  

    for epoch in range(1, epochs+1):
        avg_gen_loss = avg_disc_loss = 0
        generator.train()
        discriminator.train()
        with tqdm(train_dataloader, unit='Batch', total=len(train_dataloader)) as pbar:
            for i, (input_img, gt) in enumerate(train_dataloader):
                pbar.set_description(f'Epoch {epoch}')
                
                #Train discriminator
                d_real_output = discriminator(x=input_img, y=gt)
                disc_loss_real = criterion(d_real_output, torch.ones(size=d_real_output.shape))

                g_output = generator(input_img)
                d_fake_output = discriminator(x=input_img, y=g_output)
                disc_loss_fake = criterion(d_fake_output, torch.zeros(size=d_fake_output.shape))

                disc_loss = disc_loss_fake + disc_loss_real
                
                discriminator_optim.zero_grad()
                generator_optim.zero_grad()
                disc_loss.backward(retain_graph=True)
                discriminator_optim.step()

                avg_disc_loss += disc_loss.item()
                
                #train Generator
                g_output = generator(input_img)
                d_fake_output = discriminator(x=input_img, y=g_output)

                gen_loss = criterion(d_fake_output, torch.ones(size=d_fake_output.shape))
                l1_loss = l1(g_output,gt)
                tot_gen_loss = gen_loss + (100 * l1_loss)

                discriminator_optim.zero_grad()
                generator_optim.zero_grad()
                tot_gen_loss.backward()
                generator_optim.step()

                avg_gen_loss += tot_gen_loss.item()

                pbar.set_postfix({'gen_loss': avg_gen_loss/(i+1), 'disc_loss':avg_disc_loss/(i+1)})
                pbar.update(1)
            
            avg_disc_loss/=(i+1)
            avg_gen_loss/=(i+1)

            gen_losses.append(avg_gen_loss)
            disc_losses.append(avg_disc_loss)

            if epoch % (epochs // 10) == 0:
                print(f'saving a checkpoint at epoch {epoch}')
                
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': generator.state_dict(),
                    'optim_state_dict': generator_optim.state_dict(),
                    'gen_losses': gen_losses
                }, os.path.join(save_path,'generator.pth'))

                torch.save({
                    'epoch':epoch,
                    'model_state_dict': discriminator.state_dict(),
                    'optim_state_dict': discriminator_optim.state_dict(),
                    'disc_losses': disc_losses
                }, os.path.join(save_path, 'discriminator.pth'))

                generator.eval()
                test_img, test_gt = test_dataset[randint(0,len(test_dataset))]
                generator.generate_img(test_input=test_img, real=test_gt, 
                                       file_path=os.path.join(save_path,f'generated_imgs_epoch{epoch}.jpg'))
    
    plt.clf()
    plt.plot(gen_losses, c='r', label='generator')
    plt.plot(disc_losses, c='b', label='discriminator')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_graph.jpg'))

                
                



                


            

