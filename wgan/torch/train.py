import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os
from torch.utils.data import DataLoader
from GANs.wgan.torch.model import Generator, Discriminator
import matplotlib.pyplot as plt
from tqdm import tqdm


def gradient_penalty(d, xr, xf):
  t = torch.randn(xr.shape[0], 1, 1, 1)
  mid = t * xr + (1 - t) * xf
  mid.requires_grad_()
  pred = d(mid)
  grads = torch.autograd.grad(outputs=pred, inputs=mid,
                           grad_outputs=torch.ones_like(pred),
                           create_graph=True, retain_graph=True,
                           only_inputs=True)[0]
  gp = torch.pow(grads.norm(2,dim=1) - 1, 2).mean()
  return gp

if __name__ == '__main__':
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(.5,.5)
            ])

    download = not os.path.isdir('./data')
    if download:
        os.makedirs('./data')

    dataset = datasets.MNIST('./data', train=True, transform=transform, download=download)
    dataloader = DataLoader(dataset, 256, shuffle=True)

    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    criterion = torch.nn.BCELoss()

    gen_losses = []
    disc_losses = []

    save_dir = './log'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(1, 101):
        avg_disc_loss = avg_gen_loss = 0
        
        discriminator.train()
        generator.train()
        with tqdm(dataloader, unit='Batch', total=len(dataloader)) as pbar:
            for i, (features, _) in enumerate(dataloader):
                pbar.set_description(f'Epoch: {epoch}')
                #Train discriminator with real images
                d_real_output = discriminator(features)#.cuda(0))
                real_labels = torch.ones(size=d_real_output.shape)#.cuda(0)

                disc_loss_real = criterion(d_real_output, real_labels)

                #Train discriminator with fake images
                g_output = generator(features.shape[0])
                d_fake_output = discriminator(g_output)
                fake_labels = torch.zeros(size=d_real_output.shape)#.cuda(0)
                
                disc_loss_fake = criterion(d_fake_output, fake_labels)

                #Training discriminator:
                disc_loss = disc_loss_fake + disc_loss_real #dicriminator loss will be the sum of this two losses

                #add gradient penalty:
                gp = gradient_penalty(discriminator, features, g_output)

                disc_loss += (0.2*gp)

                discriminator_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                
                disc_loss.backward(retain_graph=True)
                discriminator_optimizer.step()

                avg_disc_loss += disc_loss.item()

                #Training generator with different output that train discriminator fake images
                g_output = generator(features.shape[0])
                d_fake_output = discriminator(g_output)

                gen_loss = criterion(d_fake_output, real_labels)

                discriminator_optimizer.zero_grad()
                generator_optimizer.zero_grad()

                gen_loss.backward(retain_graph=True)
                generator_optimizer.step()

                avg_gen_loss += gen_loss.item()

                pbar.set_postfix({'gen_loss': avg_gen_loss/(i+1), 'disc_loss' :avg_disc_loss/(i+1)})
                pbar.update(1)

            avg_gen_loss/=(i+1)
            avg_disc_loss/=(i+1)

            gen_losses.append(avg_gen_loss)
            disc_losses.append(avg_disc_loss)

            if epoch%10==0 or epoch==1:
                print(f'saving a checkpoint at epoch {epoch}')
                torch.save({'state_dict': generator.state_dict()}, os.path.join(save_dir,'generator.pth'))
                torch.save({'state_dict': discriminator.state_dict()}, os.path.join(save_dir,'discriminator.pth'))
            
                generator.eval()
                generator.generate_img(os.path.join(save_dir,f'generated_imgs_epoch{epoch}.jpg'))

    plt.plot(gen_losses, c='r', label='generator')
    plt.plot(disc_losses, c='b', label='discriminator')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'loss_graph.jpg'))
    