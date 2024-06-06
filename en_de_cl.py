import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(            
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x # shape b, 16 x imwidth/2 x imwidth/2

#input should be the output of the encoder, encoder created outside and passed in as a reference
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 32 * 32, out_features=100), #encoder output 16x32x32 for amazon, resolution 64
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=31) #office31 have 31 labels
        )

    def forward(self, x):
        x = self.classifier(x)
        #print('classifier output ', x[0])
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.decoder(x)
        return x


# Define the autoencoder architecture using Encoder and Decoder
class AutoencoderNet(nn.Module):
    def __init__(self,encoder, decoder):
        super(AutoencoderNet, self).__init__()
        self.en=encoder
        self.de=decoder
        self.debug=False
    def forward(self, x):
        if self.debug:
            print('encoder inp shape ', x.shape)
        x = self.en(x)
        if self.debug:
            print('encoder output shape ', x.shape)
        x = self.de(x)
        return x

    #save to two files, for en and de, fn is the prefix
    def save_model(self, fn):
        fne='encoder_'+fn+'.pth'
        fnd='decoder_'+fn+'.pth'
        fnwhole='autoencodernet'+fn+'.pth'
        torch.save(self.en.state_dict(), fne)# 'conv_autoencoder.pth')
        torch.save(self.de.state_dict(), fnd)# 'conv_autoencoder.pth')
        torch.save(self.state_dict(), fnwhole)# 'conv_autoencoder.pth')

# Define the classifier net architecture using Encoder and Classifier 
class ClassifyNet(nn.Module):
    def __init__(self,encoder, classifier):
        super(ClassifyNet, self).__init__()
        self.en=encoder
        self.cl=classifier
        self.debug=False
    def forward(self, x):
        if self.debug:
            print('CL encoder inp shape ', x.shape)
        x = self.en(x).reshape(-1,16384)
        if self.debug:
            print('CL encoder output shape ', x.shape)
        x = self.cl(x)
        return x
    #save to two files, for en and de, fn is the prefix
    def save_model(self, fn):
        fne='encoder_'+fn+'.pth'
        fnd='classifier'+fn+'.pth'
        fnwhole='classifiernet'+fn+'.pth'
        torch.save(self.en.state_dict(), fne)# 'conv_autoencoder.pth')
        torch.save(self.cl.state_dict(), fnd)# 'conv_autoencoder.pth')
        torch.save(self.state_dict(), fnwhole)# 'conv_autoencoder.pth')

