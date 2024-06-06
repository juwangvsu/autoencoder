import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import dataset_office31
import numpy as np
from en_de_cl import AutoencoderNet, Encoder, Decoder, ClassifyNet, Classifier

import argparse
model=None
class AutoencoderFC(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2):
        super(AutoencoderFC, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        # decoder part
        self.fc3 = nn.Linear(h_dim2, h_dim1)
        self.fc4 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        #print('autoencoderfc x.shape ', x.shape)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def decoder(self, x):
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def save_model(self, fn):
        torch.save(model.state_dict(), fn)# 'conv_autoencoder.pth')

class AutoencoderFC2(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2):
        super(AutoencoderFC2, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        # decoder part
        self.fc3 = nn.Linear(h_dim2, h_dim1)
        self.fc4 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        #print('autoencoderfc x.shape ', x.shape)
        x=torch.nn.functional.relu(self.fc1(x))
        #x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def decoder(self, x):
        x=torch.nn.functional.relu(self.fc3(x))
        #x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def save_model(self, fn):
        torch.save(model.state_dict(), fn)# 'conv_autoencoder.pth')

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
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
        self.debug=False

    def forward(self, x):
        if self.debug:
            print('encoder inp shape ', x.shape)
        x = self.encoder(x)
        if self.debug:
            print('encoder output shape ', x.shape)
        x = self.decoder(x)
        return x
    def save_model(self, fn):
        torch.save(model.state_dict(), fn)# 'conv_autoencoder.pth')

# Network Parameters
num_hidden_1 = 1024 #256  # 1st layer num features
num_hidden_2 = 512 #128  # 2nd layer num features (the latent dim)
#num_input = 4096  # flower data input (img shape: 64*64 )
#num_input = 784  # MNIST data input (img shape: 28*28)

def create_model(args):
    global model, device
    num_input=int(args.imwidth) * int(args.imwidth)
# Initialize the autoencoder
    if args.arch=='Conv':
        model = Autoencoder()
    elif args.arch=='FC':
        model = AutoencoderFC2(num_input, num_hidden_1, num_hidden_2)
    elif args.arch=='AE2':
        encoder = Encoder()
        decoder = Decoder()
        model = AutoencoderNet(encoder, decoder)
    elif args.arch=='CL':
        encoder = Encoder()
        classifier = Classifier()
        model = ClassifyNet(encoder, classifier)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def create_dataloader(args):
    batchSize=128
# Load dataset office31 amazon
    office31root = args.dsroot #/media/student/isaacsim/office32
    train_dataset_am = dataset_office31.Office31(root=office31root, train=True, download=True, domain='amazon', transform=transform)
    test_dataset_am = dataset_office31.Office31(root=office31root, train=False, download=True, domain='amazon', transform=transform)
    train_loader_am = torch.utils.data.DataLoader(dataset=train_dataset_am, 
                                        batch_size=128, 
                                        shuffle=True)
    test_loader_am = torch.utils.data.DataLoader(dataset=test_dataset_am, 
                                        batch_size=128, shuffle=True)

# Load dataseta flowers
    train_dataset_fl = datasets.Flowers102(root='flowers', 
                                    split='train', 
                                    transform=transform, 
                                    download=True)
    test_dataset_fl = datasets.Flowers102(root='flowers', 
                                split='test', 
                                transform=transform)
    train_loader_fl = torch.utils.data.DataLoader(dataset=train_dataset_fl, 
                                        batch_size=128, 
                                        shuffle=True)
    test_loader_fl = torch.utils.data.DataLoader(dataset=test_dataset_fl, 
                                        batch_size=128, shuffle=True)
# Load dataseta mnist 
    train_dataset_mn = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset_mn = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader_mn = torch.utils.data.DataLoader(dataset=train_dataset_mn, batch_size=batchSize, shuffle=True)
    test_loader_mn = torch.utils.data.DataLoader(dataset=test_dataset_mn, batch_size=batchSize, shuffle=True)

    print(train_dataset_mn)
    if args.dataset=='flower':
        train_loader=train_loader_fl
        train_dataset = train_dataset_fl
        test_loader=test_loader_fl
        test_dataset = test_dataset_fl
    elif args.dataset=='amazon':
        train_loader= train_loader_am
        train_dataset = train_dataset_am
        test_loader=test_loader_am
        test_dataset = test_dataset_am
    else:
        train_loader, train_loader_mn
        train_dataset = train_dataset_mn
        test_loader=test_loader_mn
        test_dataset = test_dataset_mn

    return train_loader, test_loader, train_dataset, test_dataset


def train(args, train_loader):
    global model
# Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=12, factor=0.5, min_lr=1e-5
        )

# Train the autoencoder
    num_epochs = 400
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            #print('train img shape,  ', img.shape)
            if args.arch=='FC':
                img = img[:,0]
                #print('train img shape,  ', img.shape)
                img = torch.reshape(img,(-1, args.imwidth*args.imwidth))
            optimizer.zero_grad()
            #print('train img shape,  ', img.shape)
            output = model(img)
            #print('train img shape, output shape ', img.shape, output.shape)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
        if epoch % 5== 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        if (epoch %5 ==0) and (epoch >10):
            print('save model param')
            model.save_model('conv_autoencoder.pth')
            #torch.save(model.state_dict(), 'conv_autoencoder.pth')
        #lr_scheduler.step(loss)
        #print('curr lr ', optimizer.state_dict()['param_groups'][0]['lr'])

#train classifynet
def train_classifier(args, train_loader):
    global model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=12, factor=0.5, min_lr=1e-5
        )
    num_epochs = 400
    for epoch in range(num_epochs):
        for data in train_loader:
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            #print('train img shape,  ', img.shape)
            optimizer.zero_grad()
            #print('train img shape,  ', img.shape)
            output = model(img)
            #print('output and labels ', output.shape, labels.shape)
            #print('train img shape, output shape ', img.shape, output.shape)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        if epoch % 5== 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        if (epoch %5 ==0) and (epoch >10):
            print('save model param')
            model.save_model('clsnet')
            #torch.save(model.state_dict(), 'conv_autoencoder.pth')
        #lr_scheduler.step(loss)
        #print('curr lr ', optimizer.state_dict()['param_groups'][0]['lr'])

#evaluate for autoencoder
def evaluate(args, test_loader):
    global model, device
    model_path=args.paramfn
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    model.debug=True
    with torch.no_grad():
        for data, _ in test_loader:
            if args.arch=='FC' and args.cnum==3:
                data=data[:,0]
            data = data.to(device)
            if args.arch=='FC':
                    data = torch.reshape(data,(-1, args.imwidth*args.imwidth))
                    print('FC input shape ', data.shape)
            recon = model(data)
            break
    #print('sample output vs input ', data[0,0,:8,:8], recon[0,0,:8,:8])
        
    import matplotlib.pyplot as plt
    plt.figure(dpi=250)
    fig, ax = plt.subplots(2, 7, figsize=(15, 4))
    for i in range(7):
        if args.arch=='FC':
            if args.cnum==1:
                ax[0, i].imshow(data[i].cpu().numpy().reshape(args.imwidth,args.imwidth)) #imwidth mnist 28
                ax[1, i].imshow(recon[i].cpu().numpy().reshape(args.imwidth, args.imwidth))
            else:
                ax[0, i].imshow(data[i].cpu().numpy().reshape(args.imwidth,args.imwidth)) #imwidth mnist 28
                ax[1, i].imshow(recon[i].cpu().numpy().reshape(args.imwidth, args.imwidth))
        else: #Conv, assume cnum=3, only deal with 3 channel dataset for now
            ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
            ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
    plt.show()

#evaluate for classifynet
def evaluate_cl(args, test_loader, train_dataset):
    global model, device
    model_path=args.paramfn
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    model.debug=True
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            pred = model(data)
            break

    import matplotlib.pyplot as plt
    plt.figure(dpi=250)
    fig, ax = plt.subplots(2, 7, figsize=(15, 4))
    for i in range(7):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
    print(' gt label ', label) #onehot
    _, labelind = torch.max(label, 1)
    print(' gt label ', np.array(train_dataset.classnames)[labelind[:7]])
    print('classify predicted label ', pred)
    _, predicted = torch.max(pred.data, 1)
    print('classify predicted index ', predicted)
    print('classify predicted classname', np.array(train_dataset.classnames)[predicted[:7].cpu()])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main script .")
    parser.add_argument("--directory", type=str, default="/media/student/datavision/save_wipe_1-14/save_data_wipe_1-14_01", help="Directory with saved data")
    parser.add_argument("--mode", type=str, default="train", help="train, eval, train_cl, eval_cl")
    parser.add_argument("--arch", type=str, default="Conv", help="Directory with saved data")
    parser.add_argument("--imwidth", type=int, default=28, help="image width 28 for mnist, 64 for flower")
    parser.add_argument("--cnum", type=int, default=1, help="image channel numb")
    parser.add_argument("--dataset", type=str, default="flower", help="flower, mnist etc")
    parser.add_argument("--dsroot", type=str, default="/data/office31_new", help="flower, mnist etc")
    parser.add_argument('-l', '--loadweight', action='store_true', help="load weight file? use --paramfn if file name not best.pt ")
    parser.add_argument("--paramfn", type=str, default="conv_autoencoder.pth", help="Directory with saved data")

    args = parser.parse_args()
    create_model(args)
    train_loader, test_loader, train_dataset, test_dataset = create_dataloader(args) #this handle the dataset name and return the correct loader
    if args.mode=="train":
        train(args, train_loader)
    elif args.mode=="train_cl":
        train_classifier(args, train_loader)
    elif args.mode=="eval_cl":
        evaluate_cl(args, test_loader, train_dataset)
    else:
        evaluate(args, test_loader)

