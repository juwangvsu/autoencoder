import torch
from torch.utils.data import DataLoader
import dataset_office31
import matplotlib.pyplot as plt
train_dataset_office = dataset_office31.Office31(root='/media/student/isaacsim/office31', train=True, download=True, domain='amazon')
test_dataloader = DataLoader(train_dataset_office, batch_size=4, shuffle=False)
train_features, train_labels = next(iter(test_dataloader))
plt.imshow(train_features[0].cpu().numpy().transpose(1,2,0))
plt.show()
