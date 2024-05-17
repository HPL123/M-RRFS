import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class Embedding_Net_vis(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net_vis, self).__init__()

        self.fc1 = nn.Linear(opt.resSize_vis, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding = self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return out_z


# class Embedding_Net_sem(nn.Module):
#     def __init__(self, opt):
#         super(Embedding_Net_sem, self).__init__()
#
#         self.fc1 = nn.Linear(opt.resSize_sem, opt.embedSize)
#         self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
#         self.lrelu = nn.LeakyReLU(0.2, True)
#         self.relu = nn.ReLU(True)
#         self.apply(weights_init)
#
#     def forward(self, features):
#         embedding = self.relu(self.fc1(features))
#         out_z = F.normalize(self.fc2(embedding), dim=1)
#         return out_z

##
class Embedding_Net_sem(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net_sem, self).__init__()

        self.fc1 = nn.Linear(opt.resSize_sem, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        # embedding = self.relu(self.fc1(features))
        # out_z = F.normalize(self.fc2(embedding), dim=1)
        embedding = self.relu(self.fc1(features))
        out_z = F.normalize(embedding, dim=1)
        return out_z



