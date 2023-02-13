import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """
        Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class UnsupervisedPredictor(nn.Module):
    def __init__(self, input_size, out_features):
        super().__init__()
        self.linear = nn.Linear(input_size, out_features)
        self.opt = Adam(self.linear.parameters(), lr=0.1)

    def forward(self, hidden_reprs):
        logits = self.linear(hidden_reprs)
        logits = F.softmax(logits, dim=-1)
        return logits

class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        linear_dim = 0
        for d in range(len(dims) - 1):
            if d > 0:
                linear_dim += dims[d]
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]
        self.linear_classifier = UnsupervisedPredictor(input_size=linear_dim, out_features=10).to("cuda")

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def predict_unsupervised(self, x):
        goodness = []
        h = x
        for c, layer in enumerate(self.layers):
            h = layer(h)
            if c > 0:
                goodness += [h / torch.norm(h)]
        representation_vectors = torch.cat(goodness, dim=-1)
        probs = self.linear_classifier(representation_vectors)
        max_prob, max_class = probs.max(dim=-1)
        return max_class

    def train(self, x_pos, x_neg, y = None):
        if y != None:
            one_hot_targets = torch.zeros(len(y), 10)
            one_hot_targets[torch.arange(len(y)), y] = 1
            one_hot_targets = one_hot_targets.to("cuda")
            h_pos, h_neg = x_pos, x_neg
            goodness = []
            criterion = nn.CrossEntropyLoss()
            for i, layer in enumerate(self.layers):
                print('training layer', i, '...')
                h_pos, h_neg = layer.train(h_pos, h_neg)
                if i > 0:
                    goodness += [h_pos / torch.norm(h_pos)]
            concat_goodness = torch.cat(goodness, 1)
            for i in tqdm(range(500)):
                linear_transform = self.linear_classifier.forward(concat_goodness)
                self.linear_classifier.opt.zero_grad()
                
                loss = criterion(linear_transform, one_hot_targets)
                loss.backward()
                self.linear_classifier.opt.step()
        else:
            h_pos, h_neg = x_pos, x_neg
            for i, layer in enumerate(self.layers):
                print('training layer', i, '...')
                h_pos, h_neg = layer.train(h_pos, h_neg)

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just computes the derivative of a single layer and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
def generate_negative_samples(x, bitmask):
    tmp = x.clone()
    tmp2 = x.clone()
    rnd = torch.randperm(x.size(0))
    bitmask = torch.flatten(bitmask)
    return tmp * bitmask + tmp2[rnd] * ~bitmask

def create_bitmask(size):
    bits = torch.cuda.FloatTensor(size, size).uniform_()
    bits = torch.unsqueeze(bits, 0)
    filter = torch.cuda.FloatTensor([[1/16, 1/8 , 1/16], [1/8, 1/4 , 1/8], [1/16, 1/8 , 1/16]])
    f = filter.expand(1,1,3,3)
    for i in range(5):
        bits = F.conv2d(bits, f, padding='same')
    return (bits.squeeze(0) > 0.5)

def supervised(x, y, vis=False):
    net = Net([784, 64, 32])
    x_pos = overlay_y_on_x(x, y)
    y_neg = y.clone()
    for idx, y_samp in enumerate(y):
        allowed_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        allowed_indices.pop(y_samp.item())
        y_neg[idx] = torch.tensor(np.random.choice(allowed_indices)).cuda()
    x_neg = overlay_y_on_x(x, y_neg)
    if vis:
        for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
            visualize_sample(data, name)
    net.train(x_pos, x_neg)
    return net

def unsupervised(x, y, vis=False):
    net2 = Net([784, 64, 64, 64, 64])
    x_pos_unsup = x.clone()
    bitmask = create_bitmask(28)
    x_neg_unsup = generate_negative_samples(x, bitmask)
    if vis:
        for data, name in zip([x, x_pos_unsup, x_neg_unsup], ['orig', 'pos', 'neg']):
            visualize_sample(data, name)
    net2.train(x_pos_unsup, x_neg_unsup, y)
    return net2

if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()

    net = supervised(x, y, vis=False)
    net2 = unsupervised(x, y, vis=False)

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()
    print('Supervised test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
    print('Unsupervised test error:', 1.0 - net2.predict_unsupervised(x_te).eq(y_te).float().mean().item())
