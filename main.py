import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import model
import dataset
import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from time import time

data_dir = './set-a'
feature_means = pd.read_csv('./feature_means.csv', names=['param', 'mean'])
NUM_FEATURES = len(feature_means)

outcomes = pd.read_csv(os.path.join(data_dir, 'Outcomes-a.txt'))
train_outcomes, test_outcomes = train_test_split(outcomes, test_size=0.1)

# load data
df = pd.read_csv('./all_data.csv')

train_physionet = dataset.PhysioNET(df, train_outcomes)
train_loader = torch.utils.data.DataLoader(dataset=train_physionet,
                                           batch_size=16,
                                           collate_fn=train_physionet.collate_batch,
                                           shuffle=True)
test_physionet = dataset.PhysioNET(df, test_outcomes, means=train_physionet.means, stds=train_physionet.stds)
test_loader = torch.utils.data.DataLoader(dataset=test_physionet,
                                          batch_size=16,
                                          collate_fn=test_physionet.collate_batch,
                                          shuffle=True)


def test_model(loader, model):
    correct = 0
    total = 0
    model.eval()
    decoder.eval()

    predictions = []
    truths = []

    for data, label in loader:
        x, delta, m = data
        x = Variable(x.float())
        delta = Variable(delta.float())
        m = Variable(m.float())

        # output, _ = model(x, delta, m)
        x[m.byte()] = 0.
        _, hidden = model(x)
        output = decoder(hidden.squeeze())

        predicted = (output.max(1)[1].data.long()).view(-1)
        predictions += list(predicted.numpy())
        truths += list(label.numpy())
        total += label.size(0)
        correct += (predicted == label).sum()

    model.train()
    decoder.train()
    return 100 * correct / total, roc_auc_score(truths, predictions)


# GRU_D = model.GRUD(NUM_FEATURES, 49, 2, feature_means=feature_means['mean'].values)
GRU_D = nn.GRU(NUM_FEATURES, 49, batch_first=True, dropout=0.3)
decoder = nn.Sequential(
                         nn.BatchNorm1d(49),
                         nn.Dropout(0.3),
                         nn.Linear(49, 2),
                         nn.LogSoftmax()
                       )
optimizer = torch.optim.Adam(list(GRU_D.parameters()) + list(decoder.parameters()), lr=1e-3)
loss_func = torch.nn.NLLLoss()

num_epochs = 20
log_interval = int(len(train_loader) / 3)

for epoch in range(num_epochs):
    start = time()
    total_loss = 0.

    for i, (data, label) in enumerate(train_loader):
        x, delta, m = data
        x = Variable(x.float())
        delta = Variable(delta.float())
        m = Variable(m.float())

        optimizer.zero_grad()
        # output, _ = GRU_D(x, delta, m)
        x[m.byte()] = 0.
        _, hidden = GRU_D(x)
        output = decoder(hidden.squeeze())

        loss = loss_func(output, Variable(label))
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]

        # report performance
        if (i + 1) % log_interval == 0:
            val_acc, test_auc = test_model(test_loader, GRU_D)
            print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Validation Acc:{5}, AUC:{6}'.format(
                epoch + 1, num_epochs, i + 1, len(train_loader), total_loss/(i+1), val_acc, test_auc))
    print("Epoch %d time: %.4f" % (epoch, time()-start))
