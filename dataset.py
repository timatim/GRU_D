# TODO: implement data loader
# preliminary idea: recordID as key, matrix as value

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import utils


class PhysioNET(Dataset):
    def __init__(self, data, outcomes, num_features=33, means=None, stds=None):
        self.outcomes = outcomes
        self.x = {}
        self.delta = {}
        self.m = {}
        self.x_obs = {}
        self.active_indices = list(range(len(outcomes)))

        # get only data in records
        self.data = data.loc[outcomes['RecordID']]

        # if not given, compute empirical mean and std
        self.means = means
        if self.means is None:
            self.means = self.data.loc[outcomes.iloc[self.active_indices]['RecordID']].iloc[:, 1:].mean().values
            # self.means = self.data.iloc[self.active_indices, 1:].mean().values
        self.stds = stds
        if self.stds is None:
            self.stds = self.data.loc[outcomes.iloc[self.active_indices]['RecordID']].iloc[:, 1:].std().values
            # self.stds = self.data.iloc[self.active_indices, 1:].std().values

        for i, record_id in enumerate(outcomes['RecordID']):
            partial_df = self.data.loc[[record_id]]
            # get x
            self.x[i] = partial_df.drop(['time'], axis=1).as_matrix()
            # self.x[i] = (self.x[i] - self.means) / self.stds

            # get m
            m = np.isnan(self.x[i]).astype(int)
            self.m[i] = m

            # get delta
            delta = np.zeros(m.shape)
            timestamps = np.array([utils.timestamp2minute(s) for s in partial_df['time']])
            x_obs = np.zeros(m.shape)
            last_obs = [None] * num_features

            for d in range(num_features):
                for t in range(len(m)):
                    # compute delta
                    if t == 0:
                        delta[t][d] = 0
                    elif t > 0 and m[t - 1][d] == 0:
                        delta[t][d] = timestamps[t] - timestamps[t - 1] + delta[t - 1][d]
                    elif t > 0 and m[t - 1][d] == 1:
                        delta[t][d] = timestamps[t] - timestamps[t - 1]

                    # compute carry-forward, if non-existent impute with mean
                    # if not missing, update last obs
                    if m[t][d] == 0:
                        last_obs[d] = self.x[i][t][d]
                    # if missing, fill with last obs or mean
                    elif m[t][d] == 1:
                        x_obs[t][d] = last_obs[d] or self.means[d]

            self.delta[i] = delta

            # normalize
            self.x[i] = (self.x[i] - self.means) / (self.stds + 1e-3)
            self.x_obs[i] = (x_obs - self.means) / (self.stds + 1e-3)

        self.labels = outcomes['In-hospital_death'].as_matrix()

    def __len__(self):
        return len(self.active_indices)

    def set_active_indices(self, new_indices, new_means=None, new_stds=None):
        self.active_indices = new_indices

        # recompute mean, std
        if new_means is None:
            new_means = self.data.loc[self.outcomes.iloc[self.active_indices]['RecordID']].iloc[:, 1:].mean().values
        if new_stds is None:
            new_stds = self.data.loc[self.outcomes.iloc[self.active_indices]['RecordID']].iloc[:, 1:].std().values

        # re normalize
        for i in new_indices:
            self.x[i] = (((self.x[i] * self.stds) + self.means) - new_means) / (new_stds + 1e-3)
            self.x_obs[i] = (((self.x_obs[i] * self.stds) + self.means) - new_means) / (new_stds + 1e-3)

        self.means = new_means
        self.stds = new_stds

        return None

    def __getitem__(self, key):
        """
        
        :param key: 
        :return: x, delta, m
        """
        real_key = self.active_indices[key]
        return (self.x[real_key], self.delta[real_key], self.m[real_key], self.x_obs[real_key]), self.labels[real_key]

    # TODO: implement batch collate by padding with 0s
    def collate_batch(batch):
        # find maximum sequence length
        max_seq_len = max([len(b[0][0]) for b in batch])
        num_features = batch[0][0][0].shape[1]

        matrices = [list() for _ in range(4)]

        lengths = torch.LongTensor(np.array([len(b[0][0]) for b in batch]))

        # pad to max_seq_len in batch with zeros
        for b in batch:
            # get length of sample
            sample_len = len(b[0][0])
            # pad the three matrices
            for i in range(4):
                matrices[i].append(torch.cat([
                                     torch.FloatTensor(b[0][i]),
                                     torch.FloatTensor(max_seq_len-sample_len, num_features).zero_()
                                    ]))
        packed_matrices = []

        for i in range(4):
            packed_matrices.append(torch.stack(matrices[i]))
        packed_matrices.append(lengths)

        return packed_matrices, torch.LongTensor([int(b[1]) for b in batch])

if __name__ == "__main__":
    df = pd.read_csv('./all_data.csv')
    outcomes = pd.read_csv('./set-a/Outcomes-a.txt')

    dataset = PhysioNET(df, outcomes)
