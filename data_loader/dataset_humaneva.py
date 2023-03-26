"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset_humaneva.py
"""


import numpy as np
import os
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


class DatasetHumanEva(Dataset):

    def __init__(self, mode, t_his=15, t_pred=60, actions='all', **kwargs):
        super().__init__(mode, t_his, t_pred, actions)

    def prepare_data(self):
        self.data_file = os.path.join('data', 'data_3d_humaneva15.npz')
        self.subjects_split = {'train': ['Train/S1', 'Train/S2', 'Train/S3'],
                               'test': ['Validate/S1', 'Validate/S2', 'Validate/S3']}
        self.subjects = [x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
                                 joints_left=[2, 3, 4, 8, 9, 10],
                                 joints_right=[5, 6, 7, 11, 12, 13])
        self.kept_joints = np.arange(15)
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        # these takes have wrong head position, excluded from training and testing
        if self.mode == 'train':
            data_f['Train/S3'].pop('Walking 1 chunk0')
            data_f['Train/S3'].pop('Walking 1 chunk2')
        else:
            data_f['Validate/S3'].pop('Walking 1 chunk4')
        for key in list(data_f.keys()):
            data_f[key] = dict(filter(lambda x: (self.actions == 'all' or
                                                 all([a in x[0] for a in self.actions]))
                                                 and x[1].shape[0] >= self.t_total, data_f[key].items()))
            if len(data_f[key]) == 0:
                data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                seq[:, 1:] -= seq[:, :1]
                data_s[action] = seq
        self.data = data_f


if __name__ == '__main__':
    np.random.seed(0)
    actions = 'all'
    dataset = DatasetHumanEva('test', actions=actions)
    generator = dataset.sampling_generator()
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)


