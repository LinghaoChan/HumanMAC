"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset.py
"""


import numpy as np

class Dataset:

    def __init__(self, mode, t_his, t_pred, actions='all'):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.prepare_data()
        self.std, self.mean = None, None
        self.data_len = sum([seq.shape[0] for data_s in self.data.values() for seq in data_s.values()])
        self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        self.normalized = False
        # iterator specific
        self.sample_ind = None

    def prepare_data(self):
        raise NotImplementedError

    def normalize_data(self, mean=None, std=None):
        if mean is None:
            all_seq = []
            for data_s in self.data.values():
                for seq in data_s.values():
                    all_seq.append(seq[:, 1:])
            all_seq = np.concatenate(all_seq)
            self.mean = all_seq.mean(axis=0)
            self.std = all_seq.std(axis=0)
        else:
            self.mean = mean
            self.std = std
        for data_s in self.data.values():
            for action in data_s.keys():
                data_s[action][:, 1:] = (data_s[action][:, 1:] - self.mean) / self.std
        self.normalized = True

    def sample(self):
        subject = np.random.choice(self.subjects)
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        # seq = dict_s['WalkDog']
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]
        return traj[None, ...]
    
    def sample_all_action(self):
        # subject = np.random.choice(self.subjects)
        dict_s = self.data['S9']

        action_list = []
        sample = []

        for i in range(0, len(list(dict_s.keys()))):
            type = list(dict_s.keys())[i].split(' ')[0]
            if type == 'Discussion':
                type = 'Discussion 1'
            action_list.append(type)

        action_list = list(set(action_list))
        
        for i in range(0, len(action_list)):
            action = action_list[i]
            seq = dict_s[action]
            fr_start = np.random.randint(seq.shape[0] - self.t_total)
            fr_end = fr_start + self.t_total
            traj = seq[fr_start: fr_end]
            sample.append(traj[None, ...])
            
        # 15 -> 30
        # for i in range(0, len(action_list)):
        #     action = action_list[i]
        #     seq = dict_s[action]
        #     fr_start = np.random.randint(seq.shape[0] - self.t_total)
        #     fr_end = fr_start + self.t_total
        #     traj = seq[fr_start: fr_end]
        #     sample.append(traj[None, ...])

        sample = np.concatenate(sample, axis=0)
        return sample
    
    def sample_iter_action(self, action_category, dataset_type):
        # subject = np.random.choice(self.subjects)
        if dataset_type == 'h36m':
            dict_s = self.data['S9']
        elif dataset_type == 'humaneva':
            dict_s = self.data['Validate/S2']
        else:
            raise
        # dict_s = self.data['S9']
        sample = []
        
        action = action_category
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]
        sample.append(traj[None, ...])

        sample = np.concatenate(sample, axis=0)
        return sample
    
    def prepare_iter_action(self, dataset_type):
        # subject = np.random.choice(self.subjects)
        if dataset_type == 'h36m':
            dict_s = self.data['S9']
        elif dataset_type == 'humaneva':
            dict_s = self.data['Validate/S2']
        else:
            raise
        # dict_s = self.data['S9']

        action_list = []
        sample = []

        for i in range(0, len(list(dict_s.keys()))):
            # type = list(dict_s.keys())[i].split(' ')[0]
            type = list(dict_s.keys())[i]
            if type == 'Discussion':
                type = 'Discussion 1'
            action_list.append(type)

        action_list = list(set(action_list))
        return action_list

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        for i in range(num_samples // batch_size):
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if aug is True:
                if np.random.uniform() > 0.5:  # x-y rotating
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotate_xy = np.matmul(sample.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
                    sample[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
                    del theta, rotate_matrix, rotate_xy
                if np.random.uniform() > 0.5:  # x-z mirroring
                    sample[..., 0] = - sample[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    sample[..., 1] = - sample[..., 1]
            yield sample

    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj



