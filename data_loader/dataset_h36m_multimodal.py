"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset_h36m_multimodal.py
"""

import numpy as np
import os
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton
from utils import util


class DatasetH36M_multi(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False, **kwargs):
        self.use_vel = use_vel
        if 'multimodal_path' in kwargs.keys():
            self.multimodal_path = kwargs['multimodal_path']
        else:
            self.multimodal_path = None

        if 'data_candi_path' in kwargs.keys():
            self.data_candi_path = kwargs['data_candi_path']
        else:
            self.data_candi_path = None
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.data_file = os.path.join('data', 'data_3d_h36m.npz')
        self.subjects_split = {'train': [1, 5, 6, 7, 8],
                               'test': [9, 11]}
        self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
        self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()
        self.S1_skeleton = data_o['S1']['Directions'][:1, self.kept_joints].copy()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if self.actions != 'all':
            for key in list(data_f.keys()):
                data_f[key] = dict(
                    filter(lambda x: all([a in str.lower(x[0]) for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        # possible candidate
        # skip_rate = 10
        # data_candi = []
        if self.multimodal_path is None:
            self.data_multimodal = \
                np.load('./data/data_multi_modal/t_his25_1_thre0.050_t_pred100_thre0.100_filtered.npz',
                        allow_pickle=True)[
                    'data_multimodal'].item()
            data_candi = \
                np.load('./data/data_multi_modal/data_candi_t_his25_t_pred100_skiprate20.npz', allow_pickle=True)[
                    'data_candidate.npy']
        else:
            self.data_multimodal = np.load(self.multimodal_path, allow_pickle=True)['data_multimodal'].item()
            data_candi = np.load(self.data_candi_path, allow_pickle=True)['data_candidate.npy']

        self.data_candi = {}

        for sub in data_f.keys():
            data_s = data_f[sub]
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                seq[:, 1:] -= seq[:, :1]

                # # get relative candidate
                # data_tmp = np.copy(seq)
                # data_tmp[:, 0] = 0
                # nf = data_tmp.shape[0]
                # idxs = np.arange(0, nf - self.t_his - self.t_pred, skip_rate)[:, None] + np.arange(
                #     self.t_his + self.t_pred)[None, :]
                # data_tmp = data_tmp[idxs]
                # data_tmp = util.absolute2relative(data_tmp, parents=self.skeleton.parents())
                # data_candi.append(data_tmp)

                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)
                data_s[action] = seq

                if sub not in self.data_candi.keys():
                    x0 = np.copy(seq[None, :1, ...])
                    x0[:, :, 0] = 0
                    self.data_candi[sub] = util.absolute2relative(data_candi, parents=self.skeleton.parents(),
                                                                  invert=True, x0=x0)

        self.data = data_f
        # self.data_candi = np.concatenate(data_candi, axis=0)

    def sample(self, n_modality=5):
        subject = np.random.choice(self.subjects)
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]
        if n_modality > 0 and subject in self.data_multimodal.keys():
            # margin_f = 1
            # thre_his = 0.05
            # thre_pred = 0.1
            # x0 = np.copy(traj[None, ...])
            # x0[:, :, 0] = 0
            # # candi_tmp = util.absolute2relative(self.data_candi, parents=self.skeleton.parents(), invert=True, x0=x0)
            candi_tmp = self.data_candi[subject]
            # # observation distance
            # dist_his = np.mean(np.linalg.norm(x0[:, self.t_his - margin_f:self.t_his, 1:] -
            #                                   candi_tmp[:, self.t_his - margin_f:self.t_his, 1:], axis=3), axis=(1, 2))
            # idx_his = np.where(dist_his <= thre_his)[0]
            #
            # # future distance
            # dist_pred = np.mean(np.linalg.norm(x0[:, self.t_his:, 1:] -
            #                                    candi_tmp[idx_his, self.t_his:, 1:], axis=3), axis=(1, 2))
            #
            # idx_pred = np.where(dist_pred >= thre_pred)[0]
            # # idxs = np.intersect1d(idx_his, idx_pred)
            idx_multi = self.data_multimodal[subject][action][fr_start]
            traj_multi = candi_tmp[idx_multi]

            # # confirm if it is the right one
            # if len(idx_multi) > 0:
            #     margin_f = 1
            #     thre_his = 0.05
            #     thre_pred = 0.1
            #     x0 = np.copy(traj[None, ...])
            #     x0[:, :, 0] = 0
            #     dist_his = np.mean(np.linalg.norm(x0[:, self.t_his - margin_f:self.t_his, 1:] -
            #                                       traj_multi[:, self.t_his - margin_f:self.t_his, 1:], axis=3),
            #                        axis=(1, 2))
            #     if np.any(dist_his > thre_his):
            #         print(f'===> wrong multi modality sequneces {dist_his[dist_his > thre_his].max():.3f}')

            if len(traj_multi) > 0:
                traj_multi[:, :self.t_his] = traj[None, ...][:, :self.t_his]
                if traj_multi.shape[0] > n_modality:
                    st0 = np.random.get_state()
                    idxtmp = np.random.choice(np.arange(traj_multi.shape[0]), n_modality, replace=False)
                    traj_multi = traj_multi[idxtmp]
                    np.random.set_state(st0)
                    # traj_multi = traj_multi[:n_modality]
            traj_multi = np.concatenate(
                [traj_multi, np.zeros_like(traj[None, ...][[0] * (n_modality - traj_multi.shape[0])])], axis=0)

            return traj[None, ...], traj_multi
        else:
            return traj[None, ...], None

    def sampling_generator(self, num_samples=1000, batch_size=8, n_modality=5):
        for i in range(num_samples // batch_size):
            sample = []
            sample_multi = []
            for i in range(batch_size):
                sample_i, sample_multi_i = self.sample(n_modality=n_modality)
                sample.append(sample_i)
                sample_multi.append(sample_multi_i[None, ...])
            sample = np.concatenate(sample, axis=0)
            sample_multi = np.concatenate(sample_multi, axis=0)
            yield sample, sample_multi

    def iter_generator(self, step=25, n_modality=10):
        for sub in self.data.keys():
            data_s = self.data[sub]
            candi_tmp = self.data_candi[sub]
            for act in data_s.keys():
                seq = data_s[act]
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    # idx_multi = self.data_multimodal[sub][act][i]
                    # traj_multi = candi_tmp[idx_multi]
                    traj = seq[None, i: i + self.t_total]
                    if n_modality > 0:
                        margin_f = 1
                        thre_his = 0.05
                        thre_pred = 0.1
                        x0 = np.copy(traj)
                        x0[:, :, 0] = 0
                        # candi_tmp = util.absolute2relative(self.data_candi, parents=self.skeleton.parents(), invert=True, x0=x0)
                        # candi_tmp = self.data_candi[subject]
                        # observation distance
                        dist_his = np.mean(np.linalg.norm(x0[:, self.t_his - margin_f:self.t_his, 1:] -
                                                          candi_tmp[:, self.t_his - margin_f:self.t_his, 1:], axis=3),
                                           axis=(1, 2))
                        idx_his = np.where(dist_his <= thre_his)[0]

                        # future distance
                        dist_pred = np.mean(np.linalg.norm(x0[:, self.t_his:, 1:] -
                                                           candi_tmp[idx_his, self.t_his:, 1:], axis=3), axis=(1, 2))

                        idx_pred = np.where(dist_pred >= thre_pred)[0]
                        # idxs = np.intersect1d(idx_his, idx_pred)
                        traj_multi = candi_tmp[idx_his[idx_pred]]
                        if len(traj_multi) > 0:
                            traj_multi[:, :self.t_his] = traj[:, :self.t_his]
                            if traj_multi.shape[0] > n_modality:
                                # idxtmp = np.random.choice(np.arange(traj_multi.shape[0]), n_modality, replace=False)
                                # traj_multi = traj_multi[idxtmp]
                                traj_multi = traj_multi[:n_modality]
                        traj_multi = np.concatenate(
                            [traj_multi, np.zeros_like(traj[[0] * (n_modality - traj_multi.shape[0])])],
                            axis=0)
                    else:
                        traj_multi = None

                    yield traj, traj_multi


if __name__ == '__main__':
    np.random.seed(0)
    actions = {'WalkDog'}
    dataset = DatasetH36M('train', actions=actions)
    generator = dataset.sampling_generator()
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)
