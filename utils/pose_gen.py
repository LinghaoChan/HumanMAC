from torch import tensor
from utils import *
from utils.script import sample_preprocessing


def pose_generator(data_set, model_select, diffusion, cfg, mode=None,
                   action=None, nrow=1):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    traj_np = None
    j = None
    while True:
        poses = {}
        draw_order_indicator = -1
        for k in range(0, nrow):
            if mode == 'switch':
                data = data_set.sample_all_action()
            elif mode == 'pred':
                data = data_set.sample_iter_action(action, cfg.dataset)
            elif mode == 'gif' or 'fix' in mode:
                data = data_set.sample()
            elif mode == 'zero_shot':
                data = data_set[np.random.randint(0, data_set.shape[0])].copy()
                data = np.expand_dims(data, axis=0)
            else:
                raise NotImplementedError(f"unknown pose generator mode: {mode}")

            # gt
            gt = data[0].copy()
            gt[:, :1, :] = 0
            data[:, :, :1, :] = 0

            if mode == 'switch':
                poses = {}
                traj_np = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred, -1])
            elif mode == 'pred' or mode == 'gif' or 'fix' in mode or mode == 'zero_shot':
                if draw_order_indicator == -1:
                    poses['context'] = gt
                    poses['gt'] = gt
                else:
                    poses[f'HumanMAC_{draw_order_indicator + 1}'] = gt
                    poses[f'HumanMAC_{draw_order_indicator + 2}'] = gt
                gt = np.expand_dims(gt, axis=0)
                traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])

            traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)

            mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, cfg, mode=mode)
            sampled_motion = diffusion.sample_ddim(model_select,
                                                   traj_dct,
                                                   traj_dct_mod,
                                                   mode_dict)

            traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            traj_est = traj_est.cpu().numpy()
            traj_est = post_process(traj_est, cfg)

            if k == 0:
                for j in range(traj_est.shape[0]):
                    poses[f'HumanMAC_{j}'] = traj_est[j]
            else:
                for j in range(traj_est.shape[0]):
                    poses[f'HumanMAC_{j + draw_order_indicator + 2 + 1}'] = traj_est[j]

            if draw_order_indicator == -1:
                draw_order_indicator = j
            else:
                draw_order_indicator = j + draw_order_indicator + 2 + 1

        yield poses
