import os
import numpy as np
from utils.pose_gen import pose_generator
from utils.visualization import render_animation


def demo_visualize(mode, cfg, model, diffusion, dataset):
    """
    script for drawing gifs in different modes
    """
    if cfg.dataset != 'h36m' and mode != 'pred':
        raise NotImplementedError(f"sorry, {mode} is currently only available in h36m setting.")
    if mode == 'switch':
        for i in range(0, cfg.vis_switch_num):
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg, mode='switch')
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col,
                             output=os.path.join(cfg.gif_dir, f'switch_{i}.gif'), mode=mode)

    elif mode == 'pred':
        action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        for i in range(0, len(action_list)):
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode='pred', action=action_list[i], nrow=cfg.vis_row)
            suffix = action_list[i]
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, f'pred_{suffix}.gif'), mode=mode)

    elif mode == 'control':
        # draw part-body controllable results
        fix_name = ['right_leg', 'left_leg', 'torso', 'left_arm', 'right_arm', 'fix_lower', 'fix_upper']
        for i in range(0, 7):
            mode_fix = 'fix' + '_' + str(i)
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode=mode_fix, nrow=cfg.vis_row)
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, fix_name[i] + '.gif'), mode=mode, fix_index=i)
    elif mode == 'zero_shot':
        amass_data = np.squeeze(np.load('./data/amass_retargeted.npy'))
        for i in range(0, 15):
            pose_gen = pose_generator(amass_data, model, diffusion, cfg, mode=mode, nrow=cfg.vis_row)
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, f'zero_shot_{str(i)}.gif'), mode=mode)
    else:
        raise
