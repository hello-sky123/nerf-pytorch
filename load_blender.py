import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


def trans_t(t):
    return torch.tensor([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., t],
        [0., 0., 0., 1.]], dtype=torch.float32)


def rot_phi(phi):
    return torch.tensor([
        [1., 0., 0., 0.],
        [0., np.cos(phi), -np.sin(phi), 0.],
        [0., np.sin(phi), np.cos(phi), 0.],
        [0., 0., 0., 1.]], dtype=torch.float32)


def rot_theta(th):
    return torch.tensor([
        [np.cos(th), 0., -np.sin(th), 0.],
        [0., 1., 0., 0.],
        [np.sin(th), 0., np.cos(th), 0.],
        [0., 0., 0., 1.]], dtype=torch.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.tensor([[-1., 0., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.]]) @ c2w
    return c2w


def load_blender_data(base_dir, half_res=False, test_skip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(base_dir, f'transforms_{s}.json'), 'r') as fp:
            metas[s] = json.load(fp)  # 读取 JSON 文本，解析 JSON，将其转换为 Python 对象（各数据类型一一映射过来

    all_imgs = []
    all_poses = []
    counts = [0]  # 累计帧数
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or test_skip == 0:
            skip = 1
        else:
            skip = test_skip

        # 隔 skip 帧读取图像和位姿
        for frame in meta['frames'][::skip]:
            f_name = os.path.join(base_dir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(f_name))  # 读图，uint8 类型的 ndarray, [H, W, 4]
            # 读位姿，float32 类型的 ndarray, [4, 4]
            poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
        imgs = np.array(imgs, dtype=np.float32) / 255.  # keep all 4 channels (RGBA)
        poses = np.array(poses)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # 将边界变成索引区间
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)  # 按第一维（三个集合）连接
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]

    # camera_angle_x 是相机的水平视场角
    camera_angle_x = float(metas['test']['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0)
                                for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4), dtype=np.float32)
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, (H, W, focal), i_split
