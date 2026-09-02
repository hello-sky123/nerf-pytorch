import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_LINEMOD_data(base_dir, half_res=False, test_skip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(base_dir, f'transforms_{s}.json'), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or test_skip == 0:
            skip = 1
        else:
            skip = test_skip

        for idx_test, frame in enumerate(meta['frames'][::skip]):
            f_name = frame['file_path']
            if s == 'test':
                print(f"{idx_test}th test frame: {f_name}")
            imgs.append(imageio.imread(f_name))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    # Read the split explicitly instead of relying on `meta` leaking out of the
    # loop above. 'test' is the last split, so this is the value the leaked
    # variable held.
    focal = float(metas['test']['frames'][0]['intrinsic_matrix'][0][0])
    K = metas['test']['frames'][0]['intrinsic_matrix']
    print(f"Focal: {focal}")

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0)
                               for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    near = np.floor(min(metas['train']['near'], metas['test']['near']))
    far = np.ceil(max(metas['train']['far'], metas['test']['far']))
    return imgs, poses, render_poses, [H, W, focal], K, i_split, near, far
