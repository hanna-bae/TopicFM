import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
from kornia.geometry.transform import warp_perspective
import kornia.augmentation as KA

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth


class GeometricSequential:
    def __init__(self, *transforms, align_corners=True) -> None:
        self.transforms = transforms
        self.align_corners = align_corners

    def __call__(self, x, mode="bilinear"):
        b, c, h, w = x.shape
        M = torch.eye(3, device=x.device)[None].expand(b, 3, 3)
        for t in self.transforms:
            if np.random.rand() < t.p:
                M = M.matmul(
                    t.compute_transformation(x, t.generate_parameters((b, c, h, w)), flags=None)
                )
        return (
            warp_perspective(
                x, M, dsize=(h, w), mode=mode, align_corners=self.align_corners
            ),
            M,
        )

    def apply_transform(self, x, M, mode="bilinear"):
        b, c, h, w = x.shape
        return warp_perspective(
            x, M, dsize=(h, w), align_corners=self.align_corners, mode=mode
        )


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        if mode == 'val':
            self.img_resize = 864
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.geometric_aug = GeometricSequential(KA.RandomAffine(degrees=90, p=0.3)) if mode == "train" else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0, _ = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1, H_mat = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, self.augment_fn, self.geometric_aug)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=None) # self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

        if self.geometric_aug:
            t_depth1 = self.geometric_aug.apply_transform(depth1[None, None], H_mat, mode='nearest')
            t_depth1 = t_depth1.squeeze(0).squeeze(0)
            depth1 = torch.zeros((self.depth_max_size, self.depth_max_size), dtype=torch.float)
            depth1[:t_depth1.shape[0], :t_depth1.shape[1]] = t_depth1
            K_1 = H_mat[0] @ K_1

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        P0 = K_0 @ torch.tensor(T0[:3, :4], dtype=torch.float)
        P1 = K_1 @ torch.tensor(T1[:3, :4], dtype=torch.float)

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'proj_mat0': P0, 'proj_mat1': P1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
