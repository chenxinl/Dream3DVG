import torch

import os
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

import random
import pathlib

import omegaconf
import pydiffvg
import numpy as np
import torch

from pytorch_svgrender.painter.dream3dvg.modules import Renderer, Optimizer
from pytorch_svgrender.utils.graphics_utils import getWorld2View
from torchvision.utils import save_image

from pytorch_svgrender.libs.modules.edge_map.DoG import XDoG
from pytorch_svgrender.diffvg_warp import DiffVGState

class Painter:
    def __init__(
        self,
        args: omegaconf.DictConfig,
        device = "cuda"
    ):
        """Main class to get sketch in a given 3D scene"""
        self.args = args.x
        self.device = device

        self.save_init = True

        self.cur_iter = 0  # initialized
        
        self.curve_params = {
            'style': self.args.style,
            'proj_mode': self.args.proj_mode,
            'proj_scale': self.args.proj_scale,
            'blender': self.args.blender,
            'use_sfm': self.args.use_sfm,
            'stroke_width': self.args.width,
            'num_strokes': self.args.num_paths,
            'num_segments': self.args.num_segments,
            'pts_per_seg': self.args.control_points_per_seg,
            'use_viewnet': self.args.use_viewnet
        }
        
        renderer_kwargs = {
            "device": self.device,
            "curve_params": self.curve_params,
        }
        
        self.renderer = Renderer(**renderer_kwargs).to(self.device)
        
        self.optim_kwargs = {
            "point_lr": self.args.lr,
            "color_lr": self.args.color_lr,
        }
        self.optimizer: Optimizer = None

        loss_params = {
            'conv': {
                'model_type': 'RN101',
                'conv_loss_type': 'L2',
                'fc_loss_type': 'Cos',
                'num_augs': 4,
                'affine': True,
                'conv_weights': [0.0, 0.0, 1.0, 1.0, 0.0],
                'c_weight': 0.0,
                'fc_weight': 75.0
            },
            'joint':{
                'loss_type': 'LPIPS',
                'size': 224,
                'weight': 1.0,
                'robust': False
            }
        }
        
        # self.loss = Loss(self.device, loss_params)
        self.loss_last = None
        self.loss_min = None
        
        self.loaded = False
        
    def prepare(
        self, dataset, train: bool = True, from_dataset=False
    ) -> torch.Tensor:
        """Prepare to optimize with the given dataset"""
        
        self.renderer.init_properties(dataset)
        
        if self.optimizer is None:
            self.optimizer = self.renderer.get_optimizer(**self.optim_kwargs)

        self.renderer.set_random_noise(save=True)
        
        if train:
            self.optimizer.set_grads()

        if not self.loaded:
            init_pose = torch.tensor(getWorld2View(dataset.test_cameras[0].R, dataset.test_cameras[0].T)).to(self.device)
            init = self.initialize(init_pose)
        else:
            init = None
        if self.save_init:
            init_svg_path = dataset.ply_path.replace('init_points3d.ply', 'svg_init.png')
            self.save_raster_svg(init, init_svg_path)
        return init
    
    def initialize(self, init_pose: torch.Tensor) -> torch.Tensor:
        if self.args.from_dataset:
            init = self.renderer.initialize_dataset(init_pose)
        else:
            init = self.renderer.initialize(init_pose)
        self.optimizer.initialize()
        self.loaded = True

        return init
    
    def save_raster_svg(self, svg, path):
        save_image(svg, path)
        
    def bezier_curve_3d(self, points: torch.Tensor, num_points: int = 100):
        """Evaluates a cubic Bezier curve at `num_points` points in 3D.
    """
        p0, p1, p2, p3 = points[0], points[1], points[2], points[3]
        t = torch.linspace(0, 1, num_points, device=points.device).unsqueeze(1)
        points = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
        return points
    
    def bezier_curve_3d_icon(self, points: torch.Tensor, num_points: int = 100):
        """Evaluates a cubic Bezier curve at `num_points` points in 3D.
    """
        line0 = points[0:4, ...]
        line1 = points[3:7, ...]
        line2 = points[6:10, ...]
        line3 =  torch.cat([points[9:, ...], points[0, ...].unsqueeze(0)], dim=0)
        lines = [line0, line1, line2, line3]
        curve_points = []
        for line in lines:
            curve_points.append(self.bezier_curve_3d(line, num_points))
        curve_points = torch.cat(curve_points, dim=0)
        return curve_points
    
    