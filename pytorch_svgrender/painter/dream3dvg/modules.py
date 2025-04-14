import torch
import torch.nn as nn
import torch.optim as optim

from typing import Dict, Any, Tuple
from dataclasses import dataclass

from pytorch_svgrender.painter.dream3dvg.network import CurveRenderer, CurveOptimizer
from pytorch_svgrender.utils.misc import conditional_decorator, HWF

class Renderer(nn.Module):  # XXX optimize this later
    def __init__(
        self,
        device: str = "cuda:0",
        curve_params: Dict[str, Any] = None,
    ):
        """Main module to represent contours and sketches based on bezier curves and superquadrics"""

        super().__init__()

        self.device = device
        
        self.curve_renderer = CurveRenderer(device, **curve_params).to(self.device)

        self.use_curve: bool = True

    def get_optimizer(self, **kwargs):
        optimizer = Optimizer(
            self.curve_renderer, **kwargs
        )
        return optimizer

    def init_properties_viewer(self, hwf: HWF):
        self.curve_renderer.init_properties_viewer(hwf)

    def init_properties(self, dataset):
        self.curve_renderer.init_properties(dataset)
    
    def init_properties_dataset(self, dataset):
        self.curve_renderer.init_properties_dataset(dataset)
        
    def set_random_noise(self, save: bool = True):
        self.curve_renderer.set_random_noise(save=save)

    def initialize(self, pose: torch.Tensor) -> torch.Tensor:
        init = self.curve_renderer.initialize(pose)
        return init

    def initialize_dataset(self, pose: torch.Tensor) -> torch.Tensor:
        init = self.curve_renderer.initialize_dataset(pose)
        return init
    
    def save_svg(self, fname: str):
        self.curve_renderer.save_svg(fname)

    def clean_strokes(self):
        self.curve_renderer.cleaning()

    def forward(
        self,
        pose: torch.Tensor,
        w2c=False,
        intrinsic=None,
        depth=None,
        opposite_pose=None,
        opposite_depth=None,
        is_test=False,
    ) -> torch.Tensor:
        # draw view-dependent sketch
        sketch = self.curve_renderer.sketch(
            pose.squeeze(),
            w2c=w2c, 
            intrinsic=intrinsic, 
            depth=depth,
            opposite_pose=opposite_pose,
            opposite_depth=opposite_depth,
            is_test=is_test
            )

        return sketch

    def gui(self) -> None:
        self.curve_renderer.gui()

    def state_dict(self) -> Dict[str, Any]:
        states = {}
        if self.use_curve:
            states.update({"curve": self.curve_renderer.state_dict()})

        return states

    def load_state_dict(self, ckpt: Dict[str, Any]):
        if self.use_curve and "curve" in ckpt.keys():
            self.curve_renderer.load_state_dict(ckpt["curve"])


class Optimizer:
    def __init__(
        self,
        curve: CurveRenderer,
        point_lr: float = 1.0,
        color_lr: float = 0.1,
    ):
        """Main class of the optimizer"""

        self.grad_curve: bool = False

        self.curve_optim = CurveOptimizer(
            module=curve,
            point_lr=point_lr,
            color_lr=color_lr,
        )

    def get_scheduler(self, steps: int, min_lr: float):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.curve_optim, T_max=steps, eta_min=min_lr
        )

        return scheduler

    def set_grads(self):
        self.grad_curve = True
        
    def initialize(self):
        self.curve_optim.initialize()

    def zero_grad(self):
        if self.grad_curve:
            self.curve_optim.zero_grad()

    def step(self):
        if self.grad_curve:
            self.curve_optim.step()

    def state_dict(self) -> Dict[str, Any]:
        states = {}
        if self.grad_curve:
            states.update({"curve_optim": self.curve_optim.state_dict()})

        return states

    def load_state_dict(self, ckpt: Dict[str, Any]):
        if self.grad_curve and "curve_optim" in ckpt.keys():
            self.curve_optim.load_state_dict(ckpt["curve_optim"])
