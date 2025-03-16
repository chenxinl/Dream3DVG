# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import os
import shutil

import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
import numpy as np
import diffusers
import imageio
from random import randint

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.libs.metric.clip_score import CLIPScoreWrapper
from pytorch_svgrender.painter.dream3dvg import (
    Painter, SDSPipeline, gs_render, get_text_embeddings, interpolate_embeddings, adjust_text_embeddings)
from pytorch_svgrender.painter.dream3dvg.scene import Scene
from pytorch_svgrender.utils.camera_utils import RCamera
from pytorch_svgrender.painter.dream3dvg.scene.gaussian_model import GaussianModel
from pytorch_svgrender.painter.dream3dvg.loss.main import Loss
from pytorch_svgrender.painter.dream3dvg.loss.gs_loss import tv_loss
from pytorch_svgrender.diffusers_warp import init_StableDiffusion_pipeline

class Dream3DVGPipeline(ModelState):

    def __init__(self, args):
        attn_log_ = ""
        logdir_ = f"sd{args.seed}-im{args.x.image_size}" \
                  f"-P{args.x.num_paths}W{args.x.width}{'OP' if args.x.optim_opacity else 'BL'}" \
                  f"{attn_log_}"
        super().__init__(args, log_path_suffix=logdir_)

        # create log dir
        self.svg_logs_dir = self.result_path / "svg_logs"
        self.gs_logs_dir = self.result_path / "gs_logs"
        
        # save ply dir
        self.ply_dir = self.result_path / "point_visual"
        
        # for convenient debug
        if os.path.exists(self.gs_logs_dir):
            shutil.rmtree(self.gs_logs_dir)
             
        if self.accelerator.is_main_process:
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.gs_logs_dir.mkdir(parents=True, exist_ok=True)
            self.ply_dir.mkdir(parents=True, exist_ok=True)
            
        if self.x_cfg.model_id == 'sd21':
            custom_pipeline = SDSPipeline
            custom_scheduler = diffusers.DDIMScheduler
        else:  # sd14, sd15
            custom_pipeline = SDSPipeline
            custom_scheduler = diffusers.DDIMScheduler
        
        self.precision_t = torch.float16 if self.x_cfg.fp16 else torch.float32
        if not self.x_cfg.from_dataset:
            self.diffusion = init_StableDiffusion_pipeline(
                self.x_cfg.model_id,
                custom_pipeline=custom_pipeline,
                custom_scheduler=custom_scheduler,
                device=self.device,
                torch_dtype = self.precision_t,
                local_files_only=not args.diffuser.download,
                force_download=args.diffuser.force_download,
                resume_download=args.diffuser.resume_download,
                ldm_speed_up=self.x_cfg.ldm_speed_up,
                enable_xformers=self.x_cfg.enable_xformers,
                gradient_checkpoint=self.x_cfg.gradient_checkpoint,
            )

        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

        # init clip model and clip score wrapper
        self.cargs = self.x_cfg.clip
        self.clip_score_fn = CLIPScoreWrapper(self.cargs.model_name,
                                              device=self.device,
                                              visual_score=True,
                                              feats_loss_type=self.cargs.feats_loss_type,
                                              feats_loss_weights=self.cargs.feats_loss_weights,
                                              fc_loss_weight=self.cargs.fc_loss_weight)

        # scene init (camera and point cloud init)
        self.scene = Scene(args, save_path=self.result_path)
        
        # sketcher init
        self.sketcher = Painter(args, device=self.device)
        
        # gaussian init
        if self.x_cfg.use_gaussian:
            self.gaussians = GaussianModel(self.x_cfg.gaussian_param.sh_degree)
        
        # set prompt
        if self.x_cfg.style == 'sketch':
            self.svg_prompt = self.args.prompt + ', minimal 2d line drawing, on a white background, black and white.'
            self.image_prompt = self.args.prompt
        elif self.x_cfg.style == 'iconography':
            self.svg_prompt = self.args.prompt + ', minimal 2D vector art, linear color'
            self.image_prompt = self.args.prompt
        elif self.x_cfg.style == 'ink':
            self.svg_prompt = self.args.prompt + ', minimal 2d ink drawing, on a white background, black and white.'
            self.image_prompt = self.args.prompt
        elif self.x_cfg.style == 'painting':
            self.svg_prompt = self.args.prompt + ', minimal 2d art drawing, on a white background, black and white.'
            self.image_prompt = self.args.prompt
            
        # view-dependent embeddings
        if not self.x_cfg.from_dataset:
            self.svg_embeddings = get_text_embeddings(self.diffusion, self.svg_prompt, self.args.neg_prompt)
        if self.x_cfg.use_gaussian:
            self.image_embeddings = get_text_embeddings(self.diffusion, self.image_prompt, self.args.neg_prompt)
        
        # loss for dataset
        loss_params = self.x_cfg.loss_params
         
        self.image_sketch_loss = Loss(self.device, **loss_params)
        self.loss_last = None
        self.loss_min = None
        self.step = 0
            
    @property
    def clip_norm_(self):
        return transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def clip_augment(self,
                          x: torch.Tensor,
                          im_res: int,
                          augments: str = "affine_norm",
                          num_aug: int = 4):
        # init augmentations
        augment_list = []
        if "affine" in augments:
            augment_list.append(
                transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5)
            )
            augment_list.append(
                transforms.RandomResizedCrop(im_res, scale=(0.8, 0.8), ratio=(1.0, 1.0))
            )
        augment_list.append(self.clip_norm_)  # CLIP Normalize

        # compose augmentations
        augment_compose = transforms.Compose(augment_list)
        # make augmentation pairs
        x_augs = [self.clip_score_fn.normalize(x)]
        # repeat N times
        for n in range(num_aug):
            augmented_pair = augment_compose(x)
            x_augs.append(augmented_pair[0].unsqueeze(0))
        xs = torch.cat(x_augs, dim=0)
        return xs
                
    def progressive_camera(self):
        self.scene.pose_args.fovy_range[0] = max(self.scene.pose_args.max_fovy_range[0], self.scene.pose_args.fovy_range[0] * self.x_cfg.camera_param.fovy_scale_up_factor[0])
        self.scene.pose_args.fovy_range[1] = min(self.scene.pose_args.max_fovy_range[1], self.scene.pose_args.fovy_range[1] * self.x_cfg.camera_param.fovy_scale_up_factor[1])

        self.scene.pose_args.radius_range[1] = max(self.scene.pose_args.max_radius_range[1], self.scene.pose_args.radius_range[1] * self.x_cfg.camera_param.scale_up_factor)
        self.scene.pose_args.radius_range[0] = max(self.scene.pose_args.max_radius_range[0], self.scene.pose_args.radius_range[0] * self.x_cfg.camera_param.scale_up_factor)

        self.scene.pose_args.theta_range[1] = min(self.scene.pose_args.max_theta_range[1], self.scene.pose_args.theta_range[1] * self.x_cfg.camera_param.phi_scale_up_factor)
        self.scene.pose_args.theta_range[0] = max(self.scene.pose_args.max_theta_range[0], self.scene.pose_args.theta_range[0] * 1/self.x_cfg.camera_param.phi_scale_up_factor)

        self.scene.pose_args.phi_range[0] = max(self.scene.pose_args.max_phi_range[0], self.scene.pose_args.phi_range[0] * self.x_cfg.camera_param.phi_scale_up_factor)
        self.scene.pose_args.phi_range[1] = min(self.scene.pose_args.max_phi_range[1], self.scene.pose_args.phi_range[1] * self.x_cfg.camera_param.phi_scale_up_factor)
        
        print('scale up theta_range to:', self.scene.pose_args.theta_range)
        print('scale up radius_range to:', self.scene.pose_args.radius_range)
        print('scale up phi_range to:', self.scene.pose_args.phi_range)
        print('scale up fovy_range to:', self.scene.pose_args.fovy_range)
    
    def get_opposite_cam(self, viewpoint_cam):
        opposite_delta_azimuth = viewpoint_cam.delta_azimuth + 0. + 180. # default_azimuth=0
        if opposite_delta_azimuth > 180:
            opposite_delta_azimuth -= 360
        viewpoint_cam_w2c=viewpoint_cam.world_view_transform.T.cpu().numpy()
        
        opposite_viewpoint_cam = RCamera(
            colmap_id=viewpoint_cam.colmap_id,
            R=(viewpoint_cam_w2c[:3, :3] @ np.diag([-1., -1., -1.])).T,
            T=viewpoint_cam_w2c[:3, -1],
            FoVx=viewpoint_cam.FoVx,
            FoVy=viewpoint_cam.FoVy,
            uid=viewpoint_cam.uid,
            delta_polar=viewpoint_cam.delta_polar, # theta keeps same
            delta_azimuth=opposite_delta_azimuth,
            delta_radius=viewpoint_cam.delta_radius,
            opt=viewpoint_cam.opt,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
            data_device=self.device,
            SSAA=False                                          
            )
        return opposite_viewpoint_cam
                                 
    def painterly_rendering(self):
        # log prompts
        self.print(f"svg_prompt: {self.svg_prompt}")
        self.print(f"image_prompt: {self.image_prompt}")
        self.print(f"negative_prompt: {self.args.neg_prompt}\n")
        
        total_iter = self.x_cfg.num_iter
        
        # Init strokes from point cloud (point-e)
        self.sketcher.prepare(self.scene.init_scene_info)
        # Init gaussians from point cloud (point-e)
        if self.x_cfg.use_gaussian:
            self.gaussians.create_from_pcd(self.scene.init_scene_info.point_cloud, self.x_cfg.camera_param.default_radius)
            self.gaussians.training_setup(self.x_cfg.gaussian_param, total_iter)
            bg_color = [1, 1, 1] if self.x_cfg.gaussian_param.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        
        # train process video
        if self.args.mv:
            save_folder_proc = os.path.join(self.result_path, "process_videos/")
            if not os.path.exists(save_folder_proc):
                os.makedirs(save_folder_proc)  # makedirs
            process_view_points = self.scene.getCircleVideoCameras(batch_size=600 if total_iter >= 600 else total_iter, render45=False).copy()
            process_final_view_points = self.scene.getCircleVideoCameras(batch_size=100, render45=False).copy()
            
            save_process_iter = total_iter // len(process_view_points)
            pro_svg_frames = []
            pro_svg_frames_final = []
            if self.x_cfg.use_gaussian:
                pro_gs_frames = []  
                pro_gs_frames_final = []
            
        self.print(f"\ntotal optimization steps: {total_iter}")
        with tqdm(initial=self.step, total=total_iter, disable=not self.accelerator.is_main_process, colour="blue") as pbar:
            while self.step < total_iter:
                if self.x_cfg.use_gaussian:
                    self.gaussians.update_learning_rate(self.step)
                    self.gaussians.update_feature_learning_rate(self.step)
                    self.gaussians.update_rotation_learning_rate(self.step)
                    self.gaussians.update_scaling_learning_rate(self.step)
                    # Every 500 its we increase the levels of SH up to a maximum degree
                    if self.step % 500 == 0:
                        self.gaussians.oneupSHdegree()
            
                # progressively relaxing view range    
                if self.x_cfg.camera_param.use_progressive:                
                    if self.step >= self.x_cfg.camera_param.progressive_view_iter and self.step % self.x_cfg.camera_param.scale_up_cameras_iter == 0:
                        self.progressive_camera()
                
                # pick a random camera
                viewpoint_stack = self.scene.getRandTrainCameras(batch=self.x_cfg.batch).copy()
                
                C_batch_size = self.x_cfg.C_batch_size
                raster_sketches = []
                images = []
                depths = []
                scales = []
                alphas = []
                weights_ = []
                weights_gs_ = []
                text_z_ = []
                text_z_gs_ = []
                opposite_depths = []
                disps = []
                opposite_disps = []

                
                text_z_inverse = torch.cat([self.svg_embeddings['uncond'], self.svg_embeddings['inverse']], dim=0)
                if self.x_cfg.use_gaussian:
                    text_z_inverse_gs = torch.cat([self.image_embeddings['uncond'], self.image_embeddings['inverse']], dim=0)
                
                for i in range(C_batch_size):
                    try:
                        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))            
                    except:
                        viewpoint_stack = self.scene.getRandTrainCameras().copy()
                        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                    azimuth = viewpoint_cam.delta_azimuth
                    
                    #pred text_z
                    text_z = [self.svg_embeddings['uncond']]
                    
                    # interpolate view embeddings
                    if self.x_cfg.sds.perp_neg:
                        text_z_comp, weights = adjust_text_embeddings(self.svg_embeddings, azimuth)
                        text_z.append(text_z_comp)
                        weights_.append(weights)
                    else:
                        text_z.append(interpolate_embeddings(self.svg_embeddings, azimuth))
                    text_z = torch.cat(text_z, dim=0)
                    text_z_.append(text_z)
                    
                    if self.x_cfg.use_gaussian:
                        # render_gs main_view
                        text_z_gs = [self.image_embeddings['uncond']]
                        if self.x_cfg.sds.perp_neg:
                            text_z_comp_gs, weights_gs = adjust_text_embeddings(self.image_embeddings, azimuth)
                            text_z_gs.append(text_z_comp_gs)
                            weights_gs_.append(weights_gs)
                        else:
                            text_z_gs.append(interpolate_embeddings(self.image_embeddings, azimuth))
                        text_z_gs = torch.cat(text_z_gs, dim=0)
                        text_z_gs_.append(text_z_gs)
                        
                        render_pkg = gs_render(viewpoint_cam, self.gaussians, background, 
                                convert_SHs_python =  self.x_cfg.gaussian_param.convert_SHs_python,
                                compute_cov3D_python = self.x_cfg.gaussian_param.compute_cov3D_python,
                                sh_deg_aug_ratio = self.x_cfg.gaussian_param.sh_deg_aug_ratio, 
                                bg_aug_ratio = self.x_cfg.gaussian_param.bg_aug_ratio, 
                                shs_aug_ratio = self.x_cfg.gaussian_param.shs_aug_ratio, 
                                scale_aug_ratio = self.x_cfg.gaussian_param.scale_aug_ratio)
                        
                        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                
                        depth, alpha, disp = render_pkg["depth"], render_pkg["alpha"], render_pkg["disp"].unsqueeze(0)
                        images.append(image)
                        depths.append(depth)
                        alphas.append(alpha)
                        disps.append(disp)
                        scales.append(render_pkg["scales"])
                        
                        # render_gs opposite_view
                        with torch.no_grad():
                            opposite_viewpoint_cam = self.get_opposite_cam(viewpoint_cam)

                            opposite_render_pkg = gs_render(opposite_viewpoint_cam, self.gaussians, background, 
                                convert_SHs_python =  self.x_cfg.gaussian_param.convert_SHs_python,
                                compute_cov3D_python = self.x_cfg.gaussian_param.compute_cov3D_python,
                                sh_deg_aug_ratio = self.x_cfg.gaussian_param.sh_deg_aug_ratio, 
                                bg_aug_ratio = self.x_cfg.gaussian_param.bg_aug_ratio, 
                                shs_aug_ratio = self.x_cfg.gaussian_param.shs_aug_ratio, 
                                scale_aug_ratio = self.x_cfg.gaussian_param.scale_aug_ratio)
                            opposite_depth, opposite_disp = opposite_render_pkg["depth"], opposite_render_pkg["disp"].unsqueeze(0)
                            opposite_depths.append(opposite_depth)
                            opposite_disps.append(opposite_disp)
                    
                    if self.x_cfg.use_gaussian:     
                        raster_sketch = self.sketcher.renderer(
                            viewpoint_cam.world_view_transform.T, 
                            w2c=True, 
                            intrinsic=viewpoint_cam.intrinsic, 
                            depth=depth if self.x_cfg.use_gaussian else None, 
                            opposite_pose=opposite_viewpoint_cam.world_view_transform.T if self.x_cfg.use_gaussian else None, 
                            opposite_depth=opposite_depth if self.x_cfg.use_gaussian else None,
                            is_test=False
                            ) # (1, 3, H, W)
                    else:
                        raster_sketch = self.sketcher.renderer(
                            viewpoint_cam.world_view_transform.T, 
                            w2c=True, 
                            intrinsic=viewpoint_cam.intrinsic
                            ) 
                    raster_sketches.append(raster_sketch)
                    
                raster_sketches = torch.cat(raster_sketches, dim=0)
                text_embeddings = torch.stack(text_z_, dim=1)
                
                if self.x_cfg.use_gaussian:
                    images = torch.stack(images, dim=0)
                    depths = torch.stack(depths, dim=0)
                    scales = torch.stack(scales, dim=0)
                    alphas = torch.stack(alphas, dim=0)
                    text_embeddings_gs = torch.stack(text_z_gs_, dim=1)
                    # disp for visulization
                    disps = torch.cat(disps, dim=0)
                    opposite_disps = torch.cat(opposite_disps, dim=0)
                
                # sketch sds loss
                sds_loss, grad = torch.tensor(0), torch.tensor(0)
                l_tvd = torch.tensor(0)
                if self.step >= self.x_cfg.sds.warmup or not self.x_cfg.use_gaussian:
                    sds_loss, grad, pred_x_0_pos, timestep, _ = self.diffusion.score_distillation_sampling(
                        method=self.x_cfg.sds.method,
                        pred_rgb=raster_sketches,
                        crop_size=self.x_cfg.sds.crop_size,
                        augments=self.x_cfg.sds.augmentations,
                        text_embeddings=text_embeddings,
                        guidance_scale=self.x_cfg.sds.guidance_scale,
                        grad_scale=self.x_cfg.sds.grad_scale,
                        perp_neg=self.x_cfg.sds.perp_neg,
                        weights=torch.stack(weights_, dim=1) if self.x_cfg.sds.perp_neg else None,
                        t_range=list(self.x_cfg.sds.t_range),
                        precision_t=self.precision_t,
                        embedding_inverse=text_z_inverse,
                        ism_param=self.x_cfg.sds.ism_param,
                        current_step_percent=self.step / total_iter
                    )
            
                # adopt loss
                if self.x_cfg.use_gaussian:
                    # image loss
                    gs_grad_scale = self.x_cfg.sds.grad_scale_gs
                    sds_loss_gs, grad_gs = torch.tensor(0), torch.tensor(0)
                    sds_loss_gs, grad_gs, pred_x_0_pos_gs, timestep_gs, vis_pkg = self.diffusion.score_distillation_sampling(
                        method=self.x_cfg.sds.method,
                        pred_rgb=images,
                        crop_size=self.x_cfg.sds.crop_size,
                        augments=self.x_cfg.sds.augmentations,
                        text_embeddings=text_embeddings_gs,
                        guidance_scale=self.x_cfg.sds.gs_guidance_scale,
                        grad_scale=gs_grad_scale,
                        t_range=list(self.x_cfg.sds.t_range),
                        precision_t=self.precision_t,
                        embedding_inverse=text_z_inverse_gs,
                        perp_neg=self.x_cfg.sds.perp_neg,
                        weights=torch.stack(weights_gs_, dim=1) if self.x_cfg.sds.perp_neg else None,
                        ism_param=self.x_cfg.sds.ism_param,
                        current_step_percent=self.step / total_iter
                    ) 
                    
                    loss_scale = torch.mean(scales,dim=-1).mean()
                    loss_tv = tv_loss(images) + tv_loss(depths) 

                    loss_gs = sds_loss_gs + 1. * loss_tv + 1. * loss_scale
                    
                    # sketch loss
                    if self.x_cfg.use_pseudo:
                        clip_loss = self.image_sketch_loss(raster_sketches, pred_x_0_pos_gs, train=True)
                    else:
                        clip_loss = self.image_sketch_loss(raster_sketches, images.detach(), train=True)

                    loss = sds_loss + clip_loss
                else:
                    # sektch loss
                    loss = sds_loss
                    
                if self.x_cfg.use_gaussian:
                    curve_points_3d = []
                    for curve_point in self.sketcher.renderer.curve_renderer.point_params:
                        if self.x_cfg.style == 'iconography':
                            curve_points_3d.append(self.sketcher.bezier_curve_3d_icon(curve_point))
                        else:
                            curve_points_3d.append(self.sketcher.bezier_curve_3d(curve_point))
                    curve_points_3d = torch.cat(curve_points_3d, dim=0)
                    if self.step % 200 == 0:
                        self.save_ply(curve_points_3d, self.gaussians.get_xyz)
                
                # optimize svg
                self.sketcher.optimizer.zero_grad()
                loss.backward()
                self.sketcher.optimizer.step()
                
                # optimize gaussian
                if self.x_cfg.use_gaussian:
                    loss_gs.backward()    
                
                    # gaussian densification
                    if self.step < self.x_cfg.gaussian_param.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if self.step > self.x_cfg.gaussian_param.densify_from_iter and self.step % self.x_cfg.gaussian_param.densification_interval == 0:
                            size_threshold = 20 if self.step > self.x_cfg.gaussian_param.opacity_reset_interval else None
                            self.gaussians.densify_and_prune(self.x_cfg.gaussian_param.densify_grad_threshold, 0.005, self.x_cfg.camera_param.default_radius, size_threshold)
                        
                        if self.step % self.x_cfg.gaussian_param.opacity_reset_interval == 0 and self.step != 0:
                            self.gaussians.reset_opacity()
                
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)  
                          
                # update lr
                # if self.x_cfg.lr_schedule:
                #     self.sketcher.optimizer.update_lr(self.step, self.x_cfg.decay_steps)

                # records
                if self.x_cfg.use_gaussian:
                    pbar.set_description(
                    f"Iter: {self.step}, "   
                    f"loss: {loss.item():.4f}, "
                    f"loss_gs: {loss_gs.item():.4f}, "
                    f"sds_grad_gs: {grad_gs.item():.4e}"
                )
                else:
                    pbar.set_description(
                    f"Iter: {self.step}, "
                    f"loss: {loss.item():.4f}, "
                    f"sds_grad: {grad.item():.4e}"
                )

                # log raster and svg
                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    if self.x_cfg.use_gaussian:
                        if self.x_cfg.use_pseudo:
                            gs_logs_path = os.path.join(self.gs_logs_dir, 'iter_{}_gs_{}.png'.format(self.step, timestep_gs.item()))
                            if self.step >= self.x_cfg.sds.warmup:
                                save_image(torch.cat([images, pred_x_0_pos_gs, raster_sketches, disps.repeat(1, 3, 1, 1), opposite_disps.repeat(1, 3, 1, 1)], dim=0), gs_logs_path, nrow=C_batch_size)
                            else:
                                save_image(torch.cat([images, pred_x_0_pos_gs, raster_sketches, disps.repeat(1, 3, 1, 1), opposite_disps.repeat(1, 3, 1, 1)], dim=0), gs_logs_path, nrow=C_batch_size)

                        else:
                            gs_logs_path = os.path.join(self.gs_logs_dir, 'iter_{}.png'.format(self.step))
                            save_image(torch.cat([images, pred_x_0_pos, raster_sketches], dim=0), gs_logs_path, nrow=C_batch_size)
                    else:
                        self.sketcher.save_raster_svg(raster_sketches, os.path.join(self.svg_logs_dir, 'iter_{}.png'.format(self.step)))
                
                # save video
                with torch.no_grad():
                    if self.args.mv:
                        if self.step % save_process_iter == 0 and len(process_view_points) > 0:
                            viewpoint_cam_p = process_view_points.pop(0)
                            if self.x_cfg.use_gaussian:
                                render_pkg = gs_render(viewpoint_cam_p, self.gaussians, background, 
                                    convert_SHs_python =  self.x_cfg.gaussian_param.convert_SHs_python,
                                    compute_cov3D_python = self.x_cfg.gaussian_param.compute_cov3D_python,
                                    sh_deg_aug_ratio = self.x_cfg.gaussian_param.sh_deg_aug_ratio, 
                                    bg_aug_ratio = self.x_cfg.gaussian_param.bg_aug_ratio, 
                                    shs_aug_ratio = self.x_cfg.gaussian_param.shs_aug_ratio, 
                                    scale_aug_ratio = self.x_cfg.gaussian_param.scale_aug_ratio)
                                depth_p = render_pkg['depth']
                                # oppo_cam
                                opposite_viewpoint_cam_p = self.get_opposite_cam(viewpoint_cam_p)
                                opposite_render_pkg = gs_render(opposite_viewpoint_cam_p, self.gaussians, background, 
                                    convert_SHs_python =  self.x_cfg.gaussian_param.convert_SHs_python,
                                    compute_cov3D_python = self.x_cfg.gaussian_param.compute_cov3D_python,
                                    sh_deg_aug_ratio = self.x_cfg.gaussian_param.sh_deg_aug_ratio, 
                                    bg_aug_ratio = self.x_cfg.gaussian_param.bg_aug_ratio, 
                                    shs_aug_ratio = self.x_cfg.gaussian_param.shs_aug_ratio, 
                                    scale_aug_ratio = self.x_cfg.gaussian_param.scale_aug_ratio)
                                opposite_depth_p = opposite_render_pkg['depth']
                            
                            svg_p = self.sketcher.renderer(viewpoint_cam_p.world_view_transform.T, w2c=True, 
                                        intrinsic=viewpoint_cam_p.intrinsic, depth=depth_p if self.x_cfg.use_gaussian else None, 
                                        opposite_pose=opposite_viewpoint_cam_p.world_view_transform.T if self.x_cfg.use_gaussian else None, 
                                        opposite_depth=opposite_depth_p if self.x_cfg.use_gaussian else None,
                                        is_test=True
                                        ).permute(0, 2, 3, 1) 
                            
                            svg_p = torch.clamp(svg_p, 0.0, 1.0)
                            svg_p = svg_p.squeeze(0).detach().cpu().numpy()
                            svg_p = (svg_p * 255).round().astype('uint8')
                            pro_svg_frames.append(svg_p)
                            
                            if self.x_cfg.use_gaussian:
                                render_p = gs_render(viewpoint_cam_p, self.gaussians, background, test=True)
                                img_p = torch.clamp(render_p["render"], 0.0, 1.0) 
                                img_p = img_p.detach().cpu().permute(1,2,0).numpy()
                                img_p = (img_p * 255).round().astype('uint8')
                                pro_gs_frames.append(img_p)

                        # final video
                        if self.step == total_iter - 1:
                            for _ in range(len(process_final_view_points)):
                                viewpoint_cam_final = process_final_view_points.pop(0)
                                if self.x_cfg.use_gaussian:
                                    render_pkg = gs_render(viewpoint_cam_final, self.gaussians, background, 
                                        convert_SHs_python =  self.x_cfg.gaussian_param.convert_SHs_python,
                                        compute_cov3D_python = self.x_cfg.gaussian_param.compute_cov3D_python,
                                        sh_deg_aug_ratio = self.x_cfg.gaussian_param.sh_deg_aug_ratio, 
                                        bg_aug_ratio = self.x_cfg.gaussian_param.bg_aug_ratio, 
                                        shs_aug_ratio = self.x_cfg.gaussian_param.shs_aug_ratio, 
                                        scale_aug_ratio = self.x_cfg.gaussian_param.scale_aug_ratio)
                                    depth_final = render_pkg['depth']
                                    # oppo_cam
                                    opposite_viewpoint_cam_final = self.get_opposite_cam(viewpoint_cam_final)
                                    opposite_render_pkg = gs_render(opposite_viewpoint_cam_final, self.gaussians, background, 
                                        convert_SHs_python =  self.x_cfg.gaussian_param.convert_SHs_python,
                                        compute_cov3D_python = self.x_cfg.gaussian_param.compute_cov3D_python,
                                        sh_deg_aug_ratio = self.x_cfg.gaussian_param.sh_deg_aug_ratio, 
                                        bg_aug_ratio = self.x_cfg.gaussian_param.bg_aug_ratio, 
                                        shs_aug_ratio = self.x_cfg.gaussian_param.shs_aug_ratio, 
                                        scale_aug_ratio = self.x_cfg.gaussian_param.scale_aug_ratio)
                                    opposite_depth_final = opposite_render_pkg['depth']
                                
                                svg_p = self.sketcher.renderer(viewpoint_cam_final.world_view_transform.T, w2c=True, 
                                            intrinsic=viewpoint_cam_final.intrinsic, depth=depth_final if self.x_cfg.use_gaussian else None, 
                                            opposite_pose=opposite_viewpoint_cam_final.world_view_transform.T if self.x_cfg.use_gaussian else None, 
                                            opposite_depth=opposite_depth_final if self.x_cfg.use_gaussian else None,
                                            is_test=True
                                            ).permute(0, 2, 3, 1)  
                                
                                svg_p = torch.clamp(svg_p, 0.0, 1.0)
                                svg_p = svg_p.squeeze(0).detach().cpu().numpy()
                                svg_p = (svg_p * 255).round().astype('uint8')
                                pro_svg_frames_final.append(svg_p)
                            
                                if self.x_cfg.use_gaussian:
                                    render_p = gs_render(viewpoint_cam_final, self.gaussians, background, test=True)
                                    img_p = torch.clamp(render_p["render"], 0.0, 1.0) 
                                    img_p = img_p.detach().cpu().permute(1,2,0).numpy()
                                    img_p = (img_p * 255).round().astype('uint8')
                                    pro_gs_frames_final.append(img_p) 
                # eval
                if (
                    self.step == 0 or self.step % self.x_cfg.eval_freq == self.x_cfg.eval_freq - 1
                ):
                    self.eval()
                 
                self.step += 1
                pbar.update(1)
        
        # save video
        if self.args.mv:
            imageio.mimwrite(os.path.join(save_folder_proc, "video_svg.mp4"), pro_svg_frames, fps=30, quality=8)
            imageio.mimwrite(os.path.join(save_folder_proc, "video_svg_final.mp4"), pro_svg_frames_final, fps=10, quality=8)
            if self.x_cfg.use_gaussian:
                imageio.mimwrite(os.path.join(save_folder_proc, "video_gs.mp4"), pro_gs_frames, fps=30, quality=8)
                imageio.mimwrite(os.path.join(save_folder_proc, "video_gs_final.mp4"), pro_gs_frames_final, fps=10, quality=8)
            
        self.close(msg="painterly rendering complete.")
                        
    def save_ply(self, curve_points, gauss_points):
        from pytorch_svgrender.painter.dream3dvg.scene.dataset_readers import storePly
        curve_points = curve_points.detach().cpu().numpy()
        rgb_curve = np.tile(np.array([0, 0, 255]), (curve_points.shape[0], 1))
        gauss_points = gauss_points.detach().cpu().numpy()
        rgb_gauss = np.tile(np.array([0, 255, 0]), (gauss_points.shape[0], 1))
        points = np.concatenate((curve_points, gauss_points), axis=0)
        colors = np.concatenate((rgb_curve, rgb_gauss), axis=0)
        storePly('{}/points_{}.ply'.format(self.ply_dir, self.step), points, colors)        
                
    @torch.no_grad()
    def eval(self):
        save_folder = os.path.join(self.result_path, 'eval/iter_{}'.format(self.step))
        gs_save_folder = os.path.join(self.result_path, 'eval_gs/iter_{}'.format(self.step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(gs_save_folder):
            os.makedirs(gs_save_folder)
        test_cameras = self.scene.getTestCameras()
        
        # add guidance visualization
        if self.x_cfg.use_gaussian:
            text_z_gs_ = []
            weights_gs_ = []
            images = []
            
        if len(test_cameras) > 0:
            for test_camera in test_cameras:
                if self.x_cfg.use_gaussian:
                    azimuth = test_camera.delta_azimuth
                    text_z_gs = [self.image_embeddings['uncond']]
                    if self.x_cfg.sds.perp_neg:
                        text_z_comp_gs, weights_gs = adjust_text_embeddings(self.image_embeddings, azimuth.cuda())
                        text_z_gs.append(text_z_comp_gs)
                        weights_gs_.append(weights_gs)
                    else:
                        text_z_gs.append(interpolate_embeddings(self.image_embeddings, azimuth))
                    text_z_gs = torch.cat(text_z_gs, dim=0)
                    text_z_gs_.append(text_z_gs)
                    
                    bg_color = [1, 1, 1] if self.x_cfg.gaussian_param.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
                    render_pkg = gs_render(test_camera, self.gaussians, background, 
                            convert_SHs_python =  self.x_cfg.gaussian_param.convert_SHs_python,
                            compute_cov3D_python = self.x_cfg.gaussian_param.compute_cov3D_python,
                            sh_deg_aug_ratio = self.x_cfg.gaussian_param.sh_deg_aug_ratio, 
                            bg_aug_ratio = 0., 
                            shs_aug_ratio = self.x_cfg.gaussian_param.shs_aug_ratio, 
                            scale_aug_ratio = self.x_cfg.gaussian_param.scale_aug_ratio)
                    depth = render_pkg["depth"]
                    image = render_pkg["render"]
                    images.append(image)
                    
                    opposite_test_camera = self.get_opposite_cam(test_camera)
                    opposite_render_pkg = gs_render(opposite_test_camera, self.gaussians, background, 
                            convert_SHs_python =  self.x_cfg.gaussian_param.convert_SHs_python,
                            compute_cov3D_python = self.x_cfg.gaussian_param.compute_cov3D_python,
                            sh_deg_aug_ratio = self.x_cfg.gaussian_param.sh_deg_aug_ratio, 
                            bg_aug_ratio = self.x_cfg.gaussian_param.bg_aug_ratio, 
                            shs_aug_ratio = self.x_cfg.gaussian_param.shs_aug_ratio, 
                            scale_aug_ratio = self.x_cfg.gaussian_param.scale_aug_ratio)
                    opposite_depth = opposite_render_pkg["depth"]
                    
                    sketch = self.sketcher.renderer(test_camera.world_view_transform.T, 
                                w2c=True, intrinsic=test_camera.intrinsic, 
                                depth=depth,
                                opposite_pose=opposite_test_camera.world_view_transform.T,
                                opposite_depth=opposite_depth,
                                is_test=True
                                )
                else:
                    sketch = self.sketcher.renderer(test_camera.world_view_transform.T, 
                                w2c=True, intrinsic=test_camera.intrinsic,
                                )
                
                sketch = torch.clamp(sketch, 0.0, 1.0)
                save_image(sketch, os.path.join(save_folder,"render_view_{}.png".format(test_camera.uid)))
                
                if self.x_cfg.use_gaussian:
                    save_image(image.unsqueeze(0), os.path.join(gs_save_folder, "render_view_{}.png".format(test_camera.uid)))
            