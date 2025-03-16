# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import PIL
from PIL import Image
from typing import Callable, List, Optional, Union, Tuple, AnyStr

import omegaconf

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision import transforms
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from pytorch_svgrender.painter.dream3dvg.sd_step import ddim_step, pred_original
from pytorch_svgrender.painter.dream3dvg.view_prompt import weighted_perpendicular_aggregator
from torchvision.utils import save_image

class SDSPipeline(StableDiffusionPipeline):
    def rgb_latent_factors(self, device):
        return torch.tensor([
            # R       G       B
            [ 0.298,  0.207,  0.208],
            [ 0.187,  0.286,  0.173],
            [-0.158,  0.189,  0.264],
            [-0.184, -0.271, -0.473]
        ], device=device)
        
    def encode_(self, images):
        images = (2 * images - 1).clamp(-1.0, 1.0)  # images: [B, 3, H, W]

        # encode images
        latents = self.vae.encode(images).latent_dist.sample()
        latents = self.vae.config.scaling_factor * latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents

    def decode_(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs.to(target_dtype)

    def score_distillation_sampling(self,
                                    method: str,
                                    pred_rgb: torch.Tensor,
                                    crop_size: int,
                                    augments: str,
                                    text_embeddings: torch.Tensor, 
                                    guidance_scale: float = 100,
                                    as_latent: bool = False,
                                    grad_scale: float = 1,
                                    t_range: Union[List[float], Tuple[float]] = (0.02, 0.98),
                                    precision_t: torch.dtype = torch.float32,
                                    embedding_inverse: torch.Tensor = None,
                                    perp_neg: bool = False,
                                    weights: torch.Tensor = None,
                                    ism_param: omegaconf.DictConfig = None,
                                    current_step_percent: float = 1.,
                                    timestep_window: float = 0.2,
                                    num_train_timesteps: int = 1000,
                                    ):
        
        Bs = pred_rgb.shape[0]
        self.scheduler.set_timesteps(num_train_timesteps, device=self.device)
        
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        
        if perp_neg:
            weights = weights.reshape(-1)
            K = text_embeddings.shape[0] - 1
            
        # sampling time setting
        eps = 0.05 # prevent the final sampling range from too small
        min_step = int(num_train_timesteps * max(t_range[0], t_range[0] + (t_range[1] - t_range[0]) * (1 - current_step_percent) - timestep_window))
        max_step = int(num_train_timesteps * min(t_range[1], t_range[0] + (t_range[1] - t_range[0]) * (1 - current_step_percent + eps)))
        
        # interp to crop_size x crop_size to be fed into vae.
        if as_latent:
            latents = F.interpolate(pred_rgb.to(precision_t), (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # encode image into latents with vae, requires grad!
            latents = self.encode_(pred_rgb.to(precision_t))
        
        #  Encode input prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = text_embeddings[:, :, ...]
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])
        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, Bs, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1]).to(precision_t)

        # timestep ~ U(0.05, 0.95) to avoid very high/low noise level
        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        prev_t = max(t - ism_param.delta_t, torch.ones_like(t) * 0)
            
        # predict the noise residual with unet, stop gradient
        with torch.no_grad():
            if method == 'ism':
                xs_delta_t = ism_param.xs_delta_t if ism_param.xs_delta_t is not None else ism_param.delta_t
                xs_inv_steps = ism_param.xs_inv_steps if ism_param.xs_inv_steps is not None else int(np.ceil(prev_t.item() / xs_delta_t))
                starting_ind = max(prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(t) * 0)
                
                noise = torch.randn_like(latents)
                # Step 2: sample x_s
                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, prev_t, starting_ind, inverse_text_embeddings, 
                                                                                ism_param.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=ism_param.xs_eta) # x_s
                
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, t, prev_t, inverse_text_embeddings, 
                                                                           ism_param.denoise_guidance_scale, ism_param.delta_t, 1, is_noisy_latent=True)   # x_t     

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]
                
            elif method == 'sds':
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t) # x_t
                target = noise
                
        with torch.no_grad():
            if perp_neg:
                latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, latents_noisy.shape[-2], latents_noisy.shape[-1], )
            else:
                latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, latents_noisy.shape[-2], latents_noisy.shape[-1], )
            
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            
            unet_output = self.unet(latent_model_input.to(precision_t), tt.to(precision_t), encoder_hidden_states=text_embeddings.to(precision_t)).sample
            
        # perform guidance (high scale from paper!)
        if do_classifier_free_guidance:
            if perp_neg:
                unet_output = unet_output.reshape(1 + K, -1, 4, latents_noisy.shape[-2], latents_noisy.shape[-1], )
                noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, latents_noisy.shape[-2], latents_noisy.shape[-1],), unet_output[1:].reshape(-1, 4, latents_noisy.shape[-2], latents_noisy.shape[-1], ) 
                delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
                delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds, weights, Bs)     
            else:
                noise_pred_uncond, noise_pred_pos = unet_output.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
                delta_DSD = noise_pred_pos - noise_pred_uncond
            pred_noise = noise_pred_uncond + guidance_scale * delta_DSD

        # w(t), sigma_t^2
        w = lambda alphas: (1 - alphas)
        # w = lambda alphas: (((1 - alphas) / alphas) ** 0.5) 
        grad = grad_scale * w(alphas[t]) * (pred_noise - target)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        # add pred_x0 as supervison
        with torch.no_grad():
            pred_noise_pos = noise_pred_uncond + (1.0 + (guidance_scale - 1.0) * (num_train_timesteps - prev_t.data) / num_train_timesteps) * delta_DSD
            
            # pred_noise_pos = noise_pred_uncond + guidance_scale * delta_DSD
            
            pred_x_0_pos_latents = pred_original(self.scheduler, pred_noise_pos, prev_t, prev_latents_noisy)
            pred_x_0_pos = self.decode_(pred_x_0_pos_latents)
        
        # visualization
        with torch.no_grad():
            vis_pkg = {}
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors(device=x.device).to(x.dtype)).permute(0,3,1,2), 0., 1.)
            vis_x_s = F.interpolate(lat2rgb(prev_latents_noisy), (512, 512), mode='bilinear', align_corners=False)
            vis_x_t = F.interpolate(lat2rgb(latents_noisy), (512, 512), mode='bilinear', align_corners=False)
            vis_pkg['x_s'] = vis_x_s
            vis_pkg['x_t'] = vis_x_t
            pred_noise_ism = noise_pred_uncond + guidance_scale * delta_DSD
            pred_x_0_pos_latents_ism = pred_original(self.scheduler, pred_noise_ism, prev_t, prev_latents_noisy)
            pred_x_0_pos_ism = self.decode_(pred_x_0_pos_latents_ism)
            vis_pkg['ism'] = pred_x_0_pos_ism
        return loss, grad.mean(), pred_x_0_pos, prev_t, vis_pkg


    def add_noise_with_cfg(self, latents, noise, 
                           t, prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):
        
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, prev_t)

        cur_t = prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []
        
        for _ in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, cur_t)
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = cur_t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = cur_t.reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = unet(cur_noisy_lat_, timestep_model_input, 
                                    encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_t, unet_output))
            
            next_t = min(cur_t + delta_t, t)
            cur_t, next_t = cur_t, next_t
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_t-cur_t
            
            cur_noisy_lat = ddim_step(self.scheduler, unet_output, cur_t.item(), cur_noisy_lat, -delta_t_.item(), eta).prev_sample
            cur_t = next_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_t == t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]


class SpecifyGradient(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
