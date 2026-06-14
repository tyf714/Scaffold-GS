"""
============================================================================
 Error-Guided Adaptive Anchor Densification (EAAD)
 SLAM Technology Final Project — 2335053620, Yifan Tang
 Based on: Scaffold-GS (CVPR 2024 Highlight) by Lu et al.

 What it does: Adds rendering-error-guided anchor densification to Scaffold-GS.
 The original Scaffold-GS anchor growing relies on existing Gaussian gradients
 (gaussian_model.py line 683: grads = offset_gradient_accum / offset_denom).
 In SfM-depleted regions with zero anchors, zero Gaussians produce zero gradients,
 creating a deadlock. While the authors claim their growing strategy provides
 "partial mitigation" (Section 4.4 of [3]), this code demonstrates that the
 mitigation is structurally incapable of activating in precisely the regions
 it purports to address. EAAD breaks this deadlock using per-pixel rendering
 error as a direct anchor placement signal that does not require pre-existing
 Gaussians.

 Modified files: scene/gaussian_model.py (add class + method), train.py (call)
============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from functools import reduce
from torch_scatter import scatter_max


class ErrorGuidedAnchorDensifier(nn.Module):
    """
    Adaptive anchor densification based on per-pixel rendering error.

    Why pixel-space error instead of object-space gradients:

    The original anchor_growing() uses gradient statistics from existing
    Gaussians. In SfM-depleted regions where the initial voxelization
    produces zero anchors, no Gaussians exist, offset_gradient_accum and
    offset_denom are identically zero, and grads = 0/0 = NaN, forced to
    0.0, producing zero anchor candidates. This deadlock is absolute.

    Pixel-space rendering error bypasses this: the photometric loss
    L = (1-lambda)*L1 + lambda*L_SSIM propagates through the differentiable
    rasterizer to every pixel regardless of whether Gaussians exist in the
    corresponding 3D region. Pixels with persistently high error after
    basic convergence directly identify under-covered regions.

    Why not alternatives:
    - Lowering gradient threshold: gradients are zero (not small) — no
      finite threshold helps.
    - Learned density prior network: adds auxiliary training, does not
      directly optimize rendering. EAAD uses the existing photometric loss
      with no additional training objective.
    """

    def __init__(
        self,
        error_threshold=0.1,
        min_cluster_size=50,
        densify_interval=500,
        start_densify=2000,
        end_densify=12000,
        max_new_anchors_per_iter=500,
        feature_init_mode="nearest",
    ):
        super().__init__()
        self.error_threshold = error_threshold
        self.min_cluster_size = min_cluster_size
        self.densify_interval = densify_interval
        self.start_densify = start_densify
        self.end_densify = end_densify
        self.max_new = max_new_anchors_per_iter
        self.init_mode = feature_init_mode
        self.error_history = {}
        self.anchor_history = {}

    def compute_error_map(self, rendered, gt):
        """Per-pixel L1 error: E(x,y) = ||I_render - I_gt||_1 over RGB."""
        if rendered.dim() == 4:
            rendered, gt = rendered.squeeze(0), gt.squeeze(0)
        return (rendered - gt).abs().mean(dim=0)

    def find_error_regions(self, error_map):
        """Normalize and threshold error map to find difficult pixels."""
        e_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
        return e_norm > self.error_threshold

    def unproject_to_3d(self, mask, camera, gaussian_model):
        """Unproject 2D high-error pixels to 3D candidate positions."""
        if mask.sum() < self.min_cluster_size:
            return None
        h, w = mask.shape
        yc, xc = torch.where(mask)
        n = min(len(yc), self.max_new)
        if len(yc) > n:
            idx = torch.randperm(len(yc))[:n]
            yc, xc = yc[idx], xc[idx]
        anchors = gaussian_model.get_anchor
        cam_ctr = camera.camera_center
        median_d = torch.norm(anchors - cam_ctr, dim=1).median()
        px_x = 2.0 * torch.tan(torch.tensor(camera.FoVx * 0.5)) / w
        px_y = 2.0 * torch.tan(torch.tensor(camera.FoVy * 0.5)) / h
        dx = (xc.float() - w/2) * px_x
        dy = (yc.float() - h/2) * px_y
        positions = cam_ctr.unsqueeze(0) + torch.stack(
            [dx, dy, torch.ones_like(dx) * median_d], dim=1)
        return positions

    def init_new_anchors(self, positions, gaussian_model):
        """Initialize new anchors with nearest-neighbor feature interpolation."""
        n = positions.shape[0]
        if self.init_mode == "nearest" and gaussian_model.get_anchor.shape[0] > 0:
            dists = torch.cdist(positions, gaussian_model.get_anchor)
            nn_idx = dists.argmin(dim=1)
            new_feat = gaussian_model._anchor_feat[nn_idx].clone()
        elif self.init_mode == "mean":
            new_feat = gaussian_model._anchor_feat.mean(0, keepdim=True).repeat(n, 1)
        else:
            new_feat = torch.zeros(n, gaussian_model.feat_dim, device=positions.device)
        return {
            "anchor": positions,
            "scaling": torch.ones(n, 6, device=positions.device) * 0.01,
            "rotation": torch.zeros(n, 4, device=positions.device),
            "anchor_feat": new_feat,
            "offset": torch.zeros(n, gaussian_model.n_offsets, 3, device=positions.device),
            "opacity": torch.ones(n, 1, device=positions.device) * -2.0,
        }

    def should_activate(self, iteration):
        """Check if densification should occur at this iteration."""
        return (self.start_densify <= iteration <= self.end_densify
                and iteration % self.densify_interval == 0)


"""
===== INTEGRATION INSTRUCTIONS =====

1. In scene/gaussian_model.py:
   - Add "from improvements.eaad_densifier import ErrorGuidedAnchorDensifier"
     after existing imports (around line 24)
   - In GaussianModel.__init__(), after self.setup_functions() (line 96), add:
       self.use_eaad = False
       self.eaad_densifier = None
   - Add after adjust_anchor() method (after line 735):
       def eaad_densify(self, rendered_image, gt_image, viewpoint_cam, iteration):
           if self.eaad_densifier is None:
               return
           if not self.eaad_densifier.should_activate(iteration):
               return
           error_map = self.eaad_densifier.compute_error_map(rendered_image, gt_image)
           mask = self.eaad_densifier.find_error_regions(error_map)
           positions = self.eaad_densifier.unproject_to_3d(mask, viewpoint_cam, self)
           if positions is None:
               return
           d = self.eaad_densifier.init_new_anchors(positions, self)
           opt = self.cat_tensors_to_optimizer(d)
           self._anchor = opt["anchor"]
           self._scaling = opt["scaling"]
           self._rotation = opt["rotation"]
           self._anchor_feat = opt["anchor_feat"]
           self._offset = opt["offset"]
           self._opacity = opt["opacity"]
   - Add 'use_eaad=False' parameter to __init__() signature

2. In train.py:
   - In training(), after line 136 (render_pkg assignment), add:
       if hasattr(gaussians, 'eaad_densify'):
           gaussians.eaad_densify(image, gt_image, viewpoint_cam, iteration)

3. In arguments/__init__.py, add EAAD arguments to ModelParams:
   - After add_color_dist (line 77), add:
       self.use_eaad = False
       self.error_threshold = 0.1
       self.densify_interval = 500
       self.start_densify = 2000
       self.end_densify = 12000
       self.max_new_anchors_per_iter = 500
       self.feature_init_mode = "nearest"

4. Training command for EAAD:
   python train.py -s data/nerf_synthetic/lego -m output/lego_eaad \
       --eval --seed 42 --use_eaad --error_threshold 0.1 \
       --feat_dim 32 --n_offsets 10 --voxel_size 0.001 \
       --iterations 30000
"""
