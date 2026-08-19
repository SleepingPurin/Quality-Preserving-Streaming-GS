import math
from dataclasses import dataclass
from random import sample

import torch

from utils.general_utils import quaternion_multiply
from utils.loss_utils import ssim
from utils.rigid_utils import from_homogenous, to_homogenous
from utils.sh_utils import eval_sh

try:
    from diff_gaussian_rasterization_fastgs import (
        GaussianRasterizationSettings as FastGSGaussianRasterizationSettings,
    )
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer as FastGSGaussianRasterizer

    FASTGS_IMPORT_ERROR = None
except ImportError as exc:
    FastGSGaussianRasterizationSettings = None
    FastGSGaussianRasterizer = None
    FASTGS_IMPORT_ERROR = exc


@dataclass(frozen=True)
class FastGSConfig:
    loss_thresh: float = 0.1
    grad_abs_thresh: float = 0.0006
    highfeature_lr: float = 0.005
    lowfeature_lr: float = 0.0025
    grad_thresh: float = 0.0002
    dense: float = 0.001
    mult: float = 0.5
    num_score_cameras: int = 10
    min_opacity: float = 0.005
    final_prune_min_opacity: float = 0.1
    metric_threshold: int = 5
    prune_budget_fraction: float = 0.5

    @classmethod
    def from_optimization_args(cls, args):
        return cls(
            loss_thresh=args.loss_thresh,
            grad_abs_thresh=args.grad_abs_thresh,
            highfeature_lr=args.highfeature_lr,
            lowfeature_lr=args.lowfeature_lr,
            grad_thresh=args.grad_thresh,
            dense=args.dense,
            mult=args.mult,
        )


def ensure_fastgs_available():
    if FastGSGaussianRasterizer is None or FastGSGaussianRasterizationSettings is None:
        raise RuntimeError(
            "FastGS was requested, but diff_gaussian_rasterization_fastgs is not installed. "
            "Please build/install the FastGS rasterizer in the Linux training environment."
        ) from FASTGS_IMPORT_ERROR


def sample_cameras(cameras, max_samples):
    if len(cameras) <= max_samples:
        return list(cameras)
    return sample(list(cameras), max_samples)


def _build_metric_map(image, gt_image, loss_thresh):
    l1_map = torch.abs(image - gt_image).mean(dim=0)
    min_value = torch.amin(l1_map)
    max_value = torch.amax(l1_map)
    range_value = max_value - min_value
    if torch.isclose(range_value, torch.zeros_like(range_value)):
        normalized = torch.zeros_like(l1_map)
    else:
        normalized = (l1_map - min_value) / range_value
    return (normalized > loss_thresh).reshape(-1).to(dtype=torch.int32).contiguous()


def _fastgs_render_impl(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    d_xyz,
    d_rotation,
    d_scaling,
    fastgs_config,
    is_6dof=False,
    scaling_modifier=1.0,
    override_color=None,
    detach_base=False,
    get_flag=False,
    metric_map=None,
):
    ensure_fastgs_available()

    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 4),
        dtype=pc.get_xyz.dtype,
        device=pc.get_xyz.device,
        requires_grad=True,
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = FastGSGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=bool(get_flag),
        metric_map=(
            metric_map
            if metric_map is not None
            else torch.empty(0, dtype=torch.int32, device=pc.get_xyz.device)
        ),
        mult=float(fastgs_config.mult),
    )
    rasterizer = FastGSGaussianRasterizer(raster_settings=raster_settings)

    base_xyz = pc.get_xyz.detach() if detach_base else pc.get_xyz
    base_rotation = pc.get_rotation.detach() if detach_base else pc.get_rotation

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = base_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(base_xyz).unsqueeze(-1)).squeeze(-1)
            )
    else:
        means3D = base_xyz if isinstance(d_xyz, float) else base_xyz + d_xyz

    valid = pc.filter_gaussians(viewpoint_camera, xyz=means3D.detach())
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[valid]
    else:
        if isinstance(d_scaling, float):
            scales = pc.get_scaling
        else:
            scales = pc.modify_scaling(d_scaling)

        if isinstance(d_rotation, float):
            rotations = base_rotation
        else:
            rotations = quaternion_multiply(d_rotation, base_rotation)

    dc = None
    shs = None
    colors_precomp = None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        colors_precomp = torch.clamp_min(
            eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5,
            0.0,
        )[valid]
    else:
        dc = pc._features_dc
        shs = pc._features_rest

    if override_color is not None:
        colors_precomp = override_color[valid]
        dc = None
        shs = None
    else:
        if dc is not None:
            dc = dc[valid]
        if shs is not None:
            shs = shs[valid]
        if scales is not None:
            scales = scales[valid]
        if rotations is not None:
            rotations = rotations[valid]

    rendered_image, radii, depth, accum_metric_counts = rasterizer(
        means3D=means3D[valid],
        means2D=screenspace_points[valid],
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity[valid],
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    visibility_filter = torch.zeros_like(valid, dtype=torch.bool, device=pc.get_xyz.device)
    try:
        visibility_filter[valid] = radii > 0
    except RuntimeError:
        visibility_filter[valid] = True
    radii_full = torch.zeros_like(valid, dtype=torch.int, device=pc.get_xyz.device)
    radii_full[valid] = radii
    result = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "depth_filter": valid,
        "radii": radii_full,
        "depth": depth,
    }
    if get_flag:
        accum_metric_counts_full = torch.zeros_like(valid, dtype=accum_metric_counts.dtype, device=pc.get_xyz.device)
        accum_metric_counts_full[valid] = accum_metric_counts
        result["accum_metric_counts"] = accum_metric_counts_full
    return result


def render_fastgs(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    d_xyz,
    d_rotation,
    d_scaling,
    fastgs_config,
    is_6dof=False,
    scaling_modifier=1.0,
    override_color=None,
    get_flag=False,
    metric_map=None,
):
    return _fastgs_render_impl(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        d_xyz=d_xyz,
        d_rotation=d_rotation,
        d_scaling=d_scaling,
        fastgs_config=fastgs_config,
        is_6dof=is_6dof,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
        detach_base=False,
        get_flag=get_flag,
        metric_map=metric_map,
    )


def render_fastgs_with_detached_base(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    d_xyz,
    d_rotation,
    d_scaling,
    fastgs_config,
    is_6dof=False,
    scaling_modifier=1.0,
    override_color=None,
    get_flag=False,
    metric_map=None,
):
    return _fastgs_render_impl(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        d_xyz=d_xyz,
        d_rotation=d_rotation,
        d_scaling=d_scaling,
        fastgs_config=fastgs_config,
        is_6dof=is_6dof,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
        detach_base=True,
        get_flag=get_flag,
        metric_map=metric_map,
    )


@torch.no_grad()
def compute_gaussian_score_fastgs(
    cameras,
    gaussians,
    pipe,
    bg_color,
    fastgs_config,
    d_xyz,
    d_rotation,
    d_scaling,
    is_6dof=False,
    detach_base=False,
    lambda_dssim=0.2,
    load2gpu_on_the_fly=False,
    densify=True,
):
    full_metric_counts = (
        torch.zeros((gaussians.get_xyz.shape[0]), device=gaussians.get_xyz.device)
        if densify
        else None
    )
    full_metric_score = torch.zeros((gaussians.get_xyz.shape[0]), device=gaussians.get_xyz.device)

    sampled_cameras = sample_cameras(cameras, fastgs_config.num_score_cameras)
    if not sampled_cameras:
        return full_metric_counts, full_metric_score

    render_fn = render_fastgs_with_detached_base if detach_base else render_fastgs

    valid_views = 0
    for camera in sampled_cameras:
        if load2gpu_on_the_fly:
            camera.load2device()
        try:
            gt_image = torch.clamp(camera.original_image, 0.0, 1.0)
            preview_pkg = render_fn(
                camera,
                gaussians,
                pipe,
                bg_color,
                d_xyz,
                d_rotation,
                d_scaling,
                fastgs_config=fastgs_config,
                is_6dof=is_6dof,
                get_flag=False,
            )
            image = torch.clamp(preview_pkg["render"], 0.0, 1.0)
            ll1 = torch.abs(image - gt_image).mean()
            photometric_loss = (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
            metric_map = _build_metric_map(image, gt_image, fastgs_config.loss_thresh)
            render_pkg = render_fn(
                camera,
                gaussians,
                pipe,
                bg_color,
                d_xyz,
                d_rotation,
                d_scaling,
                fastgs_config=fastgs_config,
                is_6dof=is_6dof,
                get_flag=True,
                metric_map=metric_map,
            )
            metric_counts = render_pkg["accum_metric_counts"].to(dtype=full_metric_score.dtype)
            if densify:
                full_metric_counts += metric_counts
            full_metric_score += photometric_loss.detach() * metric_counts
            valid_views += 1
        finally:
            if load2gpu_on_the_fly:
                camera.load2device("cpu")

    if valid_views == 0:
        return full_metric_counts, full_metric_score

    pruning_score = full_metric_score / float(valid_views)
    if torch.count_nonzero(pruning_score).item() > 0:
        prune_min = torch.amin(pruning_score)
        prune_max = torch.amax(pruning_score)
        prune_range = prune_max - prune_min
        if not torch.isclose(prune_range, torch.zeros_like(prune_range)):
            pruning_score = (pruning_score - prune_min) / prune_range
        else:
            pruning_score = torch.zeros_like(pruning_score)

    importance_score = None
    if densify:
        importance_score = torch.div(
            full_metric_counts,
            valid_views,
            rounding_mode="floor",
        )

    return importance_score, pruning_score
