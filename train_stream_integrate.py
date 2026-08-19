import csv
import gc
import json
import math
import os
import time
import uuid
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from random import randint

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import lpips

from arguments import ModelParams, OptimizationParams, PipelineParams
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import render as render_standard
from scene import DeformModel, GaussianModel, Scene
from utils.camera_utils import loadCam
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import get_expon_lr_func, quaternion_multiply, safe_state
from utils.rigid_utils import from_homogenous, to_homogenous
from utils.sh_utils import eval_sh
from utils.system_utils import searchForMaxIteration
from utils.fastgs_backend import (
    FastGSConfig,
    compute_gaussian_score_fastgs,
    ensure_fastgs_available,
    render_fastgs,
    render_fastgs_with_detached_base,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def render_with_detached_base(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    d_xyz,
    d_rotation,
    d_scaling,
    is_6dof=False,
    scaling_modifier=1.0,
    override_color=None,
):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
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
        bwd_depth=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    base_xyz = pc.get_xyz.detach()
    base_rotation = pc.get_rotation.detach()

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = base_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(base_xyz).unsqueeze(-1)).squeeze(-1)
            )
    else:
        means3D = base_xyz if isinstance(d_xyz, float) else base_xyz + d_xyz

    means2D = screenspace_points
    opacity = pc.get_opacity
    valid = pc.filter_gaussians(viewpoint_camera, xyz=means3D.detach())

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

    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(
                eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5,
                0.0,
            )[valid]
        else:
            shs = pc.get_features

    if override_color is not None:
        colors_precomp = override_color[valid]
        shs = None
    else:
        shs = shs[valid]

    rendered_image, radii, depth = rasterizer(
        means3D=means3D[valid],
        means2D=means2D[valid],
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity[valid],
        scales=None if scales is None else scales[valid],
        rotations=None if rotations is None else rotations[valid],
        cov3D_precomp=cov3D_precomp,
    )

    visibility_filter = torch.zeros_like(valid, dtype=torch.bool, device="cuda")
    try:
        visibility_filter[valid] = radii > 0
    except RuntimeError:
        visibility_filter[valid] = 1
    radii_full = torch.zeros_like(valid, dtype=torch.int, device="cuda")
    radii_full[valid] = radii

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "depth_filter": valid,
        "radii": radii_full,
        "depth": depth,
    }


@dataclass
class FrameGroup:
    frame_index: int
    time_value: float
    entries: list
    cameras: list
    view_ids: list
    split_counts: dict


@dataclass
class FrameEntry:
    split: str
    camera_info: object
    view_id: int


@dataclass
class ViewTrack:
    view_index: int
    split_hint: str
    pose_vector: np.ndarray
    frame_indices: list = field(default_factory=list)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def camera_time_value(camera):
    if torch.is_tensor(camera.fid):
        return float(camera.fid.detach().cpu().item())
    return float(camera.fid)


def camera_pose_vector(camera):
    return np.concatenate([np.asarray(camera.R).reshape(-1), np.asarray(camera.T).reshape(-1)], axis=0)


def frame_slug(frame_group):
    time_tag = f"{frame_group.time_value:.6f}".replace(".", "p")
    return f"frame_{frame_group.frame_index:04d}_t_{time_tag}"


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mae_psnr_from_values(psnr_values):
    if len(psnr_values) == 0:
        return float("nan")
    psnr_values = np.asarray(psnr_values, dtype=np.float64)
    mae = np.power(10.0, -psnr_values / 10.0)
    mae_mean = np.mean(mae)
    return float(-10.0 * np.log10(max(mae_mean, 1e-12)))


def mean_metric(records, key):
    valid_values = [record[key] for record in records if not math.isnan(record[key])]
    if not valid_values:
        return float("nan")
    return float(np.mean(valid_values))


def mean_train_seconds_excluding_frame0(frame_summaries):
    valid_values = [
        record["train_seconds"]
        for record in frame_summaries
        if record["frame_index"] != 0 and not math.isnan(record["train_seconds"])
    ]
    if not valid_values:
        return float("nan"), 0
    return float(np.mean(valid_values)), len(valid_values)


def current_cuda_total_used_mb():
    if not torch.cuda.is_available():
        return float("nan"), float("nan"), float("nan")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_mb = total_bytes / (1024.0 ** 2)
    used_mb = (total_bytes - free_bytes) / (1024.0 ** 2)
    used_percent = 100.0 * used_mb / total_mb if total_mb > 0.0 else float("nan")
    return used_mb, used_percent, total_mb


def update_cuda_total_used_peak(peak_mb, peak_percent, total_mb):
    used_mb, used_percent, sampled_total_mb = current_cuda_total_used_mb()
    if math.isfinite(used_mb):
        peak_mb = used_mb if not math.isfinite(peak_mb) else max(peak_mb, used_mb)
    if math.isfinite(used_percent):
        peak_percent = (
            used_percent
            if not math.isfinite(peak_percent)
            else max(peak_percent, used_percent)
        )
    if math.isfinite(sampled_total_mb):
        total_mb = sampled_total_mb
    return peak_mb, peak_percent, total_mb


def summarize_train_gpu_peak_total_used_excluding_frame0(frame_summaries):
    used_values = [
        record["train_gpu_peak_total_used_mb"]
        for record in frame_summaries
        if record["frame_index"] != 0
        and math.isfinite(record.get("train_gpu_peak_total_used_mb", float("nan")))
    ]
    percent_values = [
        record["train_gpu_peak_total_used_percent"]
        for record in frame_summaries
        if record["frame_index"] != 0
        and math.isfinite(record.get("train_gpu_peak_total_used_percent", float("nan")))
    ]
    if not used_values:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    avg_percent = float(np.mean(percent_values)) if percent_values else float("nan")
    max_percent = float(np.max(percent_values)) if percent_values else float("nan")
    return (
        float(np.mean(used_values)),
        float(np.max(used_values)),
        avg_percent,
        max_percent,
        len(used_values),
    )


def aggregate_metric_lists(metric_lists):
    psnr_values = [value for metric_list in metric_lists for value in metric_list["psnr"]]
    ssim_values = [value for metric_list in metric_lists for value in metric_list["ssim"]]
    lpips_values = [value for metric_list in metric_lists for value in metric_list["lpips"]]
    num_images = len(psnr_values)

    if num_images == 0:
        return {
            "num_images": 0,
            "psnr": float("nan"),
            "maepsnr": float("nan"),
            "ssim": float("nan"),
            "lpips": float("nan"),
        }

    return {
        "num_images": num_images,
        "psnr": float(np.mean(psnr_values)),
        "maepsnr": mae_psnr_from_values(psnr_values),
        "ssim": float(np.mean(ssim_values)),
        "lpips": float(np.mean(lpips_values)),
    }


def deformation_regularization_loss(d_xyz, d_rotation, d_scaling):
    total = 0.0
    if torch.is_tensor(d_xyz):
        total = total + d_xyz.square().mean()
    if torch.is_tensor(d_rotation):
        total = total + d_rotation.square().mean()
    if torch.is_tensor(d_scaling):
        total = total + d_scaling.square().mean()
    return total


def identity_quaternion(num_points, device, dtype):
    quat = torch.zeros((num_points, 4), device=device, dtype=dtype)
    quat[:, 0] = 1.0
    return quat


def plot_metric_curves(records, output_path, title, mark_current=False):
    if not records:
        return

    x_values = [record["time"] for record in records]
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    plot_specs = [
        ("PSNR", "psnr", "#1f77b4"),
        ("MAEPSNR", "maepsnr", "#ff7f0e"),
        ("SSIM", "ssim", "#2ca02c"),
        ("LPIPS", "lpips", "#d62728"),
    ]

    for axis, (label, key, color) in zip(axes, plot_specs):
        axis.plot(x_values, [record[key] for record in records], marker="o", color=color, linewidth=2)
        if mark_current:
            axis.axvline(x_values[0], linestyle="--", color="#7f7f7f", linewidth=1)
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Timestamp")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_training_curve(x_values, y_values, output_path, title, ylabel):
    fig, axis = plt.subplots(1, 1, figsize=(10, 4))
    if x_values:
        axis.plot(x_values, y_values, marker="o", linewidth=2)
    else:
        axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(frame_summaries, output_dir):
    if not frame_summaries:
        return

    times = [record["time"] for record in frame_summaries]
    train_gpu_records = [
        record
        for record in frame_summaries
        if record["frame_index"] != 0
        and math.isfinite(record.get("train_gpu_peak_total_used_mb", float("nan")))
    ]
    train_gpu_x = [record["time"] for record in train_gpu_records]
    train_gpu_y = [record["train_gpu_peak_total_used_mb"] for record in train_gpu_records]
    train_time_records = [
        record
        for record in frame_summaries
        if math.isfinite(record.get("train_seconds", float("nan")))
    ]
    train_time_x = [record["time"] for record in train_time_records]
    train_time_y = [record["train_seconds"] for record in train_time_records]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    if train_gpu_records:
        axes[0].plot(train_gpu_x, train_gpu_y, marker="o", linewidth=2)
    axes[0].set_ylabel("MB")
    axes[0].set_title("Training GPU Peak Total Used Memory (Excluding Frame 0)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(times, [record["gaussian_count"] for record in frame_summaries], marker="o", linewidth=2)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-frame Gaussian Count")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    if train_time_records:
        axes[2].plot(train_time_x, train_time_y, marker="o", linewidth=2)
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Per-frame Training Time (Evaluation Excluded)")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_dynamics.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    plot_single_training_curve(
        train_gpu_x,
        train_gpu_y,
        os.path.join(output_dir, "training_gpu_peak_total_used_mb.png"),
        "Training GPU Peak Total Used Memory (Excluding Frame 0)",
        "MB",
    )
    plot_single_training_curve(
        times,
        [record["gaussian_count"] for record in frame_summaries],
        os.path.join(output_dir, "gaussian_count.png"),
        "Per-frame Gaussian Count",
        "Count",
    )
    plot_single_training_curve(
        train_time_x,
        train_time_y,
        os.path.join(output_dir, "training_time.png"),
        "Per-frame Training Time (Evaluation Excluded)",
        "Seconds",
    )


def build_stream_groups(scene_info, round_digits):
    pose_bank = []
    view_tracks = {}
    raw_entries = []

    split_to_cameras = {
        "train": scene_info.train_cameras,
        "val": scene_info.val_cameras,
        "test": scene_info.test_cameras,
    }

    for split_name, camera_infos in split_to_cameras.items():
        for camera_info in camera_infos:
            pose_vector = camera_pose_vector(camera_info)
            view_index = None
            for existing_index, existing_pose in enumerate(pose_bank):
                if np.allclose(pose_vector, existing_pose, atol=1e-6):
                    view_index = existing_index
                    break

            if view_index is None:
                view_index = len(pose_bank)
                pose_bank.append(pose_vector)
                view_tracks[view_index] = ViewTrack(
                    view_index=view_index,
                    split_hint=split_name,
                    pose_vector=pose_vector,
                )

            raw_entries.append(
                {
                    "time": round(camera_time_value(camera_info), round_digits),
                    "split": split_name,
                    "camera_info": camera_info,
                    "view_index": view_index,
                }
            )

    sorted_times = sorted({entry["time"] for entry in raw_entries})
    time_to_frame = {time_value: frame_index for frame_index, time_value in enumerate(sorted_times)}

    frame_buckets = {
        frame_index: {
            "time": time_value,
            "entries": [],
            "split_counts": defaultdict(int),
        }
        for frame_index, time_value in enumerate(sorted_times)
    }

    for entry in raw_entries:
        frame_index = time_to_frame[entry["time"]]
        frame_buckets[frame_index]["entries"].append(
            FrameEntry(
                split=entry["split"],
                camera_info=entry["camera_info"],
                view_id=entry["view_index"],
            )
        )
        frame_buckets[frame_index]["split_counts"][entry["split"]] += 1
        view_tracks[entry["view_index"]].frame_indices.append(frame_index)

    frame_groups = []
    for frame_index in range(len(sorted_times)):
        frame_bucket = frame_buckets[frame_index]
        entries = sorted(
            frame_bucket["entries"],
            key=lambda item: item.view_id,
        )
        view_ids = [entry.view_id for entry in entries]
        frame_groups.append(
            FrameGroup(
                frame_index=frame_index,
                time_value=frame_bucket["time"],
                entries=entries,
                cameras=[],
                view_ids=view_ids,
                split_counts=dict(frame_bucket["split_counts"]),
            )
        )

    return frame_groups, view_tracks


class IntegratedVelocityStreamTrainer:
    def __init__(self, args, dataset, opt, pipe):
        self.args = args
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.use_fastgs = args.fastgs
        self.fastgs_config = FastGSConfig.from_optimization_args(opt) if self.use_fastgs else None
        if self.use_fastgs:
            ensure_fastgs_available()
        self.tb_writer = prepare_output_and_logger(args)
        self.dataset.model_path = args.model_path
        self.lpips_fn = lpips.LPIPS(net="vgg").to(torch.device("cuda:0"))
        self.lpips_fn.eval()

        self.gaussians = GaussianModel(dataset.sh_degree)
        self.deform = DeformModel(
            max_time=dataset.max_time,
            vel_start_time=args.vel_start_time,
            light=dataset.light,
            physics_code=dataset.physics_code,
            freegave=dataset.freegave,
        )
        self.deform.train_setting(opt)

        self.scene = Scene(
            dataset,
            self.gaussians,
            shuffle=False,
            skip_val=False,
            skip_test=False,
            load_cameras=False,
            lazy_image_loading=True,
        )
        self.frame0_static_model_path = args.frame0_static_model_path
        self.uses_external_frame0_result = bool(self.frame0_static_model_path)
        self.loaded_frame0_static_ply = None
        if self.uses_external_frame0_result:
            self.loaded_frame0_static_ply = self.load_frame0_static_model(self.frame0_static_model_path)
        self.gaussians.training_setup(opt, fastgs_args=self.fastgs_config)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.frame_iterations = args.frame_iterations
        self.init_frame_iterations = (
            args.frame0_iterations if args.frame0_iterations > 0 else args.frame_iterations
        )
        self.future_frame_window = args.future_frame_window
        self.frame_densification_interval = (
            opt.densification_interval if args.frame_densification_interval <= 0 else args.frame_densification_interval
        )
        self.frame_densify_from_iter = (
            opt.densify_from_iter if args.frame_densify_from_iter < 0 else args.frame_densify_from_iter
        )
        self.frame_densify_until_iter = (
            opt.densify_until_iter if args.frame_densify_until_iter < 0 else args.frame_densify_until_iter
        )
        self.frame0_densification_interval = (
            self.frame_densification_interval
            if args.frame0_densification_interval <= 0
            else args.frame0_densification_interval
        )
        self.frame0_densify_from_iter = (
            self.frame_densify_from_iter
            if args.frame0_densify_from_iter < 0
            else args.frame0_densify_from_iter
        )
        self.frame0_densify_until_iter = (
            self.frame_densify_until_iter
            if args.frame0_densify_until_iter < 0
            else args.frame0_densify_until_iter
        )
        self.frame_min_opacity = args.frame_min_opacity
        self.frame0_min_opacity = args.frame0_min_opacity
        self.vel_fixed_lr = args.vel_fixed_lr
        self.frame_opacity_reset_interval = (
            opt.opacity_reset_interval
            if args.frame_opacity_reset_interval <= 0
            else args.frame_opacity_reset_interval
        )
        self.video_fps = dataset.fps if args.video_fps <= 0 else args.video_fps
        self.video_error_clip = max(args.video_error_clip, 1e-6)
        self.error_colormap = plt.get_cmap("turbo")
        self.disable_visual_outputs = args.disable_visual_outputs
        self.train_time_cutoff = args.train_time_cutoff
        self.lambda_deformation_reg = args.lambda_deformation_reg

        self.global_step = 1
        self.frame_groups, self.view_tracks = build_stream_groups(self.scene.scene_info, args.time_round_digits)
        self.train_frame_groups = [
            frame_group for frame_group in self.frame_groups if frame_group.time_value <= self.train_time_cutoff + 1e-8
        ]
        if not self.train_frame_groups:
            raise ValueError("No frame groups fall at or before --train_time_cutoff.")
        self.last_train_frame_index = self.train_frame_groups[-1].frame_index
        self.post_cutoff_frame_groups = [
            frame_group for frame_group in self.frame_groups if frame_group.frame_index > self.last_train_frame_index
        ]
        self.current_base_time = self.frame_groups[0].time_value if self.frame_groups else 0.0
        self.frame_summaries = []
        self.current_val_history = []
        self.current_train_val_history = []
        self.future10_all_history = []
        self.current_val_metric_lists = []
        self.current_train_val_metric_lists = []
        self.future10_all_metric_lists = []

        self.stream_dir = os.path.join(args.model_path, "stream")
        self.frame_report_dir = os.path.join(self.stream_dir, "frame_reports")
        self.plot_dir = os.path.join(self.stream_dir, "plots")
        self.video_dir = os.path.join(self.stream_dir, "videos")
        output_dirs = [self.stream_dir, self.frame_report_dir, self.plot_dir]
        if not self.disable_visual_outputs:
            output_dirs.append(self.video_dir)
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)

        self.write_stream_metadata()

    def resolve_frame0_static_ply_path(self, model_path):
        if os.path.isfile(model_path):
            if model_path.endswith(".ply"):
                return model_path
            raise ValueError(
                "--frame0_static_model_path points to a file, but it is not a .ply file: "
                f"{model_path}"
            )

        if not os.path.isdir(model_path):
            raise ValueError(
                "--frame0_static_model_path must be a .ply file or a model directory: "
                f"{model_path}"
            )

        direct_ply = os.path.join(model_path, "point_cloud.ply")
        if os.path.isfile(direct_ply):
            return direct_ply

        point_cloud_root = os.path.join(model_path, "point_cloud")
        if os.path.isdir(point_cloud_root):
            try:
                loaded_iter = searchForMaxIteration(point_cloud_root)
                candidate = os.path.join(point_cloud_root, f"iteration_{loaded_iter}", "point_cloud.ply")
                if os.path.isfile(candidate):
                    return candidate
            except Exception:
                pass

        raise ValueError(
            "Could not resolve a static Gaussian point cloud from --frame0_static_model_path: "
            f"{model_path}"
        )

    def load_frame0_static_model(self, model_path):
        resolved_path = self.resolve_frame0_static_ply_path(model_path)
        print(f"Loading external frame-0 static Gaussian model from: {resolved_path}")
        self.gaussians.load_ply(
            resolved_path,
            og_number_points=len(self.scene.scene_info.point_cloud.points),
        )
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")
        return resolved_path

    def get_frame_iterations(self, frame_index):
        return self.init_frame_iterations if frame_index == 0 else self.frame_iterations

    def get_frame_densification_interval(self, frame_index):
        return self.frame0_densification_interval if frame_index == 0 else self.frame_densification_interval

    def get_frame_densify_from_iter(self, frame_index):
        return self.frame0_densify_from_iter if frame_index == 0 else self.frame_densify_from_iter

    def get_frame_densify_until_iter(self, frame_index):
        return self.frame0_densify_until_iter if frame_index == 0 else self.frame_densify_until_iter

    def get_frame_min_opacity(self, frame_index):
        return self.frame0_min_opacity if frame_index == 0 else self.frame_min_opacity

    def update_motion_learning_rate(self, iteration):
        if self.vel_fixed_lr > 0.0:
            for param_group in self.deform.optimizer.param_groups:
                if param_group["name"] == "deform":
                    param_group["lr"] = self.vel_fixed_lr
            return self.vel_fixed_lr
        return self.deform.update_learning_rate(iteration)

    @torch.no_grad()
    def compute_fastgs_scores(self, cameras, frame_index, deformation, densify=True):
        if not self.use_fastgs:
            return None, None
        d_xyz, d_rotation, d_scaling = deformation
        return compute_gaussian_score_fastgs(
            cameras=cameras,
            gaussians=self.gaussians,
            pipe=self.pipe,
            bg_color=self.background,
            fastgs_config=self.fastgs_config,
            d_xyz=d_xyz,
            d_rotation=d_rotation,
            d_scaling=d_scaling,
            is_6dof=self.dataset.is_6dof,
            detach_base=frame_index > 0,
            lambda_dssim=self.opt.lambda_dssim,
            load2gpu_on_the_fly=self.dataset.load2gpu_on_the_fly,
            densify=densify,
        )

    def reset_frame_learning_schedules(self, frame_index):
        frame_max_steps = max(self.get_frame_iterations(frame_index), 1)

        self.gaussians.xyz_scheduler_args = get_expon_lr_func(
            lr_init=self.opt.position_lr_init * self.gaussians.spatial_lr_scale,
            lr_final=self.opt.position_lr_final * self.gaussians.spatial_lr_scale,
            lr_delay_mult=self.opt.position_lr_delay_mult,
            max_steps=frame_max_steps,
        )
        self.deform.deform_scheduler_args = get_expon_lr_func(
            lr_init=self.opt.position_lr_init * self.deform.spatial_lr_scale,
            lr_final=self.opt.position_lr_final,
            lr_delay_mult=self.opt.position_lr_delay_mult,
            max_steps=frame_max_steps,
        )

        # Reset scheduled parameter groups to the start of the per-frame schedule.
        self.gaussians.update_learning_rate(0)
        self.update_motion_learning_rate(0)

    def write_stream_metadata(self):
        frame_index_records = []
        for frame_group in self.frame_groups:
            frame_index_records.append(
                {
                    "frame_index": frame_group.frame_index,
                    "time": frame_group.time_value,
                    "num_views": len(frame_group.entries),
                    "split_counts": frame_group.split_counts,
                    "view_ids": frame_group.view_ids,
                }
            )

        view_index_records = []
        for view_track in self.view_tracks.values():
            view_index_records.append(
                {
                    "view_index": view_track.view_index,
                    "split_hint": view_track.split_hint,
                    "frame_indices": sorted(view_track.frame_indices),
                }
            )

        payload = {
            "num_frames": len(self.frame_groups),
            "num_train_frames": len(self.train_frame_groups),
            "num_post_cutoff_frames": len(self.post_cutoff_frame_groups),
            "num_views": len(self.view_tracks),
            "train_time_cutoff": self.train_time_cutoff,
            "last_train_frame_index": self.last_train_frame_index,
            "frames": frame_index_records,
            "views": view_index_records,
        }
        save_json(os.path.join(self.stream_dir, "stream_index.json"), payload)

    def subset_frame_group(self, frame_group, allowed_splits):
        entries = [entry for entry in frame_group.entries if entry.split in allowed_splits]
        return FrameGroup(
            frame_index=frame_group.frame_index,
            time_value=frame_group.time_value,
            entries=entries,
            cameras=[],
            view_ids=[entry.view_id for entry in entries],
            split_counts=dict(Counter(entry.split for entry in entries)) if entries else {},
        )

    def materialize_frame_group(self, frame_group):
        entries = list(frame_group.entries)
        cameras = [loadCam(self.dataset, entry.view_id, entry.camera_info, 1.0) for entry in entries]
        return FrameGroup(
            frame_index=frame_group.frame_index,
            time_value=frame_group.time_value,
            entries=entries,
            cameras=cameras,
            view_ids=[entry.view_id for entry in entries],
            split_counts=dict(frame_group.split_counts),
        )

    def combine_frame_groups(self, frame_group, materialized_groups):
        cameras = []
        for group in materialized_groups:
            cameras.extend(group.cameras)
        return FrameGroup(
            frame_index=frame_group.frame_index,
            time_value=frame_group.time_value,
            entries=list(frame_group.entries),
            cameras=cameras,
            view_ids=list(frame_group.view_ids),
            split_counts=dict(frame_group.split_counts),
        )

    def dispose_frame_group(self, frame_group):
        for camera in frame_group.cameras:
            self.release_camera(camera)
            camera.original_image = None
            camera.gt_alpha_mask = None
            camera.world_view_transform = None
            camera.projection_matrix = None
            camera.full_proj_transform = None
            camera.camera_center = None
            camera.fid = None
            camera.depth = None
        frame_group.cameras.clear()

    def get_pattern_weights(self, base_xyz):
        if self.dataset.freegave:
            deform_code = self.deform.code_field(base_xyz.detach())
            return self.deform.code_field.seg(deform_code)
        return None

    def identity_state(self, time_value):
        base_xyz = self.gaussians.get_xyz.detach()
        num_points = base_xyz.shape[0]
        device = base_xyz.device
        dtype = base_xyz.dtype
        return {
            "base_xyz": base_xyz,
            "xyz": base_xyz,
            "time_tensor": torch.full((num_points, 1), time_value, device=device, dtype=dtype),
            "d_xyz": torch.zeros_like(base_xyz),
            "d_rotation": identity_quaternion(num_points, device, dtype),
            "d_scaling": torch.zeros_like(self.gaussians.get_scaling),
            "pattern_weights": self.get_pattern_weights(base_xyz),
        }

    def predict_state_from_current_base(self, target_time):
        base_xyz = self.gaussians.get_xyz.detach()
        num_points = base_xyz.shape[0]
        device = base_xyz.device
        dtype = base_xyz.dtype
        target_time_tensor = torch.full((num_points, 1), target_time, device=device, dtype=dtype)

        if abs(target_time - self.current_base_time) < 1e-8:
            return {
                "base_xyz": base_xyz,
                "xyz": base_xyz,
                "time_tensor": target_time_tensor,
                "d_xyz": torch.zeros_like(base_xyz),
                "d_rotation": identity_quaternion(num_points, device, dtype),
                "d_scaling": torch.zeros_like(self.gaussians.get_scaling),
                "pattern_weights": self.get_pattern_weights(base_xyz),
            }

        current_time_tensor = torch.full((num_points, 1), self.current_base_time, device=device, dtype=dtype)
        pattern_weights = self.get_pattern_weights(base_xyz)
        xyz_next, rotation_delta = self.deform.vel.integrate_pos(
            pattern_weights,
            base_xyz,
            current_time_tensor,
            target_time_tensor,
            1 / self.dataset.fps,
            rot=True,
        )
        return {
            "base_xyz": base_xyz,
            "xyz": xyz_next,
            "time_tensor": target_time_tensor,
            "d_xyz": xyz_next - base_xyz,
            "d_rotation": rotation_delta,
            "d_scaling": torch.zeros_like(self.gaussians.get_scaling),
            "pattern_weights": pattern_weights,
        }

    def frame_uses_trd_prediction(self, frame_group):
        return frame_group.frame_index > 0

    @torch.no_grad()
    def build_frame_state(self, frame_group):
        if abs(frame_group.time_value - self.current_base_time) < 1e-8:
            return self.identity_state(frame_group.time_value)

        predicted_state = self.predict_state_from_current_base(frame_group.time_value)
        return {
            "base_xyz": predicted_state["base_xyz"].detach(),
            "xyz": predicted_state["xyz"].detach(),
            "time_tensor": predicted_state["time_tensor"].detach(),
            "d_xyz": predicted_state["d_xyz"].detach(),
            "d_rotation": predicted_state["d_rotation"].detach(),
            "d_scaling": predicted_state["d_scaling"].detach(),
            "pattern_weights": predicted_state["pattern_weights"],
        }

    @torch.no_grad()
    def iter_future_states(self, current_state, future_frame_groups):
        prev_xyz = current_state["xyz"].detach()
        prev_time = current_state["time_tensor"].detach()
        prev_rotation = current_state["d_rotation"].detach()
        prev_scaling = current_state["d_scaling"].detach()
        base_xyz = current_state["base_xyz"].detach()
        pattern_weights = current_state["pattern_weights"]

        for frame_group in future_frame_groups:
            target_time = torch.full_like(prev_time, frame_group.time_value)
            xyz_next, rotation_delta = self.deform.vel.integrate_pos(
                pattern_weights,
                prev_xyz,
                prev_time,
                target_time,
                1 / self.dataset.fps,
                rot=True,
            )
            rotation_next = quaternion_multiply(rotation_delta, prev_rotation)
            state = {
                "base_xyz": base_xyz,
                "xyz": xyz_next,
                "time_tensor": target_time,
                "d_xyz": xyz_next - base_xyz,
                "d_rotation": rotation_next,
                "d_scaling": prev_scaling,
                "pattern_weights": pattern_weights,
            }
            yield frame_group, state
            prev_xyz = xyz_next.detach()
            prev_time = target_time.detach()
            prev_rotation = rotation_next.detach()

    def compute_deformation(self, camera, frame_index):
        target_time = camera_time_value(camera)
        if abs(target_time - self.current_base_time) < 1e-8:
            return 0.0, 0.0, 0.0

        predicted_state = self.predict_state_from_current_base(target_time)
        return (
            predicted_state["d_xyz"],
            predicted_state["d_rotation"],
            predicted_state["d_scaling"],
        )

    def render_camera(self, camera, frame_index, deformation=None, return_deformation=False):
        if self.dataset.load2gpu_on_the_fly:
            camera.load2device()

        if deformation is None:
            d_xyz, d_rotation, d_scaling = self.compute_deformation(camera, frame_index)
        else:
            d_xyz, d_rotation, d_scaling = deformation

        if self.use_fastgs:
            if frame_index == 0:
                render_pkg = render_fastgs(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    d_xyz,
                    d_rotation,
                    d_scaling,
                    fastgs_config=self.fastgs_config,
                    is_6dof=self.dataset.is_6dof,
                )
            else:
                render_pkg = render_fastgs_with_detached_base(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    d_xyz,
                    d_rotation,
                    d_scaling,
                    fastgs_config=self.fastgs_config,
                    is_6dof=self.dataset.is_6dof,
                )
        else:
            if frame_index == 0:
                render_pkg = render_standard(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    d_xyz,
                    d_rotation,
                    d_scaling,
                    self.dataset.is_6dof,
                )
            else:
                render_pkg = render_with_detached_base(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    d_xyz,
                    d_rotation,
                    d_scaling,
                    self.dataset.is_6dof,
                )
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(camera.original_image, 0.0, 1.0)
        if return_deformation:
            return render_pkg, image, gt_image, (d_xyz, d_rotation, d_scaling)
        return render_pkg, image, gt_image

    def release_camera(self, camera):
        if self.dataset.load2gpu_on_the_fly:
            camera.load2device("cpu")

    @torch.no_grad()
    def advance_gaussians_to_time(self, target_time):
        if abs(target_time - self.current_base_time) < 1e-8:
            self.current_base_time = target_time
            return

        predicted_state = self.predict_state_from_current_base(target_time)
        new_xyz = predicted_state["xyz"].detach()
        new_rotation = quaternion_multiply(
            predicted_state["d_rotation"].detach(),
            self.gaussians.get_rotation.detach(),
        )

        self.gaussians._xyz.data.copy_(new_xyz)
        self.gaussians._rotation.data.copy_(new_rotation)
        self.current_base_time = target_time

    def train_single_frame(self, frame_group, train_group, train_order_index, num_train_frames):
        viewpoint_stack = []
        losses = []
        recon_losses = []
        deform_reg_losses = []
        frame_iterations = self.get_frame_iterations(frame_group.frame_index)
        frame_densification_interval = self.get_frame_densification_interval(frame_group.frame_index)
        frame_densify_from_iter = self.get_frame_densify_from_iter(frame_group.frame_index)
        frame_densify_until_iter = self.get_frame_densify_until_iter(frame_group.frame_index)
        use_velocity_training = (
            self.frame_uses_trd_prediction(frame_group)
            and frame_group.time_value > self.current_base_time + 1e-8
        )
        use_original_reset = frame_group.frame_index == 0
        self.reset_frame_learning_schedules(frame_group.frame_index)

        record_train_gpu_peak = frame_group.frame_index != 0
        torch.cuda.synchronize()
        if record_train_gpu_peak:
            torch.cuda.reset_peak_memory_stats()
        train_gpu_peak_total_used_mb = float("nan")
        train_gpu_peak_total_used_percent = float("nan")
        train_gpu_total_memory_mb = float("nan")
        if record_train_gpu_peak:
            (
                train_gpu_peak_total_used_mb,
                train_gpu_peak_total_used_percent,
                train_gpu_total_memory_mb,
            ) = update_cuda_total_used_peak(
                train_gpu_peak_total_used_mb,
                train_gpu_peak_total_used_percent,
                train_gpu_total_memory_mb,
            )
        frame_start = time.perf_counter()

        progress = tqdm.tqdm(
            range(1, frame_iterations + 1),
            desc=f"Frame {train_order_index + 1}/{num_train_frames} @ t={frame_group.time_value:.6f}",
        )

        for local_step in progress:
            if self.global_step % 1000 == 0:
                self.gaussians.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = train_group.cameras.copy()

            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            if use_velocity_training:
                predicted_state = self.predict_state_from_current_base(frame_group.time_value)
                deformation = (
                    predicted_state["d_xyz"],
                    predicted_state["d_rotation"],
                    predicted_state["d_scaling"],
                )
            else:
                deformation = (0.0, 0.0, 0.0)
            render_pkg, image, gt_image, deformation = self.render_camera(
                viewpoint_cam,
                frame_group.frame_index,
                deformation=deformation,
                return_deformation=True,
            )

            ll1 = l1_loss(image, gt_image)
            recon_loss = (1.0 - self.opt.lambda_dssim) * ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            if use_velocity_training:
                deform_reg_loss = self.lambda_deformation_reg * deformation_regularization_loss(*deformation)
            else:
                deform_reg_loss = 0.0
            loss = recon_loss + deform_reg_loss
            loss.backward()
            if record_train_gpu_peak:
                (
                    train_gpu_peak_total_used_mb,
                    train_gpu_peak_total_used_percent,
                    train_gpu_total_memory_mb,
                ) = update_cuda_total_used_peak(
                    train_gpu_peak_total_used_mb,
                    train_gpu_peak_total_used_percent,
                    train_gpu_total_memory_mb,
                )
            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            deform_reg_losses.append(
                deform_reg_loss.item() if torch.is_tensor(deform_reg_loss) else float(deform_reg_loss)
            )

            self.release_camera(viewpoint_cam)

            with torch.no_grad():
                visibility_filter = render_pkg["visibility_filter"]
                viewspace_points = render_pkg["viewspace_points"]
                radii = render_pkg["radii"]

                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                if local_step < frame_densify_until_iter:
                    self.gaussians.add_densification_stats(viewspace_points, visibility_filter)

                    if (
                        local_step > frame_densify_from_iter
                        and local_step % frame_densification_interval == 0
                    ):
                        size_threshold = (
                            20 if local_step > self.frame_opacity_reset_interval else None
                        )
                        if self.use_fastgs:
                            importance_score, pruning_score = self.compute_fastgs_scores(
                                train_group.cameras,
                                frame_group.frame_index,
                                deformation,
                            )
                            self.gaussians.densify_and_prune_fastgs(
                                size_threshold,
                                self.fastgs_config.min_opacity,
                                self.scene.cameras_extent,
                                radii,
                                self.fastgs_config,
                                importance_score,
                                pruning_score,
                            )
                        else:
                            self.gaussians.densify_and_prune(
                                self.opt.densify_grad_threshold,
                                self.get_frame_min_opacity(frame_group.frame_index),
                                self.scene.cameras_extent,
                                size_threshold,
                            )

                    if use_original_reset and (
                        local_step % self.frame_opacity_reset_interval == 0
                        or (self.dataset.white_background and local_step == frame_densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                if (
                    self.use_fastgs
                    and local_step % 3000 == 0
                    and local_step > 15_000
                    and local_step < 30_000
                ):
                    _, pruning_score = self.compute_fastgs_scores(
                        train_group.cameras,
                        frame_group.frame_index,
                        deformation,
                        densify=False,
                    )
                    self.gaussians.final_prune_fastgs(
                        self.fastgs_config.final_prune_min_opacity,
                        pruning_score,
                    )

                self.gaussians.update_learning_rate(local_step)
                if use_velocity_training:
                    self.update_motion_learning_rate(local_step)

                if self.use_fastgs:
                    self.gaussians.optimizer_step(self.global_step)
                    if use_velocity_training:
                        self.deform.optimizer.step()
                        self.deform.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.deform.optimizer.zero_grad(set_to_none=True)
                else:
                    self.gaussians.optimizer.step()
                    if use_velocity_training:
                        self.deform.optimizer.step()

                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.deform.optimizer.zero_grad(set_to_none=True)

            if record_train_gpu_peak:
                (
                    train_gpu_peak_total_used_mb,
                    train_gpu_peak_total_used_percent,
                    train_gpu_total_memory_mb,
                ) = update_cuda_total_used_peak(
                    train_gpu_peak_total_used_mb,
                    train_gpu_peak_total_used_percent,
                    train_gpu_total_memory_mb,
                )

            if self.tb_writer:
                self.tb_writer.add_scalar("stream/train_loss", loss.item(), self.global_step)
                self.tb_writer.add_scalar("stream/train_recon_loss", recon_loss.item(), self.global_step)
                self.tb_writer.add_scalar("stream/train_deformation_reg_loss", deform_reg_losses[-1], self.global_step)
                self.tb_writer.add_scalar("stream/train_l1", ll1.item(), self.global_step)
                self.tb_writer.add_scalar("stream/frame_index", frame_group.frame_index, self.global_step)
                self.tb_writer.add_scalar("stream/frame_time", frame_group.time_value, self.global_step)
                self.tb_writer.add_scalar("stream/gaussian_count", self.gaussians.get_xyz.shape[0], self.global_step)

            progress.set_postfix(
                loss=f"{loss.item():.5f}",
                deform_reg=f"{deform_reg_losses[-1]:.5f}",
                points=int(self.gaussians.get_xyz.shape[0]),
                gstep=self.global_step,
            )
            self.global_step += 1

        if use_velocity_training:
            self.advance_gaussians_to_time(frame_group.time_value)
        else:
            self.current_base_time = frame_group.time_value

        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - frame_start
        if record_train_gpu_peak:
            (
                train_gpu_peak_total_used_mb,
                train_gpu_peak_total_used_percent,
                train_gpu_total_memory_mb,
            ) = update_cuda_total_used_peak(
                train_gpu_peak_total_used_mb,
                train_gpu_peak_total_used_percent,
                train_gpu_total_memory_mb,
            )
        train_gpu_peak_allocated_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            if record_train_gpu_peak
            else float("nan")
        )
        train_gpu_peak_reserved_mb = (
            torch.cuda.max_memory_reserved() / (1024.0 ** 2)
            if record_train_gpu_peak
            else float("nan")
        )

        return {
            "frame_index": frame_group.frame_index,
            "time": frame_group.time_value,
            "train_iterations": frame_iterations,
            "train_num_views": len(train_group.cameras),
            "train_seconds": train_seconds,
            "train_gpu_peak_total_used_mb": train_gpu_peak_total_used_mb,
            "train_gpu_peak_total_used_percent": train_gpu_peak_total_used_percent,
            "train_gpu_total_memory_mb": train_gpu_total_memory_mb,
            "train_gpu_peak_reserved_mb": train_gpu_peak_reserved_mb,
            "train_gpu_peak_allocated_mb": train_gpu_peak_allocated_mb,
            "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
            "mean_loss": float(np.mean(losses)),
            "mean_recon_loss": float(np.mean(recon_losses)),
            "mean_deformation_reg_loss": float(np.mean(deform_reg_losses)),
            "last_loss": float(losses[-1]),
            "last_recon_loss": float(recon_losses[-1]),
            "last_deformation_reg_loss": float(deform_reg_losses[-1]),
            "global_step_end": self.global_step - 1,
        }

    def build_loaded_frame0_summary(self, frame_group, train_group):
        self.current_base_time = frame_group.time_value
        return {
            "frame_index": frame_group.frame_index,
            "time": frame_group.time_value,
            "train_iterations": 0,
            "train_num_views": len(train_group.cameras),
            "train_seconds": 0.0,
            "train_gpu_peak_total_used_mb": float("nan"),
            "train_gpu_peak_total_used_percent": float("nan"),
            "train_gpu_total_memory_mb": float("nan"),
            "train_gpu_peak_reserved_mb": float("nan"),
            "train_gpu_peak_allocated_mb": float("nan"),
            "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
            "mean_loss": 0.0,
            "mean_recon_loss": 0.0,
            "mean_deformation_reg_loss": 0.0,
            "last_loss": 0.0,
            "last_recon_loss": 0.0,
            "last_deformation_reg_loss": 0.0,
            "global_step_end": self.global_step - 1,
        }

    def evaluate_frame(
        self,
        frame_group,
        video_writers=None,
        per_view_history=None,
        panel_dir=None,
        deformation=None,
        return_value_lists=False,
    ):
        lpips_values = []
        psnr_values = []
        ssim_values = []

        if panel_dir is not None:
            os.makedirs(panel_dir, exist_ok=True)

        if len(frame_group.cameras) == 0:
            record = {
                "frame_index": frame_group.frame_index,
                "time": frame_group.time_value,
                "num_views": 0,
                "psnr": float("nan"),
                "maepsnr": float("nan"),
                "ssim": float("nan"),
                "lpips": float("nan"),
            }
            metric_lists = {"psnr": [], "ssim": [], "lpips": []}
            if return_value_lists:
                return record, metric_lists
            return record

        for camera, view_id in zip(frame_group.cameras, frame_group.view_ids):
            _, image, gt_image = self.render_camera(camera, frame_group.frame_index, deformation=deformation)

            psnr_value = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()
            ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            lpips_value = self.lpips_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)

            if per_view_history is not None:
                per_view_history[view_id].append(
                    {
                        "frame_index": frame_group.frame_index,
                        "time": frame_group.time_value,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                        "lpips": lpips_value,
                    }
                )

            if video_writers is not None:
                video_writers[view_id].append_data(self.make_comparison_panel(image, gt_image))

            if panel_dir is not None:
                split_hint = self.view_tracks[view_id].split_hint
                image_name = camera.image_name.replace(os.sep, "_")
                panel_path = os.path.join(
                    panel_dir,
                    f"view_{view_id:03d}_{split_hint}_{image_name}.png",
                )
                imageio.imwrite(panel_path, self.make_comparison_panel(image, gt_image))

            self.release_camera(camera)

        record = {
            "frame_index": frame_group.frame_index,
            "time": frame_group.time_value,
            "num_views": len(frame_group.cameras),
            "psnr": float(np.mean(psnr_values)),
            "maepsnr": mae_psnr_from_values(psnr_values),
            "ssim": float(np.mean(ssim_values)),
            "lpips": float(np.mean(lpips_values)),
        }
        metric_lists = {
            "psnr": psnr_values,
            "ssim": ssim_values,
            "lpips": lpips_values,
        }
        if return_value_lists:
            return record, metric_lists
        return record

    def make_comparison_panel(self, image, gt_image):
        render_np = image.detach().cpu().permute(1, 2, 0).numpy()
        gt_np = gt_image.detach().cpu().permute(1, 2, 0).numpy()
        error_map = np.abs(render_np - gt_np).mean(axis=2)
        error_map = np.clip(error_map / self.video_error_clip, 0.0, 1.0)
        heatmap = self.error_colormap(error_map)[..., :3]

        panel = np.concatenate([gt_np, render_np, heatmap], axis=1)
        panel = np.clip(panel, 0.0, 1.0)
        return (panel * 255.0).astype(np.uint8)

    def save_frame_checkpoint(self, frame_group):
        iteration = frame_group.frame_index
        self.scene.save(iteration)
        out_weights_path = os.path.join(self.args.model_path, "deform", f"iteration_{iteration}")
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.vel.state_dict(), os.path.join(out_weights_path, "vel.pth"))
        if self.dataset.freegave:
            torch.save(self.deform.code_field.state_dict(), os.path.join(out_weights_path, "code_field.pth"))

    def evaluate_future_window(self, start_frame_index, current_state, report_dir):
        end_frame_index = min(start_frame_index + self.future_frame_window, len(self.frame_groups) - 1)
        records = []
        metric_lists = []

        future_states = self.iter_future_states(
            current_state,
            self.frame_groups[start_frame_index + 1:end_frame_index + 1],
        )
        for future_frame_group, future_state in future_states:
            materialized_group = self.materialize_frame_group(future_frame_group)
            try:
                future_record, future_metric_lists = self.evaluate_frame(
                    materialized_group,
                    deformation=(
                        future_state["d_xyz"],
                        future_state["d_rotation"],
                        future_state["d_scaling"],
                    ),
                    return_value_lists=True,
                )
            finally:
                self.dispose_frame_group(materialized_group)
                gc.collect()
            records.append(future_record)
            metric_lists.append(future_metric_lists)

        plot_metric_curves(
            records,
            os.path.join(report_dir, "future_window_metrics.png"),
            title=f"Frame {start_frame_index} Future {self.future_frame_window} Frames (All Views)",
        )
        save_json(os.path.join(report_dir, "future_window_metrics.json"), records)
        save_csv(
            os.path.join(report_dir, "future_window_metrics.csv"),
            records,
            fieldnames=["frame_index", "time", "num_views", "psnr", "maepsnr", "ssim", "lpips"],
        )
        return records, metric_lists

    def save_frame_summaries(self):
        save_json(os.path.join(self.stream_dir, "frame_summaries.json"), self.frame_summaries)
        save_csv(
            os.path.join(self.stream_dir, "frame_summaries.csv"),
            self.frame_summaries,
            fieldnames=[
                "frame_index",
                "time",
                "train_iterations",
                "train_num_views",
                "train_seconds",
                "train_gpu_peak_total_used_mb",
                "train_gpu_peak_total_used_percent",
                "train_gpu_total_memory_mb",
                "train_gpu_peak_reserved_mb",
                "train_gpu_peak_allocated_mb",
                "gaussian_count",
                "mean_loss",
                "mean_recon_loss",
                "mean_deformation_reg_loss",
                "last_loss",
                "last_recon_loss",
                "last_deformation_reg_loss",
                "global_step_end",
                "current_val_num_views",
                "current_val_psnr",
                "current_val_maepsnr",
                "current_val_ssim",
                "current_val_lpips",
                "current_train_val_num_views",
                "current_train_val_psnr",
                "current_train_val_maepsnr",
                "current_train_val_ssim",
                "current_train_val_lpips",
                "future10_num_frames",
                "future10_num_views",
                "future10_all_psnr",
                "future10_all_maepsnr",
                "future10_all_ssim",
                "future10_all_lpips",
            ],
        )

    def save_metric_history(self, filename_stem, records):
        output_base = os.path.join(self.stream_dir, filename_stem)
        save_json(output_base + ".json", records)
        save_csv(
            output_base + ".csv",
            records,
            fieldnames=["frame_index", "time", "num_views", "psnr", "maepsnr", "ssim", "lpips"],
        )

    def summarize_segment(self, records, metric_lists):
        metrics = aggregate_metric_lists(metric_lists)
        if records:
            time_start = records[0]["time"]
            time_end = records[-1]["time"]
        else:
            time_start = float("nan")
            time_end = float("nan")
        return {
            "num_frames": len(records),
            "num_images": metrics["num_images"],
            "time_start": time_start,
            "time_end": time_end,
            "psnr": metrics["psnr"],
            "maepsnr": metrics["maepsnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
        }

    def create_video_writers(self):
        if self.disable_visual_outputs:
            return {}, {}
        os.makedirs(self.video_dir, exist_ok=True)
        video_writers = {}
        video_paths = {}
        for view_index, view_track in self.view_tracks.items():
            video_name = f"view_{view_index:03d}_{view_track.split_hint}.mp4"
            video_path = os.path.join(self.video_dir, video_name)
            video_paths[view_index] = video_path
            video_writers[view_index] = imageio.get_writer(video_path, fps=self.video_fps, quality=8)
        return video_writers, video_paths

    def build_per_view_summary(self, per_view_history, video_paths):
        per_view_summary = {}
        for view_index, history in per_view_history.items():
            video_path = None
            if view_index in video_paths:
                video_path = os.path.relpath(video_paths[view_index], self.args.model_path).replace("\\", "/")
            per_view_summary[str(view_index)] = {
                "split_hint": self.view_tracks[view_index].split_hint,
                "num_frames": len(history),
                "psnr": float(np.mean([record["psnr"] for record in history])),
                "maepsnr": mae_psnr_from_values([record["psnr"] for record in history]),
                "ssim": float(np.mean([record["ssim"] for record in history])),
                "lpips": float(np.mean([record["lpips"] for record in history])),
                "video_path": video_path,
            }
        return per_view_summary

    def evaluate_post_cutoff_rollout(
        self,
        last_train_state,
        video_writers=None,
        per_view_history=None,
        write_panels=False,
    ):
        records = []
        metric_lists = []
        future_states = self.iter_future_states(last_train_state, self.post_cutoff_frame_groups)

        for future_frame_group, future_state in future_states:
            panel_dir = None
            if write_panels:
                report_dir = os.path.join(self.frame_report_dir, frame_slug(future_frame_group))
                os.makedirs(report_dir, exist_ok=True)
                panel_dir = os.path.join(report_dir, "post_cutoff_views")
            materialized_group = self.materialize_frame_group(future_frame_group)
            try:
                record, values = self.evaluate_frame(
                    materialized_group,
                    video_writers=video_writers,
                    per_view_history=per_view_history,
                    panel_dir=panel_dir,
                    deformation=(
                        future_state["d_xyz"],
                        future_state["d_rotation"],
                        future_state["d_scaling"],
                    ),
                    return_value_lists=True,
                )
            finally:
                self.dispose_frame_group(materialized_group)
                gc.collect()
            records.append(record)
            metric_lists.append(values)

        return records, metric_lists

    def write_segment_artifacts(self, post_cutoff_records, post_cutoff_metric_lists):
        full_timeline_records = self.current_train_val_history + post_cutoff_records

        self.save_metric_history("pre_cutoff_current_val_metrics_by_frame", self.current_val_history)
        self.save_metric_history("pre_cutoff_current_train_val_metrics_by_frame", self.current_train_val_history)
        self.save_metric_history("pre_cutoff_future10_all_metrics_by_frame", self.future10_all_history)
        self.save_metric_history("post_cutoff_rollout_metrics_by_frame", post_cutoff_records)
        save_json(os.path.join(self.stream_dir, "final_metrics_by_frame.json"), full_timeline_records)
        save_csv(
            os.path.join(self.stream_dir, "final_metrics_by_frame.csv"),
            full_timeline_records,
            fieldnames=["frame_index", "time", "num_views", "psnr", "maepsnr", "ssim", "lpips"],
        )

        plot_metric_curves(
            self.current_val_history,
            os.path.join(self.plot_dir, "pre_cutoff_current_val_metrics.png"),
            "Pre-0.75 Current-frame Val Metrics",
        )
        plot_metric_curves(
            self.current_train_val_history,
            os.path.join(self.plot_dir, "pre_cutoff_current_train_val_metrics.png"),
            "Pre-0.75 Current-frame Train+Val Metrics",
        )
        plot_metric_curves(
            self.future10_all_history,
            os.path.join(self.plot_dir, "pre_cutoff_future10_all_metrics.png"),
            "Pre-0.75 Future-10 All-view Rollout Metrics",
        )
        plot_metric_curves(
            post_cutoff_records,
            os.path.join(self.plot_dir, "post_cutoff_rollout_metrics.png"),
            "Post-0.75 RK2 Rollout Metrics",
        )
        plot_metric_curves(
            full_timeline_records,
            os.path.join(self.plot_dir, "final_average_metrics.png"),
            "Canonical Stream Metrics Across the Full Timeline",
        )
        plot_training_curves(self.frame_summaries, self.plot_dir)
        avg_train_seconds_excluding_frame0, num_train_frames_excluding_frame0 = mean_train_seconds_excluding_frame0(
            self.frame_summaries
        )
        (
            avg_train_gpu_peak_total_used_mb_excluding_frame0,
            max_train_gpu_peak_total_used_mb_excluding_frame0,
            avg_train_gpu_peak_total_used_percent_excluding_frame0,
            max_train_gpu_peak_total_used_percent_excluding_frame0,
            num_gpu_peak_frames_excluding_frame0,
        ) = summarize_train_gpu_peak_total_used_excluding_frame0(self.frame_summaries)

        segment_summary = {
            "pre_cutoff_current_val": self.summarize_segment(
                self.current_val_history,
                self.current_val_metric_lists,
            ),
            "pre_cutoff_current_train_val": self.summarize_segment(
                self.current_train_val_history,
                self.current_train_val_metric_lists,
            ),
            "pre_cutoff_future10_all": self.summarize_segment(
                self.future10_all_history,
                self.future10_all_metric_lists,
            ),
            "post_cutoff_rollout_all": self.summarize_segment(
                post_cutoff_records,
                post_cutoff_metric_lists,
            ),
            "training_time_summary": {
                "avg_train_seconds_excluding_frame0": avg_train_seconds_excluding_frame0,
                "num_frames_excluding_frame0": num_train_frames_excluding_frame0,
            },
            "training_gpu_peak_summary": {
                "metric": "torch.cuda.mem_get_info total - free",
                "avg_train_gpu_peak_total_used_mb_excluding_frame0": (
                    avg_train_gpu_peak_total_used_mb_excluding_frame0
                ),
                "max_train_gpu_peak_total_used_mb_excluding_frame0": (
                    max_train_gpu_peak_total_used_mb_excluding_frame0
                ),
                "avg_train_gpu_peak_total_used_percent_excluding_frame0": (
                    avg_train_gpu_peak_total_used_percent_excluding_frame0
                ),
                "max_train_gpu_peak_total_used_percent_excluding_frame0": (
                    max_train_gpu_peak_total_used_percent_excluding_frame0
                ),
                "num_frames_excluding_frame0": num_gpu_peak_frames_excluding_frame0,
            },
        }
        save_json(os.path.join(self.stream_dir, "segment_metric_averages.json"), segment_summary)

    def train(self):
        video_writers, video_paths = self.create_video_writers()
        per_view_history = defaultdict(list)
        post_cutoff_records = []
        post_cutoff_metric_lists = []

        try:
            for train_order_index, frame_group in enumerate(self.train_frame_groups):
                train_group_meta = self.subset_frame_group(frame_group, {"train"})
                val_group_meta = self.subset_frame_group(frame_group, {"val"})
                current_eval_group = self.subset_frame_group(frame_group, {"train", "val"})
                if len(train_group_meta.entries) == 0:
                    raise ValueError(
                        f"Frame {frame_group.frame_index} @ t={frame_group.time_value:.6f} has no train views."
                    )

                report_dir = os.path.join(self.frame_report_dir, frame_slug(frame_group))
                os.makedirs(report_dir, exist_ok=True)

                train_group = self.materialize_frame_group(train_group_meta)
                val_group = self.materialize_frame_group(val_group_meta)
                try:
                    if frame_group.frame_index == 0 and self.uses_external_frame0_result:
                        train_summary = self.build_loaded_frame0_summary(frame_group, train_group)
                    else:
                        train_summary = self.train_single_frame(
                            frame_group,
                            train_group,
                            train_order_index,
                            len(self.train_frame_groups),
                        )
                    self.save_frame_checkpoint(frame_group)

                    current_state = self.build_frame_state(frame_group)
                    current_deformation = (
                        current_state["d_xyz"],
                        current_state["d_rotation"],
                        current_state["d_scaling"],
                    )

                    current_val_record, current_val_lists = self.evaluate_frame(
                        val_group,
                        deformation=current_deformation,
                        return_value_lists=True,
                    )
                    current_train_val_group = self.combine_frame_groups(current_eval_group, [train_group, val_group])
                    try:
                        current_train_val_record, current_train_val_lists = self.evaluate_frame(
                            current_train_val_group,
                            video_writers=None if self.disable_visual_outputs else video_writers,
                            per_view_history=per_view_history,
                            panel_dir=None if self.disable_visual_outputs else os.path.join(report_dir, "current_views"),
                            deformation=current_deformation,
                            return_value_lists=True,
                        )
                    finally:
                        current_train_val_group.cameras.clear()
                finally:
                    self.dispose_frame_group(val_group)
                    self.dispose_frame_group(train_group)
                    gc.collect()

                should_evaluate_future10 = self.frame_uses_trd_prediction(frame_group)
                if should_evaluate_future10:
                    future_records, future_metric_lists = self.evaluate_future_window(
                        frame_group.frame_index,
                        current_state,
                        report_dir,
                    )
                    future10_metrics = aggregate_metric_lists(future_metric_lists)
                    future10_flat_views = sum(record["num_views"] for record in future_records)
                    future10_record = {
                        "frame_index": frame_group.frame_index,
                        "time": frame_group.time_value,
                        "num_views": future10_flat_views,
                        "psnr": future10_metrics["psnr"],
                        "maepsnr": future10_metrics["maepsnr"],
                        "ssim": future10_metrics["ssim"],
                        "lpips": future10_metrics["lpips"],
                    }
                else:
                    future_records = []
                    future_metric_lists = []
                    future10_flat_views = 0
                    future10_record = {
                        "frame_index": frame_group.frame_index,
                        "time": frame_group.time_value,
                        "num_views": 0,
                        "psnr": float("nan"),
                        "maepsnr": float("nan"),
                        "ssim": float("nan"),
                        "lpips": float("nan"),
                    }

                save_json(os.path.join(report_dir, "current_val_metrics.json"), current_val_record)
                save_json(os.path.join(report_dir, "current_train_val_metrics.json"), current_train_val_record)
                save_json(os.path.join(report_dir, "future10_all_summary.json"), future10_record)
                save_csv(
                    os.path.join(report_dir, "current_val_metrics.csv"),
                    [current_val_record],
                    fieldnames=["frame_index", "time", "num_views", "psnr", "maepsnr", "ssim", "lpips"],
                )
                save_csv(
                    os.path.join(report_dir, "current_train_val_metrics.csv"),
                    [current_train_val_record],
                    fieldnames=["frame_index", "time", "num_views", "psnr", "maepsnr", "ssim", "lpips"],
                )
                save_csv(
                    os.path.join(report_dir, "future10_all_summary.csv"),
                    [future10_record],
                    fieldnames=["frame_index", "time", "num_views", "psnr", "maepsnr", "ssim", "lpips"],
                )

                self.current_val_history.append(current_val_record)
                self.current_train_val_history.append(current_train_val_record)
                if should_evaluate_future10:
                    self.future10_all_history.append(future10_record)
                self.current_val_metric_lists.append(current_val_lists)
                self.current_train_val_metric_lists.append(current_train_val_lists)
                if should_evaluate_future10:
                    self.future10_all_metric_lists.append(
                        {
                            "psnr": [value for metric_list in future_metric_lists for value in metric_list["psnr"]],
                            "ssim": [value for metric_list in future_metric_lists for value in metric_list["ssim"]],
                            "lpips": [value for metric_list in future_metric_lists for value in metric_list["lpips"]],
                        }
                    )

                frame_summary = {
                    **train_summary,
                    "current_val_num_views": current_val_record["num_views"],
                    "current_val_psnr": current_val_record["psnr"],
                    "current_val_maepsnr": current_val_record["maepsnr"],
                    "current_val_ssim": current_val_record["ssim"],
                    "current_val_lpips": current_val_record["lpips"],
                    "current_train_val_num_views": current_train_val_record["num_views"],
                    "current_train_val_psnr": current_train_val_record["psnr"],
                    "current_train_val_maepsnr": current_train_val_record["maepsnr"],
                    "current_train_val_ssim": current_train_val_record["ssim"],
                    "current_train_val_lpips": current_train_val_record["lpips"],
                    "future10_num_frames": len(future_records),
                    "future10_num_views": future10_flat_views,
                    "future10_all_psnr": future10_record["psnr"],
                    "future10_all_maepsnr": future10_record["maepsnr"],
                    "future10_all_ssim": future10_record["ssim"],
                    "future10_all_lpips": future10_record["lpips"],
                }

                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        "stream_eval/current_val_psnr",
                        frame_summary["current_val_psnr"],
                        frame_group.frame_index,
                    )
                    self.tb_writer.add_scalar(
                        "stream_eval/current_val_maepsnr",
                        frame_summary["current_val_maepsnr"],
                        frame_group.frame_index,
                    )
                    self.tb_writer.add_scalar(
                        "stream_eval/current_train_val_psnr",
                        frame_summary["current_train_val_psnr"],
                        frame_group.frame_index,
                    )
                    self.tb_writer.add_scalar(
                        "stream_eval/current_train_val_maepsnr",
                        frame_summary["current_train_val_maepsnr"],
                        frame_group.frame_index,
                    )
                    if should_evaluate_future10:
                        self.tb_writer.add_scalar(
                            "stream_eval/future10_all_psnr",
                            frame_summary["future10_all_psnr"],
                            frame_group.frame_index,
                        )
                        self.tb_writer.add_scalar(
                            "stream_eval/future10_all_maepsnr",
                            frame_summary["future10_all_maepsnr"],
                            frame_group.frame_index,
                        )
                    self.tb_writer.add_scalar(
                        "stream_eval/train_seconds",
                        frame_summary["train_seconds"],
                        frame_group.frame_index,
                    )
                    if math.isfinite(frame_summary["train_gpu_peak_total_used_mb"]):
                        self.tb_writer.add_scalar(
                            "stream_train/train_gpu_peak_total_used_mb",
                            frame_summary["train_gpu_peak_total_used_mb"],
                            frame_group.frame_index,
                        )
                    if math.isfinite(frame_summary["train_gpu_peak_total_used_percent"]):
                        self.tb_writer.add_scalar(
                            "stream_train/train_gpu_peak_total_used_percent",
                            frame_summary["train_gpu_peak_total_used_percent"],
                            frame_group.frame_index,
                        )
                    if math.isfinite(frame_summary["train_gpu_peak_reserved_mb"]):
                        self.tb_writer.add_scalar(
                            "stream_train/train_gpu_peak_reserved_mb",
                            frame_summary["train_gpu_peak_reserved_mb"],
                            frame_group.frame_index,
                        )
                    if math.isfinite(frame_summary["train_gpu_peak_allocated_mb"]):
                        self.tb_writer.add_scalar(
                            "stream_train/train_gpu_peak_allocated_mb",
                            frame_summary["train_gpu_peak_allocated_mb"],
                            frame_group.frame_index,
                        )

                save_json(os.path.join(report_dir, "frame_summary.json"), frame_summary)
                self.frame_summaries.append(frame_summary)
                self.save_frame_summaries()

                torch.cuda.empty_cache()

            last_train_state = self.build_frame_state(self.train_frame_groups[-1])
            post_cutoff_records, post_cutoff_metric_lists = self.evaluate_post_cutoff_rollout(
                last_train_state,
                video_writers=None if self.disable_visual_outputs else video_writers,
                per_view_history=per_view_history,
                write_panels=not self.disable_visual_outputs,
            )
        finally:
            for writer in video_writers.values():
                writer.close()

        save_json(
            os.path.join(self.stream_dir, "final_metrics_by_view.json"),
            self.build_per_view_summary(per_view_history, video_paths),
        )
        self.write_segment_artifacts(post_cutoff_records, post_cutoff_metric_lists)

        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()

        print("Final artifacts written to {}".format(self.stream_dir))


if __name__ == "__main__":
    parser = ArgumentParser(description="Standalone streaming training with per-frame velocity integration")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vel_start_time", type=float, default=0.0)

    parser.add_argument("--frame_iterations", type=int, default=1000)
    parser.add_argument("--frame0_iterations", type=int, default=-1)
    parser.add_argument(
        "--vel_fixed_lr",
        type=float,
        default=-1.0,
        help="If positive, use this fixed learning rate for the motion optimizer in every frame instead of the scheduler.",
    )
    parser.add_argument("--future_frame_window", type=int, default=10)
    parser.add_argument("--frame_densification_interval", type=int, default=-1)
    parser.add_argument("--frame_densify_from_iter", type=int, default=-1)
    parser.add_argument("--frame_densify_until_iter", type=int, default=-1)
    parser.add_argument("--frame0_densification_interval", type=int, default=-1)
    parser.add_argument("--frame0_densify_from_iter", type=int, default=-1)
    parser.add_argument("--frame0_densify_until_iter", type=int, default=-1)
    parser.add_argument("--frame_min_opacity", type=float, default=0.08)
    parser.add_argument("--frame0_min_opacity", type=float, default=0.08)
    parser.add_argument("--frame_opacity_reset_interval", type=int, default=-1)
    parser.add_argument("--video_fps", type=int, default=-1)
    parser.add_argument("--video_error_clip", type=float, default=0.25)
    parser.add_argument(
        "--disable_visual_outputs",
        action="store_true",
        help="Disable mp4 video generation and per-view comparison panel image output.",
    )
    parser.add_argument("--time_round_digits", type=int, default=10)
    parser.add_argument("--train_time_cutoff", type=float, default=0.75)
    parser.add_argument("--lambda_deformation_reg", type=float, default=0.0)
    parser.add_argument(
        "--frame0_static_model_path",
        type=str,
        default=None,
        help="Optional .ply file or static model directory used as the precomputed frame-0 Gaussian result.",
    )
    parser.add_argument(
        "--fastgs",
        action="store_true",
        help="Enable the FastGS Gaussian backend. Requires diff_gaussian_rasterization_fastgs in the runtime environment.",
    )

    args = parser.parse_args()

    if args.frame_iterations <= 0:
        raise ValueError("--frame_iterations must be positive.")
    if args.frame0_iterations == 0 or args.frame0_iterations < -1:
        raise ValueError("--frame0_iterations must be positive, or -1 to reuse --frame_iterations.")
    if args.vel_fixed_lr == 0.0 or args.vel_fixed_lr < -1.0:
        raise ValueError("--vel_fixed_lr must be positive, or -1 to keep the scheduler.")
    if args.frame_densification_interval == 0 or args.frame_densification_interval < -1:
        raise ValueError("--frame_densification_interval must be positive, or -1 to reuse the base setting.")
    if args.frame_densify_from_iter < -1:
        raise ValueError("--frame_densify_from_iter must be >= -1.")
    if args.frame_densify_until_iter < -1:
        raise ValueError("--frame_densify_until_iter must be >= -1.")
    if args.frame0_densification_interval == 0 or args.frame0_densification_interval < -1:
        raise ValueError("--frame0_densification_interval must be positive, or -1 to reuse the frame setting.")
    if args.frame0_densify_from_iter < -1:
        raise ValueError("--frame0_densify_from_iter must be >= -1.")
    if args.frame0_densify_until_iter < -1:
        raise ValueError("--frame0_densify_until_iter must be >= -1.")
    if args.frame_min_opacity <= 0.0:
        raise ValueError("--frame_min_opacity must be positive.")
    if args.frame0_min_opacity <= 0.0:
        raise ValueError("--frame0_min_opacity must be positive.")
    if args.future_frame_window < 0:
        raise ValueError("--future_frame_window must be non-negative.")
    if not (0.0 <= args.train_time_cutoff <= 1.0):
        raise ValueError("--train_time_cutoff must lie in [0, 1].")
    if args.lambda_deformation_reg < 0.0:
        raise ValueError("--lambda_deformation_reg must be non-negative.")

    print("Streaming optimize {}".format(args.model_path))

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    trainer = IntegratedVelocityStreamTrainer(
        args=args,
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
    )
    trainer.train()

    print("\nStreaming training complete.")
