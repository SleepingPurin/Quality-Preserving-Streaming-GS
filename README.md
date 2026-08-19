# Streaming 3D Gaussians

Standalone per-frame streaming training for 3D Gaussians with velocity integration.

The main entry point is `train_stream_integrate.py`. This repository is for research use. See `LICENSE.md`, `LICENSE`, and `NOTICE.md`.

## Environment

Follow the installation instructions in [TRACE](https://github.com/vLAR-group/TRACE) and use its conda environment `freegave`. Do not create a separate environment for this repository.

```bash
conda activate freegave
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

`--fastgs` is optional and requires `diff_gaussian_rasterization_fastgs`, which is not bundled here.

## Data

We evaluate on two TRACE datasets: **Dynamic Object Dataset** and **Dynamic Indoor Scene Dataset**.

- [Dynamic Object Dataset](https://huggingface.co/datasets/scintigimcki/DynamicObjects)
- [Dynamic Indoor Scene Dataset](https://huggingface.co/datasets/scintigimcki/DynamicIndoorScenes)

Follow the dataset download and directory layout in [TRACE](https://github.com/vLAR-group/TRACE). A scene folder should contain `transforms_train.json` (Blender / D-NeRF convention).

## Training

```bash
conda activate freegave

python train_stream_integrate.py \
  --source_path /path/to/scene \
  --model_path ./output/scene_stream \
  --frame0_static_model_path /path/to/frame0/point_cloud.ply \
  --train_time_cutoff 0.75 \
  --frame0_iterations 15000 \
  --frame_iterations 250 \
  --vel_fixed_lr 0.0002 \
  --frame0_densify_from_iter 500 \
  --frame0_densification_interval 100 \
  --frame0_densify_until_iter 15000 \
  --frame_densify_from_iter 50 \
  --frame_densification_interval 50 \
  --frame_densify_until_iter 250 \
  --future_frame_window 10 \
  --fps 60 \
  --lambda_deformation_reg 0.0 \
  --disable_visual_outputs
```

`--frame0_static_model_path` is optional. If omitted, frame 0 is trained from scratch.

Outputs are written to `--model_path`. Streaming artifacts are under `model_path/stream/`.

### Parameters in the command above

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--source_path` / `-s` | `""` | Scene root directory (TRACE layout, with `transforms_train.json`). |
| `--model_path` / `-m` | `""` | Output directory. If empty, a folder under `./output/` is created. |
| `--frame0_static_model_path` | `None` | Optional pretrained frame-0 Gaussian. Can be a `.ply` file or a model directory containing `point_cloud.ply` / `point_cloud/iteration_*/point_cloud.ply`. Skip this flag to train frame 0 inside this script. |
| `--train_time_cutoff` | `0.75` | Normalized time in `[0, 1]`. Only frames with time `<=` this value are used for training; later frames are held out (future extrapolation). |
| `--frame0_iterations` | `-1` | Optimization steps for frame 0. `-1` means reuse `--frame_iterations`. Use a larger value (e.g. `15000`) when training frame 0 from scratch or densifying it. |
| `--frame_iterations` | `1000` | Optimization steps for every later frame. |
| `--vel_fixed_lr` | `-1` | If positive, use this fixed learning rate for the motion / velocity optimizer in every frame. `-1` keeps the default scheduler. |
| `--frame0_densify_from_iter` | `-1` | Frame-0 densification start iteration. `-1` reuses `--densify_from_iter` (default `500`). |
| `--frame0_densification_interval` | `-1` | Frame-0 densification interval. `-1` reuses the later-frame densification interval. |
| `--frame0_densify_until_iter` | `-1` | Frame-0 densification end iteration. `-1` reuses `--frame0_iterations`. Set it to `15000` to densify throughout a long frame-0 stage. |
| `--frame_densify_from_iter` | `-1` | Later-frame densification start iteration. `-1` reuses `--densify_from_iter` (`500`). If this is larger than `--frame_iterations`, later frames will not densify. |
| `--frame_densification_interval` | `-1` | Later-frame densification interval. `-1` reuses `--densification_interval` (`100`). |
| `--frame_densify_until_iter` | `-1` | Later-frame densification end iteration. `-1` reuses `--frame_iterations`. |
| `--future_frame_window` | `10` | How many future frames to render / evaluate after each trained frame. `0` disables the future window. |
| `--fps` | `30` | Dataset frame rate. Used as the velocity integration time step `dt = 1 / fps`. TRACE scenes are typically `60`. |
| `--lambda_deformation_reg` | `0.0` | Weight of the deformation regularizer. `0` disables it. |
| `--disable_visual_outputs` | off | Disable mp4 videos and per-view comparison images. Metrics and model checkpoints are still saved. |

### Other commonly used flags

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--freegave` | off | Enable TRACE's FreeGave physics code / motion decomposition. |
| `--fastgs` | off | Use the FastGS rasterizer. Requires `diff_gaussian_rasterization_fastgs`. |
| `--white_background` / `-w` | on | White background (default for TRACE synthetic scenes). |
| `--sh_degree` | `3` | Spherical-harmonic degree of the Gaussians. |
| `--max_time` | `0.7` | Time upper bound used by the velocity integrator. |
| `--vel_start_time` | `0.0` | Time lower bound for velocity integration. |
| `--fps` vs `--video_fps` | `30` / `-1` | `--fps` is the physical frame rate for integration. `--video_fps` is only the saved video frame rate; `-1` copies `--fps`. |
| `--feature_lr` | `0.0025` | Gaussian SH / appearance learning rate. Set to `0` to freeze appearance and optimize motion only. |
| `--opacity_lr` | `0.05` | Gaussian opacity learning rate. Set to `0` to freeze opacity. |
| `--scaling_lr` | `0.001` | Gaussian scale learning rate. Set to `0` to freeze scale. |
| `--rotation_lr` | `0.001` | Gaussian rotation learning rate. |
| `--position_lr_init` | `0.00016` | Initial XYZ learning rate. |
| `--lambda_dssim` | `0.2` | DSSIM term weight in the reconstruction loss. |
| `--frame0_min_opacity` | `0.08` | Opacity prune threshold for frame 0. |
| `--frame_min_opacity` | `0.08` | Opacity prune threshold for later frames. |
| `--frame_opacity_reset_interval` | `-1` | Opacity reset interval for later frames. `-1` disables reset. |
| `--quiet` | off | Reduce logging. |
| `--detect_anomaly` | off | Enable `torch.autograd.set_detect_anomaly`. |

Values of `-1` on the frame-wise densify / iteration flags mean “inherit the corresponding base setting”, not “disable”. To freeze Gaussian appearance while training velocity, add `--feature_lr 0 --opacity_lr 0 --scaling_lr 0`.

## Acknowledgement

This repository is built upon [TRACE](https://github.com/vLAR-group/TRACE) and [3DGStream](https://github.com/SJoJoK/3DGStream). We thank the authors for releasing their code.

## Citation

Please cite 3DGStream, TRACE, and 3D Gaussian Splatting if you use this code.

```bibtex
@InProceedings{sun20243dgstream,
    author    = {Sun, Jiakai and Jiao, Han and Li, Guangyuan and Zhang, Zhanjie and Zhao, Lei and Xing, Wei},
    title     = {3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {20675-20685}
}

@article{li2025trace,
  title={TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos},
  author={Jinxi Li and Ziyang Song and Bo Yang},
  year={2025},
  journal={ICCV}
}

@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
