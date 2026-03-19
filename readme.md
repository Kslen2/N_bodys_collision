# 多体颗粒碰撞 DEM 仿真程序说明

## 项目概述

本项目用于模拟尘埃等多体颗粒在动力学条件下的碰撞、黏附与吸积过程，基于离散元方法（DEM）并实现 JKR 黏附接触模型。当前为三维（3D）实现，覆盖规则网格初始化、随机团簇初始化、团簇碰撞、两颗粒相图扫描等典型场景，支持输出快照、能量与结构指标，便于进行物理过程分析。

核心能力概览：

- 规则网格初始化：N×N×N 颗粒、周期边界条件、随机初速度；
- 团簇碰撞：随机连通团簇 A/B、可调平均配位数（树状到致密）；
- 接触模型：JKR 法向力 + 切向/滚动模型，支持插值表加速；
- 输出记录：快照、能量、最大速度、结构指标随时间演化；
- 两颗粒扫描：支持 (|Vx|, Vy) 与 (Vn, gamma) 相图；
- 加速路径：Numba 后端、并行邻域搜索、邻域缓存与自适应网格。

## 运行环境与依赖

- 推荐 Python 3.10 或更高；
- 主要依赖：numpy、numba（可选加速）、matplotlib、pyyaml、Pillow、opencv-python，及标准库 tkinter 等。

安装示例：

```bash
pip install numpy numba matplotlib pyyaml pillow opencv-python
```

## 快速开始

在项目根目录运行默认仿真：

```bash
python scripts/run_simulation.py --config configs/default.yaml
```

启用 numba 的命令行开关：

```bash
python scripts/run_simulation.py --config configs/default.yaml --numba
```

说明：若配置文件中显式设置了 simulation.use_numba，脚本会以配置为准。

## 常用工作流

### 1) 规则网格初始化仿真

- 修改 configs/default.yaml 的初始化参数与材料参数；
- 运行 run_simulation.py；
- 结果输出至 data/ 指定目录。

### 2) 两团簇碰撞仿真

- 修改 configs/cluster_collision.yaml 中 cluster 段参数；
- 运行 run_cluster_collision.py；
- 结果包含快照与 report.txt。

```bash
python scripts/run_cluster_collision.py --config configs/cluster_collision.yaml
```

### 3) 两颗粒相图扫描

- 设置 N1=N2=1；
- 启用 scan_enabled 或 vn_gamma_scan_enabled；
- 运行 run_cluster_collision.py；
- 输出热力图文件到 data/ 目录。

## 输出与结果解释

### 输出目录结构

data/ 下的每次运行通常包含：

- report.txt：本次仿真的参数与过程量统计；
- 快照图像：按间隔保存的 PNG 序列；
- final_snapshot.png：最终状态快照。

### report.txt 的主要内容

报告文件包含：

- 输入参数摘要；
- 能量项与最大速度随时间演化；
- 结构指标（每个快照时刻记录）：
  - cluster_count：连通体数量；
  - coordination_mean：平均配位数；
  - cluster_rg_mean：连通体平均回转半径。

### 后处理脚本

- plot_result.py：绘制能量与最大速度曲线；
- plot_cluster_evolution.py：绘制结构指标演化；
- pic2video.py：将快照序列拼接成视频。

## 配置文件说明（概要）

### default.yaml

常用配置分为：

- materials：密度、半径、弹性模量、泊松比、表面能等；
- contact：摩擦、切向与滚动参数；
- initialization：粒子数 N、网格间距 h、速度范围、随机种子；
- simulation：dt、t_end、自适应步长、邻域搜索与 numba 开关；
- output：快照间隔、输出目录、是否显示速度矢量。

数值稳定性提示：适当增大半径并提升表面能可放大稳定时间步，但会改变特征时间尺度。若保持速度不变，物理过程整体变慢；时间步过大可能引入穿透或切向历史误差。

### cluster_collision.yaml

cluster 段主要参数：

- N1, N2：团簇 A/B 颗粒数；
- seed：团簇形态随机种子；
- gap_factor：团簇间初始间隙（以粒径为单位）；
- z_target：目标配位数，控制树状到致密；
- vxB, vyB：B 团簇来流速度；
- adhesion_vn_thresh, adhesion_steps：两颗粒黏附判据；
- scan_enabled：(|Vx|, Vy) 扫描；
- vn_gamma_scan_enabled：(Vn, gamma) 扫描与输出控制。

## 性能与加速建议

- 建议优先组合：串行 numba + 邻域缓存 + JKR 插值；
- use_numba_parallel 与 use_adaptive_grid 在稀疏系统或大盒子场景更有优势；
- adaptive_timestep 适用于速度逐步降低的长时演化，但需关注能量数值噪声。

## 项目结构

```text
N_bodys_collision/
├─ configs/
│  ├─ default.yaml
│  └─ cluster_collision.yaml
├─ data/
│  ├─ animation_saver/
│  └─ cluster_collision/
├─ scripts/
│  ├─ interactive_player.py
│  ├─ pic2video.py
│  ├─ plot_cluster_evolution.py
│  ├─ plot_result.py
│  ├─ replay_hdf5.py
│  ├─ run_cluster_collision.py
│  └─ run_simulation.py
├─ src/
│  ├─ initialization/
│  │  ├─ cluster.py
│  │  └─ grid.py
│  ├─ io/
│  │  ├─ snapshot_saver.py
│  │  └─ trajectory_recorder.py
│  ├─ physics/
│  │  ├─ contact/
│  │  │  ├─ jkr_normal.py
│  │  │  ├─ rolling.py
│  │  │  └─ tangential.py
│  │  ├─ materials.py
│  │  └─ quaternion.py
│  ├─ simulation/
│  │  ├─ contact_detector.py
│  │  ├─ contact_detector_2d_backup.py
│  │  ├─ particle_system.py
│  │  ├─ simulator.py
│  │  └─ simulator_2d_backup.py
│  └─ visualization/
│     └─ report_generator.py
└─ readme.md
```

## 核心模块说明（精简版）

### initialization

- grid.py：生成规则网格初始条件；
- cluster.py：生成随机连通团簇，控制平均配位数。

### physics

- materials.py：材料参数到等效接触参数的推导；
- contact/jkr_normal.py：JKR 法向力；
- contact/tangential.py：切向摩擦；
- contact/rolling.py：滚动阻力；
- quaternion.py：旋转与姿态更新。

### simulation

- contact_detector.py：邻域搜索与接触对生成（固定网格 + 空间哈希）；
- particle_system.py：系统状态与粒子属性容器；
- simulator.py：DEM 主循环与数值更新。

### io 与 visualization

- snapshot_saver.py：快照输出与可视化渲染；
- trajectory_recorder.py：轨迹与时间序列记录；
- report_generator.py：report.txt 生成。

## 复现实验与随机性控制

- 所有随机初始化均由配置文件 seed 控制；
- 使用相同配置与 seed 可复现实验结果；
- 建议在报告或论文中记录配置文件与输出目录。

## 常见问题

1) 结果不稳定或穿透明显：
   - 降低 dt；
   - 增大阻尼或摩擦参数；
   - 确认接触模型参数与材料量级匹配。

2) 速度逐渐趋近零但计算很慢：
   - 启用 adaptive_timestep；
   - 使用邻域缓存与 JKR 插值。

3) 输出文件过多：
   - 增大 snapshot_interval 或 save_stride；
   - 仅保留 report.txt 与 final_snapshot.png。

---

如需扩展新的接触模型、初始化方式或后处理流程，可在 src/physics、src/initialization 与 scripts 中参考现有接口进行扩展。


