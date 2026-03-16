# 多体颗粒碰撞 DEM 仿真程序说明

## 程序概况

本程序用于模拟尘埃等多体颗粒在动力学条件下的碰撞与吸积过程，采用离散元方法（DEM）并包含 JKR 黏附接触模型。当前版本为三维（3D）实现，主要支持：

- 以 N×N×N 规则网格初始化颗粒系，周期性边界条件；
- 以两个多粒子团簇 A/B 为初始条件的碰撞模拟（含 JKR 黏附、滚动/切向力），支持树状/致密等不同内部结构；
- 输出碰撞与凝聚过程中的快照图像；
- 记录能量与结构指标随时间的演化，用于分析团簇形成与演化特征；
- 在两颗粒情形下支持扫描生成相图：
	- (|Vx|, Vy) 扫描：输出碰后法向相对速度大小 |v_n|；
	- (Vn, gamma) 扫描：横轴法向速度 Vn、纵轴表面能 gamma，可输出 |v_n| 或法向恢复系数（可配置）。

## 运行环境与依赖

- 建议 Python ≥ 3.10；
- 主要依赖：`numpy`、`numba`（可选，用于加速）、`matplotlib`、`pyyaml`、`Pillow`、`opencv-python`，以及标准库中的 `tkinter` 等。

可使用虚拟环境并按需安装上述依赖，例如：

```bash
pip install numpy numba matplotlib pyyaml pillow opencv-python
```

## 快速开始

在项目根目录（N_bodys_collision）下：

```bash
python scripts/run_simulation.py --config configs/default.yaml
```

- 默认使用 `configs/default.yaml` 中的参数；
- 若加上 `--numba` 选项，则在配置允许的前提下尝试启用 numba 加速。

## 项目结构

```text
N_bodys_collision/
├─ configs/
│  ├─ default.yaml
│  └─ cluster_collision.yaml
├─ data/
│  ├─ 10_small_t_10e-5_seed6/
│  ├─ 30_small_t_30e-5_seed6/
|  └─ cluster_collision/
├─ scripts/
│  ├─ pic2video.py
│  ├─ plot_cluster_evolution.py
│  ├─ plot_result.py
│  ├─ run_simulation.py
│  └─ run_cluster_collision.py
├─ src/
│  ├─ initialization/
│  │  ├─ grid.py
│  │  └─ cluster.py
│  ├─ io/
│  │  └─ snapshot_saver.py
│  ├─ physics/
│  │  ├─ contact/
│  │  │  ├─ jkr_normal.py
│  │  │  ├─ rolling.py
│  │  │  └─ tangential.py
│  │  └─ materials.py
│  ├─ simulation/
│  │  ├─ contact_detector.py
│  │  ├─ particle_system.py
│  │  └─ simulator.py
│  └─ visualization/
│     └─ report_generator.py
└─ readme.md
```

## 文件说明

### configs

#### default.yaml

**信息说明**  
配置文件 `configs/default.yaml` 记录了模拟所需的各类参数，包括：

- 颗粒材料参数（半径、密度、弹性模量、泊松比、表面能等）；
- 接触模型参数（摩擦系数、恢复系数、滚动阻尼等）；
- 初始化参数（粒子数 N、网格间距 h、初始速度范围 [vmin, vmax]、随机种子等，生成 N×N×N 颗粒）；
- 仿真控制参数（时间步长 dt、总时长 t_end、自适应步长相关参数等）；
- 绘图与快照输出参数（输出目录、间隔、是否显示速度矢量等）。

**补充说明**  
可以通过适当放大粒径半径并同时提高表面能参数来提高稳定时间步，从而在数值上“加速”模拟。但需要注意：

- 若在放大粒径尺度的同时保持速度不变，则物理过程的特征时间会变长，颗粒凝聚所需的模拟时长（以物理时间计）会增加；
- 从数值实现角度看，时间步增大可以减少步数，总计算时间可能下降，但过大的时间步会导致接触穿透、切向历史失真等数值误差；
- 在模拟后期，团簇尺度增大、整体运动减慢，团簇间的有效碰撞率会显著下降。

当前版本常用的仿真开关包括：

- `use_numba`：启用 numba 后端；
- `use_numba_parallel`：启用 numba 邻域搜索并行路径；
- `use_neighbor_cache`：启用 Verlet 邻域缓存；
- `use_adaptive_grid`：启用自适应网格；Numba 路径使用空间哈希，Python 路径当前回退为固定网格；
- `use_jkr_interp`：启用 JKR 法向力插值表，减少牛顿迭代开销；
- `adaptive_timestep`：启用按全局最大速度平滑缩放的自适应时间步长。

当前版本通常建议优先使用“串行 numba + 邻域缓存 + JKR 插值”这一组合；`use_numba_parallel`、`use_adaptive_grid` 与 `adaptive_timestep` 是否带来净收益，取决于具体粒子数、空间稀疏程度与时间步设置，建议结合实际算例测试。

#### cluster_collision.yaml

该配置文件配合 `scripts/run_cluster_collision.py` 使用，用于控制两团簇 A/B 碰撞问题，主要包括：

- `materials`、`simulation` 段：与 `default.yaml` 中含义基本一致；
- `cluster` 段：
	- `N1`, `N2`：团簇 A/B 内颗粒数；
	- `seed`：控制团簇内部几何结构的随机种子，相同 seed 可完全复现；
	- `gap_factor`：初始 A/B 团簇之间的间隙（以粒径半径 r 为单位）；
	- `z_target`：目标平均配位数（每个颗粒的平均邻居数），用于调节团簇内部结构的“紧凑/松散”程度：
		- 小 `z_target`（≈2–3）→ 更树状/枝状的稀疏团簇；
		- 大 `z_target`（≈4 以上）→ 更致密、近似块状团簇；
	- `vxB`, `vyB`：B 团簇质心的初始速度分量（A 质心初始静止在原点）；
	- `adhesion_vn_thresh`, `adhesion_steps`：两颗粒情形下的黏附判据；
	- `scan_enabled` 及 `vx_min/vx_max/vx_steps/vy_min/vy_max/vy_steps`：在 `N1=N2=1` 时启用 (|Vx|, Vy) 扫描并输出 |v_n| 热力图；
	- `vn_gamma_scan_enabled` 及 `vn_min/vn_max/vn_steps/gamma_min/gamma_max/gamma_steps`：在 `N1=N2=1` 时启用 (Vn, gamma) 扫描；
		- `vn_gamma_heatmap_filename`：输出文件名；
		- `vn_gamma_use_restitution`：若为 true，则热力图数值不再是 |v_n|，而是法向恢复系数（用“碰后 |v_n| / 碰前 Vn”定义）。

### data

#### 10_small_t_10e-5_seed6

目录中 `report.txt` 记录了本次模拟的参数信息：N = 10，较小时间步长（未开启自适应时间步长），总模拟时长为 10×10⁻⁵ s，随机种子为 seed = 6。同时提供最终颗粒分布的快照 `final_snapshot.png`。

从近期版本开始，`report.txt` 的数据列在原有能量与最大速度基础上，额外记录了若干结构指标（在每个快照时刻写入一行）：

- `cluster_count`：当前连通体数量（基于 2r 内真实接触构造的接触图）；
- `coordination_mean`：全局平均配位数（接触图中每个粒子的平均邻居数）；
- `cluster_rg_mean`：全局连通体平均回转半径（对每个连通体计算回转半径后取平均）。

#### 30_small_t_30e-5_seed6

目录中 `report.txt` 记录了本次模拟的参数信息（N、时间步长、总时长、随机种子等），并包含最终颗粒分布的快照 `final_snapshot.png`。

#### 补充说明

后续版本中，为节约云端存储与传输资源，`data` 目录中将仅保留每次模拟的 `report.txt` 与最后一张快照图像，其余中间快照可按需在本地生成与保存。

### scripts

#### run_simulation.py

模拟主入口脚本。其核心流程为：

- 解析命令行参数，读取 `configs/default.yaml` 或用户指定的配置文件；
- 根据材料与初始化参数构造 `Materials`、`InitParams` 和 `SystemState`；
- 根据仿真参数构造 `SimParams`，可选择启用 numba、邻域缓存、自适应网格、JKR 插值与自适应时间步长等；
- 调用 `DEMSimulator` 执行完整 DEM 仿真，按配置输出快照与报告。

说明：脚本支持 `--numba` 命令行开关，但若配置文件中已经显式写入 `simulation.use_numba`，则以配置文件为准。

示例用法见“快速开始”章节。

#### plot_result.py

用于绘制模拟过程中能量随时间的变化曲线，并监督系统中最大速度的演化，可用于：

- 检查仿真是否稳定收敛；
- 复盘物理参数（如接触刚度、阻尼系数等）是否设置合理。

**兼容性说明**

由于 `report.txt` 可能包含更多列（例如团簇结构指标），`plot_result.py` 只读取每行的前 5 列（t、三种能量、Vmax），其余列会自动忽略。

#### plot_cluster_evolution.py

用于绘制团簇结构指标随时间的演化（从 `report.txt` 读取）：

- 连通体数量 `cluster_count`；
- 全局平均配位数 `coordination_mean`；
- 全局连通体平均回转半径 `cluster_rg_mean`。

运行后会在同目录输出 `cluster_evolution.png`。

#### pic2video.py

一个基于 GUI 的小工具，可将多张快照图片按顺序拼接为视频，以便动态展示碰撞与凝聚过程。支持：

- 选择包含快照的目录；
- 预览与调整播放帧率；
- 导出为常见视频格式（依赖 `opencv-python`）。

#### run_cluster_collision.py

两团簇 A/B 碰撞专用脚本，读取 `configs/cluster_collision.yaml`：

- 根据 `cluster` 段构造两个随机团簇 A/B：
	- 使用 `src/initialization/cluster.py` 中的随机连通团簇生成器；
	- 团簇内部颗粒构成连通体，相邻颗粒中心距约为 `2r×0.99`，初始化时已有轻微重叠接触；
	- 通过 `z_target` 控制平均配位数，从而在“树状（松散）–致密”之间连续调节；
	- 通过 `seed` 控制几何结构的随机性（相同 seed 完全复现）。
- 将 A 质心放在原点、静止不动，将 B 放在右侧一定距离，并赋予给定 `(vxB, vyB)` 作为来流速度；
- 使用 DEM 框架推进时间，输出快照与报告，与 `run_simulation.py` 保持一致格式；
- 当 `N1=N2=1` 时，可启用两颗粒专用判据与扫描：区分未接触/分离/黏附等状态，并生成相图（见 `cluster_collision.yaml` 中 scan 配置）。

### src

#### initialization

##### grid.py

初始化程序：在三维域中生成 N×N×N 规则网格，每个网格中心放置一个粒子。

- 粒子位置：由网格间距 h 和 N 决定；
- 初始速度：幅值在 `[vmin, vmax]` 内均匀随机，方向在球面上均匀随机；
- 随机性由配置文件中的 `seed` 控制，便于复现实验。

##### cluster.py

团簇初始化程序：在三维 FCC 结构上生成两个随机连通团簇 A/B，用于 A/B 碰撞脚本。

- 使用前沿生长算法逐个添加新颗粒：
	- 始终保证新颗粒至少与一个已有颗粒接触，因此团簇整体是连通体；
	- 通过限制每个格点度数上限并对候选位置打分，使实际平均配位数尽量贴近 `z_target`；
	- 在较小 `z_target` 下优先形成带分支的树状结构，而不是单一长链；
	- 不同 `seed` 会得到不同的树状/团簇形态，但同一 `seed` 可完全复现。

#### io

##### snapshot_saver.py

快照记录与输出模块，用于在仿真过程中定期保存粒子位置/速度等信息的可视化结果：

- 3D 快照显示粒子位置、速度大小色标与速度箭头（可选）；
- 可按设定步长（`snapshot_interval`、`save_stride`）定期生成 PNG 图像。

#### physics

##### materials.py

根据颗粒材料参数（密度、半径、弹性模量、泊松比、表面能等）推导接触模型所需的各种有效参数，如等效质量、等效刚度、黏附相关参数等。

##### contact

- **jkr_normal.py**：
	- 采用 JKR 模型进行法向接触力计算；
	- 支持牛顿迭代或 JKR 插值两种求解策略，以平衡精度与效率。
- **rolling.py**：
	- 滚动阻力模型，计算颗粒间相对滚动的阻尼力矩；
- **tangential.py**：
	- 切向力模型，考虑摩擦与切向接触刚度等效应。

#### simulation

##### contact_detector.py

碰撞检测模块，当前实现要点：

- 使用周期边界条件，对坐标进行周期性包裹；
- 采用空间划分 + 邻域搜索，将复杂度由 O(N²) 降为近似 O(N)；
- 具体而言：
	- 固定网格路径中，格子边长取不小于阈值 `2*r + skin`，因此仅需扫描 3×3 邻域即可保证不漏检；
	- Python 回退路径支持固定网格搜索；
	- Numba 路径支持固定网格搜索，以及基于空间哈希的自适应紧凑存储搜索；
	- 输出所有距离小于阈值 `2*r + skin` 的颗粒对。
- 支持串行与并行的 numba 邻域搜索版本；并行实现采用“两阶段统计 + 前缀和写回”策略以避免写冲突。

**补充说明**  
当前实现中，固定网格与空间哈希两条 numba 邻域搜索路径都已使用纯数组数据结构，并统一为 3×3 邻域扫描。空间哈希的主要优势是将存储量从固定网格的 O(Nx×Ny) 降为与粒子数成正比的 O(N)，因此在“盒子较大但颗粒只占据局部区域”的场景下更省内存、缓存更友好；但若系统主要耗时在接触力计算而非邻域搜索，则总加速收益可能有限。

##### particle_system.py

定义颗粒状态的数据结构（位置、速度、自旋、质量、半径等），并提供对系统整体状态的基础操作封装，为模拟器与接触检测等模块提供统一接口。

##### simulator.py

DEM 主模拟器：

- 调用接触检测、法向/切向/滚动接触模型，计算每一步的粒子受力；
- 更新粒子位置、速度与自旋，并根据需要调用快照输出与报告记录模块；
- 根据配置在 Python / Numba / C++ 后端之间选择可用实现；当前默认主要使用 Python 与 Numba 两条路径；
- 支持 JKR 法向力插值表，以减少频繁牛顿迭代带来的开销；
- 支持邻域缓存（Verlet 邻居表）：
	- 初始时通过完整碰撞检测构建邻居表；
	- 后续时间步复用该邻居表，只要所有粒子位移未超过 `skin/2`；
	- 一旦有粒子位移超过阈值，则重新构建邻居表，以保证邻域信息正确。
	- 在启用 `use_adaptive_grid` 时，邻居表的重建会切换到相应的自适应搜索路径。

**补充说明**  
当前版本会在快照与 `report.txt` 中记录能量、最大速度以及若干结构指标；若启用自适应时间步长，步长倍率也可选择性写入快照标注。当前版本中基于团簇的“刚体化”近似尚不成熟，因此暂未在此处展开说明，后续若算法稳定将进一步整理文档。

#### visualization

##### report_generator.py

用于生成本次仿真的文字报告与过程数据摘要，包括：

- 输入参数与物理量配置；
- 关键过程量（如最大速度、能量等）的统计信息；
- 输出到 `data/.../report.txt` 以便事后分析与复现实验。

---

如需在此基础上扩展新的接触模型、不同维度的初始化方式或更复杂的后处理，可参考对应模块的接口设计进行继承或修改。


