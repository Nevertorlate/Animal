# PlayerOne on DiffSynth-Studio

这个目录里我已经做了三件事：

1. 下载了 `PlayerOne: Egocentric World Simulator` 的论文：
   `paper/playerone_arxiv_2506.09995.pdf`
2. 拉下了参考仓库：
   `playerone-official/`
   `diffsynth-studio/`
3. 基于 `DiffSynth-Studio` 的 `Wan` 组件，做了一份按论文结构组织的本地复现骨架：
   `playerone_repro/`

## 这份复现包含什么

这不是官方权重复刻，因为 `playerone-official` 当前公开仓库基本只有 README 和素材，没有训练/推理实现代码。
所以这里做的是一份“结构复现”：

- `Part-disentangled Motion Injection`
  对应 `playerone_repro/modules.py`
- `Scene-frame Reconstruction`
  对应 `playerone_repro/modules.py`
- `Wan` backbone 包装、scheduler、VAE 编解码
  对应 `playerone_repro/pipeline.py`
- 两阶段训练策略
  对应 `playerone_repro/training.py`

实现思路和论文保持一致：

- 首帧图像先编码成 VAE latent。
- 人体动作拆成 `body_feet`、`hands`、`head` 三路。
- `head` 再转成相机旋转序列，并复用 DiffSynth 的 `SimpleAdapter` 做 camera control。
- `SMPL` 注入不再只有一条 dense 分支，现在是：
  `part-disentangled` 体条件 + `head/camera` 控制 + `global motion tokens` 三路并行。
- 点云 / point map 不再被当成“一次性整段输入”。
- 现在改成 I2V 自回归流程：
  先生成首段视频，再基于已生成视频更新 scene point cloud；如果暂时没有真实 renderer，也会默认把已生成视频编码成 `scene memory`，作为下一段生成时的 scene 条件。
- 训练代码也同步改成与推理一致的 chunked autoregressive flow-matching，并支持用模型自己 rollout 出来的 chunk 更新 scene state，而不是只用 teacher-forcing。

## 文件说明

- `playerone_repro/data.py`
  读取和切分 `SMPL/body/head/hands` 动作序列。
- `playerone_repro/modules.py`
  PMI、camera 条件、global motion tokens、scene memory / point-map encoder、backbone wrapper。
- `playerone_repro/rendering.py`
  场景状态 / scene-conditioner 抽象。仓库里提供了预计算 point map renderer，方便在没有 CUT3R backend 时调试新的自回归闭环。
- `playerone_repro/scene_conditioners.py`
  内置 scene-conditioner。当前提供一个可直接落地的 `HistoryStructureSceneConditioner`，会从已生成视频中抽取 `depth/normal/canny` 结构历史，作为下一段 chunk 的 scene 条件。
- `playerone_repro/pipeline.py`
  推理管线，直接复用 `DiffSynth-Studio` 的 `WanVideoPipeline.from_pretrained` 来加载基础模型。
  当前默认走 `Wan2.2-TI2V-5B` 的单 DiT I2V 自回归续写流程。
- `playerone_repro/training.py`
  stage-1 LoRA 预训练、stage-2 最后 6 个 block 微调，以及新的 chunked autoregressive flow-matching loss。
- `scripts/smoke_test.py`
  不下载大模型，直接用一个 tiny `WanModel` 验证 shape-flow。
- `scripts/infer_playerone.py`
  默认用真实 `Wan2.2-TI2V-5B` 权重跑这套结构，可通过参数覆盖成其他兼容的单 DiT Wan 2.2 仓库。

## 快速验证

先跑结构烟测：

```bash
python3 scripts/smoke_test.py
```

如果你后面要真跑推理，直接用：

```bash
python3 scripts/infer_playerone.py \
  --first-frame path/to/frame.png \
  --motion path/to/motion.npz \
  --chunk-frames 49 \
  --prompt "first person view, walking forward in a kitchen"
```

如果你想直接启用内置结构 scene-conditioner，可以这样跑：

```bash
python3 scripts/infer_playerone.py \
  --first-frame path/to/frame.png \
  --motion path/to/motion.npz \
  --chunk-frames 49 \
  --prompt "first person view, walking forward in a kitchen" \
  --scene-conditioner-factory playerone_repro.scene_conditioners:build_structural_scene_conditioner
```

这个脚本会按需从 Hugging Face / ModelScope 拉对应的 Wan 2.2 权重。默认流程是：

1. 用首帧 + 第一段 motion 先生成一个 chunk。
2. 用 renderer 更新 scene point cloud / point maps；如果没接 renderer，则自动把已生成 chunk 编码进 latent scene memory。
3. 用累计 scene 条件 + 下一段 motion 继续生成下一个 chunk。
4. 重复直到拼完整段视频。

当前仓库没有内置 CUT3R renderer backend，所以如果你想真实跑“生成后再 render 点云”的闭环，需要在代码里传入自定义 `BasePointMapRenderer` / `BaseSceneConditioner` 实现。命令行里的 `--point-maps` 主要用于调试预计算的 point map 序列；如果你已经有外部 renderer，也可以通过 `--scene-conditioner-factory module.path:callable --scene-conditioner-config config.json` 挂进去。  
如果你手头有兼容单 DiT 的 `Wan2.2 7B` 仓库，并且文件布局和 `Wan2.2-TI2V-5B` 一致，也可以额外传 `--wan-model-id your/model-id` 复用同一套加载逻辑。双 DiT 的 A14B 系列目前还没有接进这份 PlayerOne wrapper。
