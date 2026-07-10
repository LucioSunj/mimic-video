# mimic-video 模型架构、训练流程与推理数据流

本文档基于当前仓库代码整理，重点解释 mimic-video 中 Video-Action Model（VAM）的实际实现路径。核心代码主要在 `model/cosmos_predict2/`，训练入口为 `model/scripts/train.py`，评测和机器人闭环封装在 `eval/libero/run.py` 与 `eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py`。

## 1. 总体思路

mimic-video 的策略不是直接从图像和语言回归动作，而是把机器人控制拆成两层：

```text
历史图像 + 语言指令
        |
        v
Video2World 视频扩散/流模型
        |
        | 选定 denoising step 和中间层 hidden state
        v
World2Action 动作扩散/流解码器
        |
        v
未来动作 chunk
```

第一层是 Cosmos-Predict2 Video2World DiT。它建模“给定当前图像和语言，未来世界会怎样变化”，输出未来视频潜变量，或者更重要地输出某一层的中间 token 表征。

第二层是 World2Action DiT。它不重新理解原始图像，而是通过 cross-attention 读取 Video2World 的视频 token 表征，再结合当前机器人低维状态，生成未来一段动作。

这样做的关键好处是：动作解码器可以利用大视频模型中的世界动态知识，而且训练动作解码器时可以冻结 Video2World，只学习较小的 action decoder。

## 2. 主要代码层级

训练和推理都围绕同一组模块，但外层包装不同。

| 层级 | Video2World | World2Action | 作用 |
| --- | --- | --- | --- |
| 训练外壳 | `Predict2Video2WorldModel` | `World2ActionModel` | 定义训练目标、loss、日志、checkpoint state_dict |
| Pipeline | `Video2WorldPipeline` | `World2ActionPipeline` | 加载权重、tokenizer、normalizer、denoising/sampling |
| 网络 | `MinimalV1LVGDiT` | `World2ActionDIT` | 纯 DiT 前向，不知道训练循环和数据读取 |
| 数据 | `Dataset` | `MimicDataset` | 分别读取视频数据和视频+动作 zarr 数据 |
| 组合推理 | `Video2World2ActionPipeline` | 同左 | 将视频模型 hidden state 接给动作模型 |

相关文件：

- `model/cosmos_predict2/pipelines/video2world.py`
- `model/cosmos_predict2/pipelines/world2action.py`
- `model/cosmos_predict2/pipelines/video2world2action.py`
- `model/cosmos_predict2/models/video2world_model.py`
- `model/cosmos_predict2/models/world2action_model.py`
- `model/cosmos_predict2/models/video2world_dit.py`
- `model/cosmos_predict2/models/world2action_dit.py`

## 3. 数据与符号约定

常用张量维度：

| 符号 | 含义 |
| --- | --- |
| `B` | batch size |
| `C` | 图像或 latent channel |
| `T` | 视频帧数或 latent 时间长度 |
| `H, W` | 空间高宽 |
| `HO` | observation lowdim horizon |
| `HA` | action horizon |
| `A` | 动作维度，当前配置为 10 |
| `D` | transformer hidden width |
| `sigma` | Video2World 的噪声尺度 |
| `t` | World2Action 的 action flow 时间 |

默认机器人低维动作维度为 `A=10`：

```text
3D 末端位置 + 6D 旋转表示 + 1D gripper
```

图像输入统一为 `480 x 640`，值域通常为 `[-1, 1]`，布局为 `[B, C, T, H, W]`。

## 4. Video2World 架构

### 4.1 输入与 tokenizer

Video2World 接收历史图像和语言条件。训练或动作解码训练中，常见视频序列为：

```text
obs/workspace_rgb:    5 帧历史图像
action/workspace_rgb: 56 帧未来图像
拼接后:               61 帧视频
```

Video tokenizer 在 `TokenizerInterface` 中封装，使用 Cosmos VAE，把像素视频编码成 latent：

```text
像素视频: [B, 3, 61, 480, 640]
latent:  [B, 16, 16, 60, 80]
```

这里空间压缩因子为 8，时间压缩关系为：

```text
latent_frames = 1 + (pixel_frames - 1) // 4
61 pixel frames -> 16 latent frames
5 pixel frames  -> 2 latent frames
1 pixel frame   -> 1 latent frame
```

推理时如果只输入 1 或 5 帧历史图像，`Video2WorldPipeline.encode()` 会只编码条件前缀，并把剩余 latent 时间位置填零；真正生成部分由 denoising sample 提供。

### 4.2 条件构造

Video2World 的条件由 `VideoConditioner` 产生，配置在 `config_video2world.py`：

- `obs/language_embedding` 通过 `TextAttr` 变成 `crossattn_emb`。
- `fps` 和 `padding_mask` 通过 `ReMapkey` 传入模型。
- `use_video_condition` 通过 `BooleanFlag` 控制是否使用视频条件。
- `condition_video_input_mask_B_C_T_H_W` 标记哪些 latent frame 是条件帧。

条件帧策略为 `frame_replace`：

```text
denoising 输入中的前 N 个条件 latent frame 被替换成真实观测 latent
非条件 frame 仍由噪声 sample 演化
```

`MinimalV1LVGDiT` 还会把条件 mask 作为额外 channel 拼到输入上，让网络明确知道哪些位置是条件帧。

### 4.3 Video2World DiT

Video2World 网络是 `MinimalV1LVGDiT`，继承自 `MiniTrainDIT`。默认 2B 配置在 `config_video2world.py`：

| 配置项 | 默认值 |
| --- | --- |
| latent channels | 16 |
| latent time length | 16 |
| DiT hidden width | 2048 |
| blocks | 28 |
| attention heads | 16 |
| spatial patch size | 2 |
| temporal patch size | 1 |
| positional encoding | learnable 3D RoPE |
| AdaLN-LoRA | enabled, dim 256 |
| language cross-attention dim | 1024 |
| video hidden state dim | 2048 |

前向过程：

```text
1. 输入 latent x_B_C_T_H_W
2. 拼接 padding mask 和 condition mask channel
3. PatchEmbed:
   [B, C, 16, 60, 80] -> [B, 16, 30, 40, 2048]
4. 生成 3D RoPE
5. timestep embedding 经 AdaLN 调制每个 block
6. 每个 block:
   self-attention over video tokens
   cross-attention to language embedding
   MLP
7. FinalLayer unpatchify:
   [B, 16, 30, 40, hidden] -> [B, 16, 16, 60, 80]
```

Video2World 的 `denoise()` 输出 `DenoisePrediction`，其中包括：

- `x0`: 预测的干净 video latent。
- `eps`: 推导出的噪声预测。
- `hidden_states`: 可选的中间层 hidden state，用于动作解码。
- `sample_decoded_video`: 可选解码后视频，仅调试或可视化时使用。

## 5. World2Action 架构

World2Action 是本仓库新增的动作解码器，核心是 `World2ActionDIT`。

### 5.1 配置

默认配置在 `world2action_pipe.py`：

| 数据集 | `max_horizon` | `HO` | `HA` | 动作频率 |
| --- | ---: | ---: | ---: | --- |
| Bridge | 16 | 1 | 15 | 5 Hz |
| LIBERO | 61 | 1 | 60 | 20 Hz |

通用网络配置：

| 配置项 | 默认值 |
| --- | --- |
| input/action dim | 10 |
| hidden width | 1024 |
| blocks | 24 |
| heads | 8 |
| MLP ratio | 4.0 |
| video context dim | 2048 |
| AdaLN-LoRA | enabled, dim 128 |
| action denoising steps | 10 |
| action scheduler | BetaScheduler(alpha=1, beta=1) |

### 5.2 输入 token

World2Action 的输入由两段 token 拼成：

```text
obs token:    state_B_HO_O
action token: xt_B_HA_A
拼接后:       [B, HO + HA, D]
```

其中：

- `state_B_HO_O` 是当前或历史机器人低维状态。
- `xt_B_HA_A` 是 action flow 当前时间 `t` 下的 noisy action sample。
- 两者分别通过 `ActionEmbedder` 从 10 维映射到 1024 维。
- 拼接后加 learned positional embedding。
- 训练时 `obs_dropout=0.2`，部分 observation token 会被 `obs_mask_token` 替换。

### 5.3 视频条件 cross-attention

动作解码器不直接处理原始图像。它读取 Video2World 某层 hidden state：

```text
Video2World hidden state: [B, T, H, W, 2048]
reshape:                 [B, T * H * W, 2048]
```

默认动作训练实验使用 `xattn_layer_idx=20`，也就是取 Video2World 第 20 层附近的中间表征。

World2Action 每个 block 的顺序是：

```text
1. cross-attention: action/obs token query，Video2World token key/value
2. self-attention: action/obs 序列内部建模
3. MLP
```

每个子层都有 AdaLN 的 shift、scale、gate 调制。

### 5.4 双时间调制

World2Action 同时知道两个“时间”：

```text
action timestep t_action: 当前动作 denoising 进度
video sigma:              Video2World hidden state 对应的噪声尺度
```

`PairTimeEmbedder` 分别对二者做连续 sinusoidal embedding，再通过 `PairTimestepEmbedding` 做低秩双线性交互，产生：

- 给 block 使用的时间 embedding。
- 给 AdaLN-LoRA 使用的 `adaln_lora_B_T_3D`。

这就是仓库 README 中提到的 decoupled flow times：视频 denoising 的 `sigma` 和动作 flow 的 `t` 是分开的，但动作模型能同时感知二者。

### 5.5 输出

`World2ActionDIT` 输出完整序列的 velocity field：

```text
vt_pred_B_T_A: [B, HO + HA, 10]
```

`World2ActionPipeline.denoise()` 会丢掉 observation 部分，只保留 action 部分：

```text
vt_pred_B_HA_A = vt_pred_B_T_A[:, HO:, :]
```

推理结束后通过 normalizer 反归一化得到真实动作尺度。

## 6. 训练流程

训练分为两个主要阶段：

```text
阶段 1: Video2World 微调
阶段 2: World2Action 动作解码器训练
```

两者共用 `ImaginaireTrainer` 外层训练循环，但数据、模型 wrapper 和 loss 不同。

### 6.1 通用训练入口

命令入口：

```bash
torchrun -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=...
```

执行流程：

```text
scripts/train.py
  -> make_config()
  -> override experiment
  -> distributed.init()
  -> instantiate(config.model)
  -> instantiate(config.dataloader_train)
  -> trainer.train(model, dataloader_train, dataloader_val_cfg)
```

`ImaginaireTrainer.train()` 负责：

- 初始化分布式、DDP/FSDP。
- 保存 effective config。
- 调用 `model.on_train_start()`。
- 创建 optimizer、scheduler、grad scaler。
- 加载 checkpoint。
- 循环读取 batch。
- 调用 `model.training_step(data_batch, iteration)`。
- backward、optimizer step、scheduler step。
- EMA 更新、checkpoint、validation callback。

### 6.2 Video2World 微调数据

Video2World 数据来自 mp4 和预计算 T5 embedding：

```text
dataset/
  video/ep.mp4
  metas/ep.txt
  t5_xxl/ep.pickle
```

`dataset_video.py` 读取视频时：

1. 随机选 episode。
2. 随机选时间点。
3. 按 5 Hz 采样总共 61 帧。
4. 不足边界时重复首帧或末帧 padding。
5. resize 到 `480 x 640`。
6. 读取 `t5_xxl/*.pickle`，补齐到 Cosmos T5 token 数。

训练 batch 包含：

```text
video:                  [B, C, 61, 480, 640]
obs/language_embedding: [B, N_text, D_text]
fps
padding_mask
```

### 6.3 Video2World 微调目标

`Predict2Video2WorldModel.training_step()` 的核心步骤：

```text
1. pipe.get_data_and_condition(data_batch)
   - 视频归一化到 [-1, 1]
   - tokenizer.encode -> latent x0
   - conditioner 构造语言和条件帧 mask

2. draw_training_sigma_and_epsilon()
   - sigma ~ exp(N(0, 1))
   - epsilon ~ N(0, I)
   - 视频 batch 可乘 sqrt(state_t) 做 noise adjustment

3. xt = x0 + sigma * epsilon

4. pipe.denoise(xt, sigma, condition)
   - RectifiedFlowScaling:
     t = sigma / (sigma + 1)
     c_skip = 1 - t
     c_out  = -t
     c_in   = 1 - t
     c_noise = t
   - DiT 预测 net_output
   - x0_pred = c_skip * xt + c_out * net_output

5. loss = weighted MSE(x0_pred, x0)
   weight = ((1 + sigma)^2 / sigma^2)
```

实验配置在 `experiment/video2world.py`，当前默认微调方式是 LoRA：

- rank 256。
- learning rate 约 `1.778e-4`。
- global batch size 32。
- 目标模块包括 attention q/k/v/output、patch embed、timestep MLP、MLP 等。

### 6.4 World2Action 训练数据

动作训练数据来自 zarr，每个 episode 中存多模态序列和对应 timestamps。配置在 `configs/dataloading/`。

最终 batch 主要字段为：

```text
obs/language_embedding
obs/lowdim_concat
obs/workspace_rgb
action/lowdim_concat
action/workspace_rgb
```

Bridge 默认：

```text
obs/workspace_rgb:       5 帧，5 Hz
action/workspace_rgb:    56 帧，5 Hz
obs/lowdim_concat:       1 x 10
action/lowdim_concat:    15 x 10
```

LIBERO 默认：

```text
obs/workspace_rgb:       5 帧，5 Hz
action/workspace_rgb:    56 帧，5 Hz
obs/lowdim_concat:       1 x 10
action/lowdim_concat:    60 x 10，20 Hz
```

`ChunkReader` 做时间对齐：

- 以 `workspace_rgb` 时间戳为 anchor。
- 对每个 modality 按自己的 horizon 和 target frequency 插值或取最近值。
- observation chunk 向过去取，action chunk 向未来取。
- episode 边界通过重复首/末值 padding。

`data_transforms.py` 做后处理：

- 图像 resize 并转为 `[-1, 1]`。
- pose matrix 拆成 position 和 rotation。
- rotation matrix 转 6D 表示。
- Bridge action pose 转为相对当前 proprio pose。
- 低维字段 flatten 后 concat 成 `*_lowdim_concat`。

### 6.5 World2Action normalizer

动作模型训练开始时，`World2ActionModel.on_train_start()` 会读取数据集统计量：

```text
dataloader_train.dataset.get_statistics()
```

`StaticBatchNormalizer` 按配置归一化低维字段：

- `VARIANCE`: clip outliers，然后减均值、除以标准差。
- `IDENTITY`: 不改变数值。
- `NONE`: 不建立 normalizer，图像和语言不归一化。

concat group 会把各字段的 normalizer 参数拼成：

```text
obs/lowdim_concat
action/lowdim_concat
```

推理时也必须加载同一份 dataset statistics json，否则动作尺度会错。

### 6.6 World2Action 训练目标

`World2ActionModel.training_step()` 的核心步骤：

```text
1. data_batch["obs/language_embedding"] squeeze

2. get_crossattn_emb(data_batch)
   - 把 obs/workspace_rgb 和 action/workspace_rgb 拼成 61 帧视频
   - tokenizer.encode -> video latent x0_video
   - 采样 video_sigma
   - 加噪: x_video = x0_video + sigma * epsilon
   - 冻结的 Video2World 只前向到 xattn_layer_idx
   - 取 hidden state 并 reshape 为 [B, N_video_tokens, 2048]

3. normalizer(data_batch)
   - 归一化 obs/lowdim_concat 和 action/lowdim_concat

4. action flow 加噪
   - x0_action = normalized action/lowdim_concat
   - t ~ Beta(alpha=1, beta=1)，再限制到 [0.001, 1.0)
   - epsilon ~ N(0, I)
   - xt = (1 - t) * x0_action + t * epsilon
   - 目标 velocity: ut = epsilon - x0_action

5. World2ActionPipeline.denoise()
   - observation token 的 timestep 固定为 1e-3
   - action token 使用采样的 t
   - cross-attention 读取 Video2World hidden state
   - 输出 vt_pred

6. loss = MSE(vt_pred, ut) * loss_scale
```

默认 `loss_scale=10.0`。

需要注意的是，动作解码器训练时 Video2World 是冻结的：

```text
self.video2world_pipe.requires_grad_(False)
```

因此训练主要更新 `World2ActionDIT`。代码保留了 LoRA 训练选项，但当前实验配置默认是 action decoder 全参数训练。

### 6.7 World2Action validation

validation 会看两类误差：

1. `gtvid/full`: 使用真实未来视频 latent，加不同 sigma 噪声，然后取 Video2World hidden state 解码动作。
2. `genvid/full`: 先从历史图像真正跑 Video2World 采样，再在每个 video denoising step 抽取 hidden state 解码动作。

这能同时评估：

- 动作解码器在“真实未来视频上下文”下是否学到动作。
- 动作解码器在“模型生成的视频上下文”下是否能工作。

## 7. 推理时的数据流

推理主要由 `Video2World2ActionPipeline` 组合完成。

### 7.1 加载模型

LIBERO 和 Bridge 的加载逻辑基本一致：

```text
make_config()
override experiment=...
disable guardrails
load Video2WorldPipeline(video checkpoint)
load World2ActionPipeline(action checkpoint)
instantiate data_config
load dataset_statistics json
world2action_pipe.normalizer.build_from_stats(...)
return Video2World2ActionPipeline(...)
```

相关入口：

- LIBERO: `eval/libero/run.py`
- Bridge: `eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py`

### 7.2 环境观测预处理

每个环境 step：

1. 获取相机图像。
2. 转成 `[C, 1, H, W]`。
3. 归一化到 `[-1, 1]`。
4. 加入图像历史队列。
5. 获取机器人 proprio state。
6. position + rotation 6D + gripper 拼成 10 维 lowdim。
7. 加入低维历史队列。

当 action buffer 为空时，才重新查询模型生成一个动作 chunk。评测脚本默认每次执行前 5 个动作，然后重新规划。

### 7.3 Video2World 产生视频上下文

`Video2World2ActionPipeline.__call__()` 接收：

```text
input_vid:     [B, 3, 1 or 5, 480, 640]
state_B_HO_O:  [B, HO, 10]
prompt:        language instruction
```

处理流程：

```text
1. 检查 prompt guardrail，评测里默认关闭。

2. 根据输入帧数选择 latent conditional frames:
   1 pixel frame -> 1 latent condition frame
   5 pixel frames -> 2 latent condition frames

3. Video2WorldPipeline.generate_video()
   - 构造 data_batch
   - tokenizer.encode 输入历史图像
   - 随机初始化完整 16 latent frame 的 noisy sample
   - Karras-like sigma schedule, 默认 35 steps
   - 每个 step 调用 Video2World DiT 预测 x0

4. 在 stop_after_step 指定的 video denoising step:
   - 返回 hidden_states[xattn_layer_idx]
   - 同时返回该 step 的 sigma
```

`stop_after_step` 是重要超参。评测脚本会 sweep `0..35`。当 `stop_after_step=0` 时，只需要第一步 video denoising 的一次 Video2World 前向就能给动作解码器提供上下文；更大的值会运行更多 video denoising step，视频上下文更接近干净未来视频，但计算更重。

返回的 hidden state 会 reshape：

```text
[B, T, H, W, 2048] -> [B, T * H * W, 2048]
```

然后作为 World2Action 的 cross-attention context。

### 7.4 World2Action 生成动作 chunk

World2Action 推理入口是 `World2ActionPipeline.__call__()`：

```text
输入:
  state_B_HO_O
  crossattn_emb
  context_timesteps_B_1 = video_sigma

输出:
  unnormalized actions [B, HA, 10]
```

具体步骤：

```text
1. 归一化 state_B_HO_O。

2. 初始化动作 sample:
   sample_B_HA_A ~ N(0, I)
   timestep_B_HA_1 = 1

3. 运行 10 步 Euler denoising:
   vt_pred = World2ActionDIT(...)
   sample = sample + dt * vt_pred
   timestep = timestep + dt
   其中 dt = -1 / 10

4. 反归一化 action/lowdim_concat。
```

因为训练目标是：

```text
xt = (1 - t) * x0 + t * epsilon
ut = epsilon - x0
```

所以推理从 `t=1` 的纯噪声开始，沿负时间方向积分到 `t=0`，得到动作样本。

### 7.5 动作后处理与执行

LIBERO：

```text
action[0:3]   -> delta position
action[3:9]   -> 6D rotation -> rotation matrix -> rotvec
action[9]     -> sign -> gripper
拼接为环境动作 [dx, dy, dz, rx, ry, rz, gripper]
```

Bridge / SimplerEnv：

```text
action[0:3], action[3:9] 先转成目标绝对 pose
根据当前 pose 计算 rot_axangle 和 world_vector 控制量
action[9] 映射为 gripper
terminate_episode 固定为 0
```

Bridge 代码还维护 `_plan_abs_Ts`，把预测的相对 pose delta 累乘到当前观测 pose 上，得到未来每一步目标绝对位姿。

### 7.6 HIL / oracle future video 路径

`video2world2action_gtvid.py` 是 human-in-the-loop 或 oracle future video 版本：

```text
input_vid + gt_future_vid -> 拼成真实 61 帧视频
tokenizer.encode
按 stop_after_step 取 sigma
对真实未来视频 latent 加对应 sigma 噪声
Video2World denoise 一次并取 hidden state
World2Action 解码动作
```

这条路径不跑完整视频生成循环，用于分析“如果未来视频上下文是真实的，动作解码能达到什么水平”。

## 8. 端到端训练与推理对照

### 8.1 动作训练时的数据流

```text
zarr episode
  |
  v
ChunkReader 按 timestamp 对齐
  |
  v
data_transforms:
  image -> [-1, 1]
  pose -> relative / 6D rot
  lowdim concat
  |
  v
batch:
  obs/workspace_rgb
  action/workspace_rgb
  obs/lowdim_concat
  action/lowdim_concat
  obs/language_embedding
  |
  v
Video2World frozen:
  真实 obs+future video latent + sampled sigma noise
  -> hidden state at layer 20
  |
  v
World2Action:
  normalized noisy action xt
  + normalized obs state
  + video hidden state
  -> velocity vt
  |
  v
MSE(vt, epsilon - action)
```

### 8.2 推理时的数据流

```text
环境当前图像、proprio、语言任务
  |
  v
图像历史队列 + lowdim 历史队列
  |
  v
Video2World:
  历史图像作为条件帧
  随机 future latent sample
  denoise 到 stop_after_step
  返回 hidden state + sigma
  |
  v
World2Action:
  从动作噪声开始
  10 步 Euler flow
  反归一化
  |
  v
动作 chunk buffer
  |
  v
每个环境 step 执行前 num_execute_actions 个动作
```

## 9. 关键实现细节

### 9.1 为什么不一定要解码未来视频

`Video2WorldPipeline.generate_video()` 正常可以返回解码后视频，但 VAM 策略真正需要的是中间 hidden state。`Video2World2ActionPipeline` 使用 `return_context_at_step` 直接拿 hidden state，避免把 latent decode 成像素视频。

### 9.2 为什么动作解码器要接收 video sigma

训练时动作解码器看到的 Video2World hidden state 来自不同 `sigma` 的 noisy video latent。推理时 hidden state 也可能来自任意 `stop_after_step`。如果动作模型不知道该 hidden state 对应的视频噪声水平，分布会混在一起。`context_timesteps_B_1` 把 video sigma 显式传给 `PairTimeEmbedder`，让动作模型适配不同视频 denoising 阶段。

### 9.3 为什么训练时用真实未来视频，推理时用生成上下文

动作训练中用真实 `obs + future` 视频加噪，再通过冻结 Video2World 抽 hidden state。这提供稳定监督，避免动作训练同时受视频生成误差影响。validation 和真实推理再检验 generated video context 下是否仍然有效。

### 9.4 Bridge 与 LIBERO 的主要差异

| 项 | Bridge | LIBERO |
| --- | --- | --- |
| action horizon | 15 | 60 |
| action frequency | 5 Hz | 20 Hz |
| execute actions | eval 默认 5 | eval 默认 5 |
| action pose | 相对当前 proprio pose | delta/ref lowdim |
| 环境动作格式 | SimplerEnv world vector + axis-angle + gripper | LIBERO delta pos + rotvec + gripper |
| video history | 5 帧 | 原始 20 Hz 队列下采样到 5 帧 |

## 10. 代码索引

| 主题 | 文件 |
| --- | --- |
| 训练入口 | `model/scripts/train.py` |
| 通用训练循环 | `model/imaginaire/trainer.py` |
| Video2World model wrapper | `model/cosmos_predict2/models/video2world_model.py` |
| World2Action model wrapper | `model/cosmos_predict2/models/world2action_model.py` |
| Video2World pipeline | `model/cosmos_predict2/pipelines/video2world.py` |
| World2Action pipeline | `model/cosmos_predict2/pipelines/world2action.py` |
| 组合推理 pipeline | `model/cosmos_predict2/pipelines/video2world2action.py` |
| HIL / gt future video pipeline | `model/cosmos_predict2/pipelines/video2world2action_gtvid.py` |
| Video DiT | `model/cosmos_predict2/models/video2world_dit.py` |
| Action DiT | `model/cosmos_predict2/models/world2action_dit.py` |
| Video2World 配置 | `model/cosmos_predict2/configs/config_video2world.py` |
| World2Action 配置 | `model/cosmos_predict2/configs/defaults/world2action_pipe.py` |
| Video training experiments | `model/cosmos_predict2/configs/experiment/video2world.py` |
| Action training experiments | `model/cosmos_predict2/configs/experiment/world2action.py` |
| Video dataset | `model/cosmos_predict2/data/dataset_video.py` |
| Action dataset | `model/cosmos_predict2/data/action/dataset_action.py` |
| Chunk timestamp reader | `model/cosmos_predict2/data/action/chunk_reader.py` |
| Action transforms | `model/cosmos_predict2/data/action/data_transforms.py` |
| Normalizer | `model/cosmos_predict2/module/normalizer.py` |
| LIBERO inference | `eval/libero/run.py` |
| Bridge inference | `eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py` |
