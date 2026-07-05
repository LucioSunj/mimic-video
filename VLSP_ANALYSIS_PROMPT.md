# VLSP 项目分析请求 (Video-Latent Source Prior for Action Flow)

> 用途:发送给独立分析 agent 的自包含 prompt。仓库内配套文档:
> `VLSP.md`(方法/配置/协议)、`VLSP_RUN1_ANALYSIS.md`(Run-1 证据与诊断)、
> `VLSP_EXPERIMENT_PLAN.md`(分阶段实验计划)。本文件与它们保持一致,冲突时以
> 仓库文档 + 训练标量为准。

你是一个研究分析 agent。请基于以下完整背景,回答文末 §5 的分析问题。

## 1. Motivation

在 flow-matching 动作模型中,动作由「从 source 端点积分到动作」生成,基线 source 是高斯噪声
(`randn_like(action)`)。VLSP 的假设:视频世界模型的(部分去噪)latent 已经包含对未来动作的
预测信息,应当用它来**初始化**动作生成过程(定义 flow 的 source 端点),而不只是当 cross-attn
条件。即:

    x_t = (1-t)·a + t·s      u_t = s − a      L = ||v_θ(x_t,t,c) − u_t||² + λ·KL
    baseline: s ~ N(0,I)     VLSP: s ~ q_φ(s | video_latent, ...)   # 归一化动作空间

相关工作:VITA (ICLR 2026, arXiv:2507.13231) 用视觉 latent 直接当 flow source(noise-free、
conditioning-free,靠动作 AE + 重建/对比/一致性锚定);A2A (RoboVerse) 用历史状态编码当 source、
视觉当条件(source 与 target 不同源,结构上防坍缩)。VLSP 与两者的差异:flow target 是**原始
GT 动作**(无动作 AE),source 网络自由可训——这带来独有的坍缩风险(见 §3)。

## 2. Implementation(细化到算子与张量维度)

### 2.1 共同的 flow 约定(两个 backbone 一致)
训练:`x_t = (1−t)·x0 + t·s`,回归目标 `u_t = s − x0`(velocity),loss = MSE。
推理:从 `s`(t=1)出发,Euler 积分 t:1→0。**VLSP 只替换 s 的来源**,插值式、
target 公式、采样循环全部不动;`s` 与 `x0` 同形状、同归一化动作空间。

### 2.2 mimic-video 的 source prior 网络(VideoLatentSourcePrior)

**输入张量**(LIBERO 配置,B=batch):
- 视频 latent = 冻结 Cosmos video DiT 在 `xattn_layer_idx=20` 的 hidden states:
  `[B, T_lat, H_lat, W_lat, 2048]` → reshape → `crossattn_emb [B, N, 2048]`,N=T·H·W
  (训练时该 latent 处于随机采样的视频噪声级 σ_v;`detach_video_latents=true` 切断回传)。
- 机器人状态 `state [B, HO, 10]`(HO=观测步数);视频噪声级 `σ_v [B, 1]`;
- GT 动作 `x0 [B, HA, 10]`,HA = max_horizon(61) − HO,动作维 A=10。

**前向逐步维度变换**(hidden=1024):
1. `ctx_norm = LayerNorm(2048)`:`[B,N,2048] → [B,N,2048]`。
2. **池化(3 选 1,输出统一 `h [B,1024]`)**:
   - `mean`:token 维求均值 `[B,N,2048] --mean(dim=1)--> [B,2048] --Linear(2048→1024)--> [B,1024]`;
   - `attention`:`kv=Linear(2048→1024)` 得 `[B,N,1024]`;1 个可学 query `[1,1,1024]`
     expand→`[B,1,1024]`;`MultiheadAttention(1024, 8头, batch_first)` 做 1×N cross-attn
     → `[B,1,1024] --取[:,0]--> [B,1024]`;
   - `perceiver`:8 个可学 latent `[1,8,1024]`→`[B,8,1024]` 对 `kv [B,N,1024]` cross-attn
     → 残差相加 → 残差 MLP → `[B,8,1024] --mean(dim=1)--> [B,1024]`。
3. **条件注入(逐项加到 h 上,均可开关)**:
   - state:`[B,HO,10] --mean(dim=1)--> [B,10] --Linear(10→1024)--> [B,1024]`,h += ;
   - 视频噪声级:`σ_v [B,1] --σ/(1+σ)--> [B] --正弦嵌入(1024维)--> [B,1024] --Linear(1024→1024)-->`,h += ;
   - 语言(默认关):`[B,L,D_lang] --mean(dim=1)--> [B,D_lang] --LazyLinear(→1024)-->`,h += 。
   缺失输入时走 `0.0·proj(zeros)` 零系数分支,保证参数始终在计算图里(DDP/FSDP 安全)。
4. **horizon 感知展开**:可学查询 `horizon_queries [1,61,1024]` 切片 `[:, :HA]`,与
   `h.unsqueeze(1) [B,1,1024]` 广播相加 → `[B,HA,1024]`;过残差 trunk
   (`LayerNorm → depth×(Linear 1024→1024 + GELU)`,残差连接)→ `[B,HA,1024]`。
   ⇒ 每个动作步有独立表征,而非单向量盲广播。
5. **双头输出**:`mu_head Linear(1024→10)` ⇒ `μ [B,HA,10]`;
   `logstd_head Linear(1024→10)` ⇒ clamp 到 `[logstd_min=−5, logstd_max=1]` ⇒ `logσ [B,HA,10]`。
   初始化:μ 头全零(初始 μ≡0)、logstd bias=init_logstd(−1) ⇒ 初始 q≈N(0, e⁻²·I)。

**source 生成(全部在 `[B,HA,10]` 上逐元素)**,ε~N(0,I) 同形:
- `video_prior_sample`: s = μ + temperature·e^{logσ}·ε(reparameterized)
- `video_prior_mean`:  s = μ(确定性)
- `video_prior_residual`: s = ε + residual_scale·μ
- `video_prior_blend`: s = α·(μ+e^{logσ}ε) + √(1−α²)·ε′
- `video_prior_dropout`: s = m·s_video + (1−m)·ε′,m~Bernoulli(1−p),
  粒度 sample→`[B,1,1]` / element→`[B,HA,10]` 广播
- `shuffled_video_prior`: 先 `crossattn_emb[randperm(B)]` 再走 sample(负对照)
- 恒加 `s += 0.0·(μ.mean()+logσ.mean())`:数值为零,只为让两头始终留在 autograd 图
  (DDP `find_unused_parameters=False` / 确定性模式安全)。

**训练接入**(`world2action_model.training_step`):先采 source(gaussian 时 =原 epsilon,
**保持原 RNG 顺序**:source 在 t 之前抽)→ 再采 t → `x_t=(1−t)x0+t·s`、`u=s−x0` →
DiT 前向(obs_dropout=0.2)→ `loss = 10·MSE + λ_KL·KL + ...`,
KL = ½(μ²+σ²−1−2logσ).mean()。**条件通路独立配置**:`prepare_action_condition` 对
crossattn_emb 做 normal(原样)/ zero_video(置零 `[B,N,2048]→0`)/ shuffled_video
(batch 置换)—— source 输入与 DiT 条件互不影响,这就是 B/C 实验的实现基础。

**推理接入**(`World2ActionPipeline.__call__`):原 `arch_invariant_rand` 画的
`[B,HA,10]` 起点替换为 `sample_action_source(...)`;之后 10 步 Euler(t:1→0,dt=−0.1)
的去噪循环、以及最终 unnormalize 完全不动。**一个与坍缩分析相关的细节**:`denoise()`
在进 DiT 前做 `x_t / √((1−t)²+t²)` 的单位方差重标定——该公式隐含 **Var[s]=1** 的假设;
σ 坍缩(Var[s]≈0)时 t≈1 处的实际输入尺度 ≈ 期望的 1/√2 倍,重标定失配。

**诊断指标的精确定义**(hardening 后新增):
- `source/mu_batch_std = μ.std(dim=0).mean()`:先对 **batch 维**求 std(逐 (step,dim)),
  再平均 ⇒ ≈0 意味着「μ 对输入不敏感/所有视频同一条轨迹」(Run-1 的盲点;旧的全局
  `mu_std=μ.std()` 分不清这个);另有 `mu_horizon_std = μ.std(dim=1).mean()`(跨步多样性)。
- `source/logstd_floor_frac = (logσ ≤ −5+0.05).float().mean()`:钉在 clamp 下限的元素占比
  (Run-1 全程 ≈1.0)。
- 其它:`source_vs_x0_mse`、`source_vs_gaussian_mse`、`std_mean/min/max`、
  训练中采样探针 `probe/sampled_action_mse_gtvid`(每 500 iter 完整采样一次算动作 MSE)。

**工程保障**:enabled=false ⇒ 模块零参数、逐位复现基线(含 RNG 流,已验证);
`enabled=true+mode=gaussian` 被显式拒绝(歧义配置);FSDP 下 source-prior 梯度手动
all-reduce;source prior 纳入 EMA;ckpt 以 `source_prior.*` 前缀非严格加载(旧 ckpt 兼容),
eval 路径 `from_config` 也会加载 prior 权重。

### 2.3 FastWAM 的移植(维度差异 + 接缝方式)

**输入/输出维度**(与 mimic-video 的关键不同):
- 动作 `[B, T, A]`:T=action_horizon,A=7(LIBERO)/14(RoboTwin);
- source prior 输入 = **干净首帧 VAE latent** `[B, 48, 1, h′, w′]`,h′=H/16
  (224×224 ⇒ 14×14),flatten `permute+reshape` → `[B, 196, 48]`——**注意 token 维只有
  48**(mimic-video 是 2048),信息容量低得多;
- prior 结构同 2.2(hidden=512,horizon_queries `[1,64,512]`,μ/logσ 头 `Linear(512→A)`),
  proprio 经 `Linear(proprio_dim→512)` 注入。

**接缝(Template-Method,+564/−0 行)**:base `FastWAM` 上加
`_make_action_source(gaussian, video_latent, ...)`——disabled 时**原样返回传入的
gaussian**(逐位等于基线);enabled 时把该 gaussian 复用为 reparam 的 ε(确定性/种子
从调用点继承)。三个变体的 randn 位点各改 1–2 行:base `training_loss`(用
`input_latents[:,:,0:1]`,即 VAE 编码 GT 视频的首帧——与推理用的 `first_frame_latents`
是同一分布)/ base `infer_joint`/`infer_action` / joint `infer_action` / idm
`training_loss`+`infer_joint`。prior 注册在 `self.mot` 下 ⇒ 既有 trainer(优化
`model.dit==mot`)与 ckpt(`mot.state_dict()`, strict=False)零改动。

**结构性差异(分析时重要)**:FastWAM 的 action 分支通过 MoT 混合注意力**每层都能看到
视频 token**(base 只看首帧 token,joint 看全部)——等效于「永远 source+cond」,不存在
mimic-video 的 zero_video 开关;因此「μ 无信息压力」问题在 FastWAM 结构上更严重,且
VLSP-future(部分去噪未来 latent 当 source)尚未实现,当前 source 只含首帧信息。

## 3. 实验现状(Run 1 修正版,关键!)

**先修正一个事实:Run-1 最初报告的「每个 rollout 执行固定的任务无关动作」是 eval 侧
实验设置问题造成的假象(已定位修复)。修复后重新评估的真实结果:**

- **C(source+cond):能正常生成动作,但成功率不高**(具体数字待补;且当时没有在
  同一 eval 设置下跑 baseline 对照,无法判断 C 是"低于基线"还是"≈基线")。
- **B(source-only):完全不行,动作明显怪异**(不是"合理但做错任务",而是运动学上
  就不像正常轨迹)。

**训练端证据不受 eval 修正影响,以下事实依然成立:**
- `source/logstd_mean` 钉死在 clamp 下限 −5.0(σ≈0.0067,B、C 两个 run 都是)
  ⇒ **source 方差坍缩为 per-input Dirac,确凿**。
- `source/source_vs_x0_mse` 随训练**上升**至 0.81–0.91,且 > Var[x0]≈0.55
  ⇒ μ 没有坍向 x0,甚至比预测数据均值更差 ⇒ 疑似 **input-independent(近常数)μ**。
- `loss/flow` 极小(0.006–0.03)⇒ 早期「快 10×」是 target 变可预测的坍缩伪象,不可与
  baseline 的 loss 直接比较(量纲不同)。

**修正后的联合解读(B/C 行为差分是新的强证据):**
1. **C 正常但弱** ⇒ cross-attn 条件通路是健康的:即使 source 退化成(近似)确定性起点,
   DiT 仍能靠条件学出任务相关速度场并积分出合理动作。坍缩没有"摧毁"模型,它的真实代价是
   ① source 不再提供信息增益(VLSP 的核心假设没兑现),② 失去高斯 source 的噪声鲁棒性
   (t≈1 处分布过窄,eval 起点对训练分布外的视频 latent 更脆弱)——这两条是 C 成功率
   不高的候选解释,但**尚不能与"训练量不足 / 生成视频 gap / 本就≈baseline"区分开**。
2. **B 彻底崩坏** ⇒ source 是 B 唯一的任务信息入口;B 动作怪异(而非"合理但错误")与
   「μ 近常数 + σ≈0」的预测一致:DiT 只能学"从一个固定起点到各种 x0 的平均场",eval 时
   输出不连贯/均值化的轨迹。**B 的失败是"μ 不携带视频信息"的行为学证据**,但仍需 D2
   探针在权重层面确认(喂 K 个不同视频 latent,测 μ 的 cross-input 差异)。
3. 结构性根因假设(维持):source+cond 训练中 **μ 没有任何携带视频信息的损失压力**
   (视频信息反正能从 cross-attn 进来,常数 μ 使 target 最可预测);kl_weight=0 则给了
   σ→0 的明确梯度激励。坍缩(σ)已证实,μ 无信息(μ)待 D2 定罪。
4. 当前结论边界:**VLSP 的正增益尚未在任何配置中显现**;C 的表现目前无法排除
   "全部功劳属于 cross-attn 条件、source 白搭甚至略有拖累"这一最保守解释。

## 4. 后续实验计划(修正版)

**第 0 优先:补齐归因所需的对照与诊断(全部便宜,先于任何重训)**
- **A' — baseline 同设置对照(最重要的缺口)**:用修复后的同一 eval 设置跑
  `vlsp_baseline_gaussian` 的成功率。没有这个数字,"C 成功率不高"无法归因。
- **D2 — prior 退化探针(必跑,CPU 数分钟)**:`scripts/vlsp_probe_prior.py` 加载 Run-1
  ckpt,K 个不同(真实)视频 latent 测 μ 输入敏感度 + logstd floor fraction。预期确认
  「σ 坍缩 + μ 近常数」;若 μ 其实 input-sensitive,则 B 的崩坏需要另找解释(优先查
  B 训练本身与 zero_video 下 DiT 的退化)。
- **D3 — stop-step 扫描(用现有 C ckpt)**:`stop_video_denoising_step ∈ {5,15,25,35}`
  看 C 成功率是否随视频去噪程度单调变化 ⇒ 分离"生成视频 gap"对低成功率的贡献。
- **C 失败模式定性**:抽看 C 的失败 rollout,标注是"接近成功的精度问题"还是"系统性
  偏差/早期就走错"——前者指向训练量/精修,后者指向分布性问题。

**Run 2 = R1–R3(维持,均 source+cond,同 seed/数据/eval),已注册:**
| Run | 实验名 | 改动 | 理由 |
|---|---|---|---|
| R1 主线 | `vlsp_r1_kl_1e3` | kl_weight=1e-3 | 消除 σ→0 梯度激励,source 保持 "hint" |
| R2 结构兜底 | `vlsp_r2_blend_050` | blend α=0.5 + KL | 不可坍缩的 0.87-std 高斯分量,最坏退化=baseline |
| R3 鲁棒 | `vlsp_r3_kl_dropout_020` | R1 + 20% source dropout | DiT 保留纯高斯模式 |

**注意:KL 只治 σ 坍缩,不治「μ 无信息压力」这个结构问题。** 因此新增两个候选
(是否采纳请分析 agent 评估,见 §5-Q3):
- **R4(候选)— source-only 训练配置作为 μ 的信息压力源**:`zero_video` 训练下 μ 是唯一
  信息通道,被迫学习 video→action;可作为 curriculum 第一阶段(先 source-only 预训 prior,
  再切回 source+cond)或独立验证「μ 到底能不能从该视频 latent 学出动作信息」。
- **R5(候选)— μ 显式监督**:辅助损失 `||μ − x0||²`(小权重,类 VITA 锚定在 raw-action
  空间的适配),直接给 μ 信息压力;与 KL 并用时注意两者拉扯方向相反,需权重调和。

**训练中 gate(维持并强化,2k/5k/10k iter)**:`logstd_mean > −2.5`;`logstd_floor_frac ≈ 0`;
`mu_batch_std` 明显 > 0(Run-1 的盲点指标);`probe/sampled_action_mse_gtvid` 下降且与
baseline 同 iter 可比(禁止拿原始 flow loss 跨 run 比较——量纲不同)。**新增行为学 gate**:
任何配置宣布"健康"前,必须在修复后的 eval 上给出与 A' 可比的成功率;B(source-only)线
保持挂起,直到某配置的 `mu_batch_std` 健康(否则 B 必然重蹈覆辙)。FastWAM 侧继续等
mimic-video 结论。

## 5. 请你分析的问题

1. **B/C 行为差分的解释是否唯一?** 「μ 近常数 + σ 坍缩」能同时解释"C 正常但弱、B 崩坏"
   吗?B 的怪异动作还有哪些竞争性解释(如 zero_video 训练下 DiT 本身退化、B 的坍缩时间线
   更早、B/C 除 conditioning 外的其它设置差异)?用什么最小实验区分?
2. **C 成功率低的归因分解**:候选因子 = ①Dirac source 的 eval 脆弱性(含 §2.2 提到的
   `x_t/√((1−t)²+t²)` 重标定在 Var[s]≈0 时失配)、②有效训练量不足(loss 早熟不代表学够)、
   ③生成视频 vs GT 视频 gap(D3)、④本就 ≈ baseline(待 A' 对照)。给出最小实验集把四者
   分开,并评估各自先验概率。
3. **KL 之外要不要治 μ?** R1–R3 只防 σ 坍缩;若 D2 确认 μ 无信息,只靠 KL 的 R1 即使
   "健康"也可能只是回到 ≈baseline(source≈N(0,1) 的白噪声)。R4(source-only 压力/curriculum)
   与 R5(μ 辅助回归)哪个更值得先做?有没有更好的给 μ 加信息压力的机制(对比学习、
   信息瓶颈、把 source 与条件做 dropout 互补而非独立)?
4. **指标体系还缺什么?** 现有:mu_batch_std、mu_horizon_std、logstd_floor_frac、
   source_vs_x0_mse、采样 MSE 探针 + 行为学 gate。是否需要:μ 与任务标签的可分性探针
   (linear probe)、source 的 per-task 聚类可视化、eval 起点 OOD 度量(source 与训练
   source 分布的距离)?
5. **FastWAM 迁移教训**:FastWAM 的 MoT 混合注意力意味着 action 分支**永远**能看到视频 token
   (相当于永远 source+cond)⇒ 「μ 无信息压力」问题必然复现;且 VLSP-current 的输入只是
   首帧 latent(48 维 token,信息量低于 mimic-video 的 2048 维部分去噪未来 latent),μ 可学
   的上限更低。FastWAM 首跑应该直接带哪些防线(KL 默认开?blend 结构?R4/R5 机制)?以及
   是否应该先等 mimic-video 的 D2/R1 结论再动 FastWAM?
6. **行动排序**:给出你认为最优的执行顺序(A'/D2/D3/失败定性 → R1–R3 → R4/R5 的
   触发条件),并指出哪一步的结果会最大地改变后续决策(最高信息价值)。
