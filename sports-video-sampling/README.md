# 体育视频查询生成与关键帧采样

这是独立的离线组件，适配多个体育视频问答数据集。保留体育场景的查询规划逻辑，通过标准输入解除与 Sports-QA 标注格式的绑定。

本目录不导入或修改 `Qwen3-VL`、`sportsqa-offline-sampling`，不启动训练，也不改变现有推理入口。继续执行原命令即使用原路线；需要查询引导选帧时，单独运行本包。生成选帧文件后，现有训练和推理不会自动使用它；接入模型需通过独立入口调用消费接口。

## 安装

建议使用独立环境，在本目录执行：

```powershell
cd "E:\codex\qwen rp\sports-video-sampling"
python -m pip install -e ".[queries,scores]"
```

先安装与运行机器兼容的 PyTorch。Decord 的可用安装方式取决于系统和 Python 环境；本包不会自动切换解码后端。服务端可使用已有可用的 Decord 环境。

可选依赖按用途拆分：

| 安装项 | 用途 |
| --- | --- |
| `pip install -e .` | 标注转换、已有分数的选帧、结构校验 |
| `pip install -e ".[queries]"` | 视频预览和远程查询生成 |
| `pip install -e ".[scores]"` | Decord 解码和 OpenCLIP 打分 |
| `pip install -e ".[video]"` | 消费选帧结果、真实视频校验 |
| `pip install -e ".[test]"` | 单元测试 |

只有查询阶段读取 `DASHSCOPE_API_KEY`。密钥不写入配置文件或产物。

## 统一输入

支持 JSON 数组和 JSONL。每条记录必需 `id`、`video`、`question`：

```json
{"id":"clip_001_question_1","video":"basketball/clip_001.mp4","question":"Was contact made before the shot?","metadata":{"dataset":"custom_sports","split":"challenge"}}
```

- 一个样本对应一个视频和一个问题；同一视频可以有多个问题。
- ID 是非空字符串或整数，读取后统一为字符串，必须在当前输入文件内唯一。
- 视频是本地绝对路径，或相对于 `--video-root` 的路径。未设置根目录时，使用当前工作目录。
- 无扩展名的视频 ID 可探测常用扩展名；出现多个候选时要求显式指定路径。
- 可用 `--video-index examples/video_index.json` 将任意视频 ID 映射到文件路径。
- `metadata` 可选，仅用于追溯和输出传递，不发送给查询模型。
- 数据集名称和划分名称不限，不要求 `train/val/test` 或答案类别文件。
- 每个数据集、每个划分使用独立输入和输出目录。多个文件合并前需使 ID 唯一。

规范详见 [docs/schema.md](docs/schema.md)。`examples/sports_samples.jsonl` 中的视频路径是示例，需要替换为真实文件。

## 标注转换

标准输入可以直接运行。已有 Sports-QA 或体育数据的 Qwen 对话标注可先转换：

```powershell
python -m sports_video_sampling normalize `
  --format sportsqa `
  --input "D:\dataset\meta-data\train.json" `
  --output "D:\dataset\standard\train.jsonl"

python -m sports_video_sampling normalize `
  --format qwen `
  --input "D:\other-sports-dataset\conversations.json" `
  --output "D:\other-sports-dataset\standard.jsonl"
```

Sports-QA 适配器转换 `qa_id/video/question`。Qwen 适配器接受一个视频、一个 human 问题，移除 `<video>`，不读取 assistant 答案；多视频、混合图片或多轮 human 上下文明确拒绝。

其他体育数据集只需转换成标准格式；需要保留专用转换时，在 `adapters/` 增加字段转换函数，并在 `CONVERTERS` 中登记。核心查询、打分和采样无需修改。

## 完整离线运行

```powershell
$env:DASHSCOPE_API_KEY = "your-key"

python -m sports_video_sampling all `
  --input "D:\dataset\standard\train.jsonl" `
  --video-root "D:\dataset\videos" `
  --output-dir "D:\sampling-output\dataset-a-train" `
  --frame-budget 16 `
  --num-queries 7 `
  --check-videos
```

默认查询模型为 `qwen3-vl-flash`，接口地址沿用来源项目；用 `--query-model`、`--query-base-url` 显式覆盖。默认 OpenCLIP 为 `ViT-L-14` / `openai`，使用 CUDA；调试时可指定 `--device cpu`。

`all` 按 queries、scores、select、validate 执行。任一阶段存在失败记录时，以非零状态退出，并停止后续阶段。成功记录保留，原命令可续跑。生成摘要包含选帧文件的实际路径。

小样本测试可以添加 `--limit 10`，并使用单独的输出目录。

## 分阶段运行

```powershell
python -m sports_video_sampling queries `
  --input "D:\dataset\standard\train.jsonl" `
  --video-root "D:\dataset\videos" `
  --output-dir "D:\sampling-output\dataset-a-train"

python -m sports_video_sampling scores `
  --video-root "D:\dataset\videos" `
  --output-dir "D:\sampling-output\dataset-a-train"

python -m sports_video_sampling select `
  --output-dir "D:\sampling-output\dataset-a-train" `
  --frame-budget 8 --num-queries 5
```

`scores` 默认读取输出目录中的 `queries.jsonl`，也接受 `--queries PATH`。`select` 默认读取 `scores.jsonl`，也接受 `--scores PATH`。独立选择阶段不需要视频、API、Decord 或 OpenCLIP。

选择参数：

| 参数 | 默认值 |
| --- | --- |
| `--frame-budget` | 16 |
| `--num-queries` | 7，允许 3～7 |
| `--alpha` | 0.5 |
| `--gaussian-radius` | 4 |
| `--gaussian-sigma` | 1.0 |
| `--fast-ratio` | 0.25 |

只改变以上参数，重新执行 `select` 即可。选帧文件名包含配置指纹，不覆盖其他选帧实验。

```powershell
python -m sports_video_sampling validate `
  --selections "D:\sampling-output\dataset-a-train\selections\q5_f8_FINGERPRINT.jsonl" `
  --input "D:\dataset\standard\train.jsonl" `
  --video-root "D:\dataset\videos" `
  --output-dir "D:\sampling-output\dataset-a-train" `
  --check-videos
```

将 `FINGERPRINT` 替换为 `select` 摘要中的实际文件名。`--input` 启用完整性与问题匹配检查；不提供时只校验产物结构。`--check-videos` 解码指定帧，检查帧数和解码器时间戳。不提供 `--selections` 时，需要分数文件及与选择阶段相同的采样参数来定位产物。

## 体育查询与采样行为

查询提示词定义了运动识别、帧检索查询和视觉名词锚点。

- 默认四张按时间排序的预览，最长边不超过 224 像素。
- 运动类型仅由预览图的可见证据推断，不使用路径、数据集标签或问题中的运动名称来判断类型。
- 保留 `sport_type` 与 `sport_type_confidence`；不明确时输出 unknown / low。
- 固定七条互补且排序的英文查询，每条不超过 20 个英文词。
- `focus_nouns` 包含 2～8 条按重要性排序的英文名词短语，每条最多 4 个英文词，随选帧产物保存供后续视觉 token 压缩读取。
- 保留对犯规、接触时机和位置、前后过程、裁判反应、边界和争议事件的证据要求。
- 问题仅作为检索上下文，不生成答案，不预判违规，不把答案选项抄入查询。

候选帧默认在不超过 30 秒、30～120 秒、超过 120 秒的视频中分别使用 4/2/1 FPS，并保留末帧。CLIP 分数取前 K 条查询等权平均，再高斯平滑和检测相关片段。默认 16 帧分配为 12 Slow、4 Fast；标签只记录采样来源。

相关帧池不足时调整阈值并补充采样；补充后重新排序。输出是唯一、有序的零基原始帧索引。候选不足时输出实际可用数量，不重复帧补预算。时间戳读取 Decord，不使用 `index / average_fps` 替代。

## 产物与续跑

```text
OUTPUT_DIR/
  queries.jsonl
  queries.config.json
  scores.jsonl
  scores.config.json
  cache/features/<CLIP配置指纹>/*.npz
  selections/q7_f16_<选择配置指纹>.jsonl
  selections/q7_f16_<选择配置指纹>.config.json
  reports/validation_q7_f16_<选择配置指纹>.json
  failures/queries.jsonl
  failures/scores.jsonl
  failures/select.jsonl
```

同一视频的多个问题共用预览与候选帧图像编码，但分别打分和选择。特征缓存以视频路径、文件大小、修改时间、候选帧与 CLIP 配置校验。输入、提示词、上游产物和阶段参数纳入配置指纹。

改变上游数据、视频或模型设置时，推荐使用新输出目录；配置不符会在追加前失败。`--overwrite` 显式替换当前阶段产物；对 `all` 使用该参数会重跑查询生成。不要在同一目录并发运行多个写入命令。最后一条被中断的阶段输出可恢复，文件中间的损坏会报错；下游拒绝读取不完整的上游文件。

失败日志是历次尝试的追加记录，保留历史失败；当前运行结果以命令摘要及验证报告为准。

## 按需拼接到其他入口

```python
from sports_video_sampling import (
    load_selections, get_selection, decode_selected_frames,
)

records = load_selections("/data/run/selections/q7_f16_FINGERPRINT.jsonl")
selection = get_selection(records, "clip_001_question_1")
decoded = decode_selected_frames("/data/videos/basketball/clip_001.mp4", selection)

frames = decoded["frames"]             # RGB uint8，形状 [T,H,W,C]
indices = decoded["frame_indices"]     # 原视频索引
timestamps = decoded["timestamps"]    # 原视频时间戳，单位秒
```

消费接口使用同一 Decord 后端，核对原视频帧数及保存时间戳，不进行第二次采样。它不负责模型预处理、时间位置编码、推理或训练。模型旁路需要使用这些帧和时间信息，并禁用处理器二次抽帧；尤其不能将非均匀帧简单解释为固定 FPS。原有 Qwen 主路线不会自动加载本包。

## 文件职责与验证

```text
src/sports_video_sampling/
  cli.py             命令参数与阶段启动
  records.py         标准记录、输入校验、视频解析
  adapters/          Sports-QA 与 Qwen 对话标注转换
  query_bank.py      体育提示词、API 请求、响应校验
  video_io.py        Decord 信息、预览和指定索引解码
  scorer.py          OpenCLIP、批处理、特征缓存
  selection.py       无模型依赖的 Slow-Fast 算法
  artifacts.py       JSONL、配置校验、续跑、失败记录
  pipeline.py        数据集无关的阶段编排
  consumer.py        选帧校验及读取接口
```

```powershell
python -B -m pytest -q -p no:cacheprovider
```

测试覆盖两种标注格式、体育查询约束、按 QA 选帧、图像特征复用、参数传递、配置隔离、失败重试、续跑和精确索引读取。API 与 OpenCLIP 使用替身验证编排；可选真实视频测试需要 Decord 及 FFmpeg，没有这些依赖时明确跳过。测试不会调用付费 API 或下载 CLIP 权重。


高斯平滑使用连续高斯系数：

```text
s_smoothed[i] = sum(s[i+j] * exp(-j²/(2σ²)), j=-r..r) / (sqrt(2π) * σ)
```

有限窗口权重不再归一化为和为 1；边界使用首尾值重复填充（edge）。
`r=0` 时仍乘以 `1/(sqrt(2π)*σ)`，单元素序列同样执行卷积。
新选帧配置及记录增加 `gaussian_normalization="continuous"`，产生独立的选帧配置指纹。
已有查询、图像特征与 CLIP 分数可复用，只需重新运行 `select` 和 `validate`。
相关性分数尺度随核权重和改变；均值加标准差阈值同比缩放，理论上选择保持一致，浮点边界比较可能存在差异。
