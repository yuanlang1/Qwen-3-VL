# 数据协议 v1

本协议服务于多个体育视频问答数据集，单条记录为单视频、单问题。文件采用 UTF-8。

## 标准输入

JSON 数组或 JSONL，每条包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string / integer | 当前文件内唯一，规范化为 string |
| `video` | string | 本地视频路径或映射表中的视频 ID |
| `question` | string | 非空检索问题 |
| `metadata` | object，可选 | 任意追溯信息，不发送给查询模型 |

顶层其他字段不会自动进入核心记录。答案不是必需字段。适配器不读取答案；即使用户在 metadata 中保存答案，查询调用也只接受 question 和预览图。

不要直接混合多个 ID 重复的数据集或划分。将它们分别处理，或在合并前显式构造唯一 ID。

## 视频映射

`--video-index` 接受 JSON 对象，值为本地绝对路径或相对于 `--video-root` 的路径：

```json
{"metadata_video_id":"actual-files/renamed-video.mp4"}
```

路径不作为运动类别输入。映射只负责定位文件。

## 查询产物

`queries.jsonl` 保留标准字段，并增加：

| 字段 | 说明 |
| --- | --- |
| `resolved_video` | 本次运行解析出的绝对视频路径 |
| `sport_type` | 模型根据预览图推断的运动类型或 unknown |
| `sport_type_confidence` | high / medium / low，unknown 必须为 low |
| `sport_type_source` | preview_inference |
| `query_bank` | 恰好七条、非空、互不重复的英文查询，按重要性排序 |
| `preview_frame_indices` | 按时间排序的原视频预览帧索引 |
| `query_generator` | 查询模型名称 |

单独提供外部查询文件时，也必须包含标准字段及合法的 sport_type、sport_type_confidence、query_bank。

## 分数产物

`scores.jsonl` 保留查询记录，并增加：

- `decoder`: 固定为 decord，表示索引和时间戳来源。
- `source_video`: total_frames、average_fps。
- `candidate_frames`: fps、frame_indices、timestamps。
- `query_frame_scores`: `[7, 候选帧数量]` 的有限数值矩阵，分数为余弦相似度。
- `frame_scorer`: model、pretrained、score_type。

时间戳是解码器返回的帧开始时间，单位为秒。候选索引为原视频零基索引，递增且唯一。

## 选帧产物

保留标准字段、运动类型、query_bank、resolved_video、source_video、decoder，并包含：

| 字段 | 说明 |
| --- | --- |
| `selection` | num_queries、frame_budget、alpha、gaussian_radius、gaussian_sigma、fast_ratio |
| `selected_queries` | 排序查询的前 K 条 |
| `candidate_count` | 候选帧数量 |
| `selected_frame_indices` | 按时间排序、唯一的原视频零基索引 |
| `selected_timestamps` | 对应的解码器时间戳 |
| `sampling_types` | 每帧 slow / fast 来源标签 |
| `clip_ids` | 相关片段编号，无关联时为 -1 |
| `relevance_scores` | 对应帧的平滑后相关性分数 |
| `actual_frame_count` | min(frame_budget, candidate_count) |
| `effective_alpha` | 实际选用的阈值参数 |
| `clips` | 相关片段在候选序列中的闭区间位置，不是原视频索引 |

帧、时间戳、来源标签、片段编号和相关分数数组长度相同。Slow 标签可能包含为满足预算而补充的帧，此时 clip_id 可以为 -1。不要将标签解释为视觉模型的输入模态。

消费接口要求完整选帧记录，不自动兼容旧 Sports-QA 工具的 schema。迁移旧产物应显式转换并验证，不能仅改字段名后假定索引来源一致。

## 配置与运行约束

各阶段配置文件包含 config 及其 SHA-256 fingerprint。选帧配置包含分数文件的内容哈希，因此改变分数或参数将产生不同选帧文件。

视频文件缓存使用路径、大小及修改时间识别变更，不是完整内容哈希；如替换文件并刻意保留大小和修改时间，应使用新输出目录。一个输出目录只允许一个写入命令运行。

结构校验可独立执行；完整性校验需要原标准输入；真实视频校验需要相同视频及 Decord 后端。模型预处理与时间位置编码不属于本协议的离线实现。
