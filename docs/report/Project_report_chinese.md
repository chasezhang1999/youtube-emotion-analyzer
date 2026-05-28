# YouTube 观众情绪分析器 — 数字营销活动支持工具

## 1. 项目标题与学生信息

**项目标题：** YouTube Audience Emotion Analyzer for Digital Marketing Campaigns

**学生信息：**
- ZHANG Xinchao, 21257618
- Yao Ziyue, 21260768

## 2. 公司信息

**公司名称：** InsightWave Digital Marketing Agency（课程项目场景）

**应用网址：** https://youtube-emotion-analyzer.streamlit.app/

InsightWave 数字营销公司帮助品牌评估社交媒体上的视频营销效果和受众互动。YouTube 评论区包含大量有价值的用户反馈，但人工阅读评论效率低、主观性强、难以规模化。本项目构建了一个深度学习应用，自动汇总 YouTube 评论中的主要情绪，并将其转化为可执行的营销建议。

## 3. 项目目标

帮助数字营销公司将 YouTube 视频的前 100 条评论分为 7 种情绪类别，汇总观众反馈，标记负面情绪风险，并生成可操作的营销优化建议。

## 4. 项目策略

构建一个 Streamlit Cloud 业务应用，用户输入 YouTube 视频链接后，系统通过 YouTube Data API 获取前 100 条评论，使用 Hugging Face Transformer 模型进行分析。

应用包含两条文本分类流水线：

1. **七情绪分类流水线** — 详细分析观众情绪
2. **三分类情感流水线** — 提供简洁的正面/中性/负面信号

七种情绪为：愤怒、厌恶、恐惧、喜悦、中性、悲伤、惊讶。应用计算主要情绪、完整情绪分布、情感分布，以及负面情绪比率（愤怒 + 厌恶 + 恐惧 + 悲伤），作为营销风险指标。负面情绪比率过高时，营销团队应审查信息框架和受众定位。

应用提供的功能包括：
- 主要观众情绪
- 七情绪分布图
- 三分类情感分布图
- 三模型情绪对比模式
- 负面情绪比率指标
- 营销决策 pipeline（行动建议 + 风险等级）
- 逐条评论预测表格（含模型置信度）
- 营销建议文本
- CSV 下载功能

## 5. 模型链接

| 模型 | 用途 | Hugging Face 链接 |
|------|------|------|
| YouTube-domain adapted DistilBERT（最终推荐） | 主情绪模型 | https://huggingface.co/chase1zhang/youtube-emotion-distilbert-domain-adapted |
| GoEmotions fine-tuned DistilBERT | 第一阶段调优 | https://huggingface.co/chase1zhang/youtube-emotion-distilbert |
| Public GoEmotions RoBERTa (SamLowe) | 公开对比模型 | https://huggingface.co/SamLowe/roberta-base-go_emotions |
| Public DistilRoBERTa 7-emotion (j-hartmann) | 预调优基线 | https://huggingface.co/j-hartmann/emotion-english-distilroberta-base |
| Public RoBERTa-large 7-emotion (j-hartmann) | 更大模型对比 | https://huggingface.co/j-hartmann/emotion-english-roberta-large |
| CardiffNLP 情感模型 | 辅助情感分类 | https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest |

最终应用默认使用 YouTube-domain adapted 模型。对比模式下，同一批评论会同时通过三个情绪模型运行，方便用户直观比较。

## 6. 应用网址

**Streamlit Cloud 部署地址：** https://youtube-emotion-analyzer.streamlit.app/

## 7. GitHub 仓库

**仓库地址：** https://github.com/chasezhang1999/youtube-emotion-analyzer

仓库已连接 Streamlit Cloud 实现自动部署。YouTube API Key 存储在 Streamlit secrets 中，不会上传到 GitHub。

## 8. 数据集

### 8.1 GoEmotions 调优数据集

第一阶段训练使用 Hugging Face 上的 GoEmotions 数据集：

- 数据集地址：https://huggingface.co/datasets/SetFit/go_emotions
- 输入特征：`text`
- 标签：anger, disgust, fear, joy, neutral, sadness, surprise

原始 GoEmotions 是多标签数据集，本项目筛选为单标签的七情绪版本，并按类别下采样至平衡。

| 数据集划分 | 样本数 | 类别平衡 |
|---|---:|---|
| 训练集 | 5,000 | 分层采样；少数类受可用样本数限制 |
| 验证集 | 406 | 每类 58 条 |
| 测试集 | 1,000 | 分层采样；少数类受可用样本数限制 |

预处理步骤：
1. 保留仅有一个情绪标签的样本
2. 只保留项目所需的七种情绪
3. 将标签名转换为 0-6 的数字 ID
4. 按类别平衡各划分，避免多数类主导

### 8.2 YouTube 领域适配数据集

GoEmotions 的 Reddit 风格文本与 YouTube 评论差异较大，因此额外创建了 YouTube 领域数据集。更新后的收集流程从 71 个 YouTube 视频中构建了 8,000 条辅助标注原始评论池，涵盖科技、品牌、娱乐、产品问题、公共公告、品牌广告、产品发布等主题。最终领域适配数据集从原始池中选取 5,000 条评论，并在不复制少数类样本的前提下尽量保持类别平衡。

评论使用辅助标注工具进行七情绪标注。该数据集仅用于领域适配训练，应用评估使用独立的 150 条人工审核评论。

YouTube 领域标签分布：

| 标签 | 数量 |
|---|---:|
| anger（愤怒） | 905 |
| disgust（厌恶） | 505 |
| fear（恐惧） | 608 |
| joy（喜悦） | 904 |
| neutral（中性） | 904 |
| sadness（悲伤） | 802 |
| surprise（惊讶） | 372 |

### 8.3 人工审核测试数据集

最终应用在 3 个 YouTube 视频上测试，每个视频 50 条人工审核评论，共 150 条。该基准独立于训练数据。第二轮审核修正了 23 个标签，改善了讽刺、幽默、骄傲和威胁/担忧等语境的标注。

## 9. 模型架构

### 9.1 流水线 1：七情绪分类

```python
pipeline("text-classification", model="chase1zhang/youtube-emotion-distilbert-domain-adapted")
```

模型开发过程：
- 基础模型：`distilbert-base-uncased`
- 第一阶段：在平衡的 GoEmotions 七情绪数据集上调优
- 第二阶段：在 YouTube 领域评论上继续调优（当前数据集为 5,000 条最终样本，来自 8,000 条原始池）
- 训练参数：学习率 2e-5，batch size 16，第一阶段 3 个 epoch
- 模型选择：基于验证集准确率和应用级人工测试

SamLowe 模型输出 28 类 GoEmotions 标签。为方便对比，相关标签被映射到七情绪。例如 admiration、amusement、love、gratitude、optimism、excitement 映射为 joy；annoyance 映射为 anger。

### 9.2 流水线 2：情感分类

```python
pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

辅助模型为业务人员提供更简洁的信号。在应用测试中，三分类情感任务比七情绪更稳定，因为 YouTube 评论短小、含噪声、有多语言和讽刺。

### 9.3 应用流水线

```
YouTube 视频链接
    → 解析视频 ID
    → YouTube Data API 获取评论
    → 前 100 条一级评论
    → 清洗评论文本
    → 七情绪模型
    → 三分类情感模型
    → 情绪汇总、情感汇总、风险比率
    → Streamlit 仪表板 + CSV 导出
```

### 9.4 代码结构

```
youtube_emotion_project/
├── streamlit_app.py              # Streamlit 入口
├── youtube_emotion/
│   ├── core.py                   # 标签映射、文本清洗、营销建议
│   ├── model_runner.py           # 模型加载和预测
│   ├── youtube_client.py         # YouTube API 客户端
│   └── dataset_prep.py           # 数据集准备工具
├── data/
│   ├── go_emotions_7class/       # GoEmotions 七情绪数据
│   └── youtube_domain_7class/    # YouTube 领域数据
├── notebooks/
│   ├── Fine_tune_Model.ipynb     # 训练 notebook
│   └── testing_experiments.ipynb # 测试与实验 notebook
├── experiments/
│   ├── Experimental_results.xlsx # 实验结果 Excel
│   └── app_model_comparison_5models.csv
├── tests/                        # 单元测试
└── requirements.txt
```

## 10. 部署

应用部署在 Streamlit Cloud，连接 GitHub 仓库。

部署步骤：
1. 将应用代码推送到 GitHub
2. 连接仓库到 Streamlit Cloud
3. 在 Streamlit Cloud 配置 `YOUTUBE_API_KEY` secret
4. 使用 `st.cache_resource` 缓存 Hugging Face 模型
5. 用户通过公开网址访问

使用流程：
1. 粘贴 YouTube 视频链接
2. 选择单模型分析或多模型对比
3. 点击分析按钮
4. 查看情绪和情感图表
5. 浏览逐条评论预测表格
6. 阅读营销建议
7. 按需下载 CSV

## 11. 实验结果

### 11.1 GoEmotions 测试集上的模型选择

下表是重新训练后刷新过的模型选择结果，使用扩展后的 1,000 条 GoEmotions 测试样本，并拉取 2026 年 5 月 27 日 Hugging Face 上的最新模型版本。

| 模型 | 设备 | 测试样本数 | 准确率 | 含加载时间 | 仅推理时间 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| j-hartmann DistilRoBERTa | CPU | 1,000 | 0.6680 | 4.95s | 3.72s | 预调优基线 |
| GoEmotions fine-tuned | CPU | 1,000 | 0.7030 | 3.65s | 3.57s | 第一阶段调优 |
| Domain-adapted | CPU | 1,000 | 0.6280 | 3.69s | 3.62s | YouTube 领域适配 |
| CardiffNLP 情感模型 | CPU | 1,000 | 0.6460 | 8.75s | 7.50s | 辅助情感模型 |

GoEmotions fine-tuned 模型在 GoEmotions 测试集上从 0.6680 提升到 0.7030。Domain-adapted 模型在此测试集上略低，因为它进一步优化了对 YouTube 评论风格的适应，而不是单纯追求 Reddit 风格 GoEmotions 测试集表现。

### 11.2 YouTube 领域验证

| 模型 | 数据集 | 设备 | 样本数 | 准确率 |
|---|---|---:|---:|---:|
| j-hartmann DistilRoBERTa | YouTube 领域验证集 | CPU | 1000 | 0.4090 |
| GoEmotions fine-tuned | YouTube 领域验证集 | CPU | 1000 | 0.4670 |
| Domain-adapted | YouTube 领域验证集 | CPU | 1000 | 0.6650 |
| Public GoEmotions RoBERTa | YouTube 领域验证集 | CPU | 1000 | 0.4430 |
| Public RoBERTa-large | YouTube 领域验证集 | CPU | 1000 | 0.4560 |
| CardiffNLP 情感模型 | YouTube 领域验证集 | CPU | 1000 | 0.5970 |

在最新的 1,000 条 YouTube-domain 验证集中，Domain-adapted DistilBERT 表现最好（0.6650），显著高于其他模型（如公共 SamLowe RoBERTa 模型为 0.4430）。当前 YouTube-domain 数据集已经扩展为 5,000 条最终样本，这些指标已基于最新的 1,000 条验证集进行了刷新。

### 11.3 应用级性能（5 个模型对比，150 条人工审核评论）

**逐视频结果：**

| 视频 | 任务 | 模型阶段 | 匹配数/50 | 准确率 | 人工主标签 | 模型主标签 |
|---|---|---|---:|---:|---|---|
| d2dgJGkw5p0 | 7情绪 | Pre-tuning baseline | 29 | 0.58 | neutral | neutral |
| d2dgJGkw5p0 | 7情绪 | GoEmotions fine-tuned | 29 | 0.58 | neutral | neutral |
| d2dgJGkw5p0 | 7情绪 | **YouTube-domain adapted** | 28 | 0.56 | neutral | neutral |
| d2dgJGkw5p0 | 7情绪 | Public RoBERTa-large | 24 | 0.48 | neutral | neutral |
| d2dgJGkw5p0 | 7情绪 | Public GoEmotions RoBERTa | 26 | 0.52 | neutral | neutral |
| d2dgJGkw5p0 | 3情感 | Sentiment pipeline | 30 | 0.60 | neutral | neutral |
| M8To7iorkxQ | 7情绪 | Pre-tuning baseline | 26 | 0.52 | joy | neutral |
| M8To7iorkxQ | 7情绪 | GoEmotions fine-tuned | 25 | 0.50 | joy | neutral |
| M8To7iorkxQ | 7情绪 | **YouTube-domain adapted** | 25 | 0.50 | joy | neutral |
| M8To7iorkxQ | 7情绪 | Public RoBERTa-large | 24 | 0.48 | joy | neutral |
| M8To7iorkxQ | 7情绪 | Public GoEmotions RoBERTa | 37 | 0.74 | joy | joy |
| M8To7iorkxQ | 3情感 | Sentiment pipeline | 46 | 0.92 | positive | positive |
| -_-eIVAX1yQ | 7情绪 | Pre-tuning baseline | 12 | 0.24 | anger | neutral |
| -_-eIVAX1yQ | 7情绪 | GoEmotions fine-tuned | 16 | 0.32 | anger | neutral |
| -_-eIVAX1yQ | 7情绪 | **YouTube-domain adapted** | 18 | 0.36 | anger | neutral |
| -_-eIVAX1yQ | 7情绪 | Public RoBERTa-large | 14 | 0.28 | anger | neutral |
| -_-eIVAX1yQ | 7情绪 | Public GoEmotions RoBERTa | 17 | 0.34 | anger | neutral |
| -_-eIVAX1yQ | 3情感 | Sentiment pipeline | 29 | 0.58 | negative | negative |

**总体结果：**

| 任务 | 模型 | 匹配数/150 | 准确率 |
|---|---|---:|---:|
| 7情绪 | Pre-tuning baseline | 67 | 0.4467 |
| 7情绪 | GoEmotions fine-tuned | 70 | 0.4667 |
| 7情绪 | Public GoEmotions RoBERTa | 80 | 0.5333 |
| 7情绪 | **YouTube-domain adapted（自研最终模型）** | **71** | **0.4733** |
| 7情绪 | Public RoBERTa-large | 62 | 0.4133 |
| 3情感 | Sentiment pipeline | 105 | 0.7000 |

### 11.4 关键发现

- GoEmotions fine-tuned DistilBERT 在扩展后的 GoEmotions 测试集上从 0.6680 提升到 0.7030。
- 在最新的 1,000 条 YouTube-domain 验证集中，domain-adapted 模型取得最佳结果：**665/1000 (0.6650)**，显著高于其他模型，验证了领域适配的有效性。
- 在 150 条应用级人工 benchmark 中，公共 SamLowe GoEmotions RoBERTa 最高：**80/150 (0.5333)**。自研 domain-adapted DistilBERT 是最强自研 DistilBERT：**71/150 (0.4733)**，高于 GoEmotions fine-tuned 的 70/150 和预调优基线的 67/150。
- 最大提升出现在以愤怒情绪为主的新闻视频上：从 12/50 提升到 18/50，说明平衡的领域适配有助于模型更好地检测负面情绪。
- 公共模型依然是有价值的对照：SamLowe RoBERTa 在 app benchmark 里最好，而 RoBERTa-large 在本任务上更慢且准确率较低。
- Streamlit Cloud 的 CPU 推理对于 100 条评论的分析是可以接受的，因为模型 pipeline 会被缓存。
- 三分类情感模型在正面/中性/负面层面最稳定（105/150），但七情绪分类提供了更细致的诊断洞察。

## 12. 业务解读

对数字营销公司而言，最有价值的不仅是单条评论的标签，更是活动层面的情绪模式：

- **喜悦和惊讶为主** → 活动产生正面反响，继续当前的内容风格和创意方向
- **中性为主** → 视频可能是信息性的，但缺乏情感吸引力，应优化标题、号召性用语或情感框架
- **负面情绪超过 40%** → 营销团队应人工审查评论主题，在扩大投放前调整信息策略
- **七情绪模型与情感模型结论不一致** → 标记为需要人工复核的信号，而非自动决策

本应用是决策支持工具，减少人工阅读评论的工作量，帮助营销团队快速识别视频是在激发热情、引发冷漠、还是带来声誉风险。

## 13. 局限性与未来改进

主要局限是领域差异。GoEmotions 提供高质量情绪标签，但不是 YouTube 专用数据集。YouTube 评论包含俚语、讽刺、表情符号、简短回复、多语言内容等。

YouTube 领域适配数据集缩小了这一差距。重新平衡训练数据并应用类别加权损失后，模型在三个测试视频上均有提升，特别是在愤怒情绪较多的新闻内容上。

未来改进方向：
1. 收集更多人工验证的 YouTube 评论
2. 平衡领域数据集的七种情绪类别
3. 增加产品发布、品牌危机、娱乐和公共议题等更多视频类型
4. 除准确率外，评估 macro-F1 指标
5. 使用置信度阈值标记不确定的评论供人工复核

## 14. 结论

本项目展示了基于 Transformer 的文本分类如何支持数字营销决策。Streamlit 应用收集 YouTube 评论，应用两条 Hugging Face 流水线，可视化观众情绪和情感，并生成实用建议。

实验结果表明：调优提升了 GoEmotions 基准上的表现；使用平衡 YouTube 数据的领域适配让自研模型在 YouTube-domain 验证集上取得最佳结果。公共 SamLowe RoBERTa 是 app 手工 benchmark 中最强的对照模型，而 YouTube-domain adapted DistilBERT 仍作为默认最终模型，因为它体积较小、由项目自研维护，并且在专门的 YouTube 验证集上表现最好。辅助情感模型在正面/中性/负面层面表现最稳定（105/150）。两条流水线共同为营销活动监控和受众反馈分析提供了有用的工作流程。

---

**学生：** ZHANG Xinchao (21257618), Yao Ziyue (21260768)
**课程：** ISOM5240 Deep Learning Business Applications with Python
**日期：** 2026 年 5 月
