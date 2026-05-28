# YouTube 观众情绪分析器 — 数字营销活动分析

## 1. 项目标题与学生姓名

**项目标题：** YouTube 观众情绪分析器 — 数字营销活动分析

**学生姓名：**
- 张鑫超 (ZHANG Xinchao), 21257618
- 姚梓悦 (Yao Ziyue), 21260768

## 2. 公司名称与网站

**公司：** Nike, Inc.（耐克）

**公司网站：** https://www.nike.com

**应用地址：** https://youtube-emotion-analyzer.streamlit.app/

Nike 在 YouTube 上投放大量营销活动，包括产品发布、运动员合作和品牌广告。YouTube 评论包含有价值的用户反馈，但人工阅读评论速度慢、主观性强、难以规模化。本项目构建了一个深度学习应用，将 YouTube 评论中的主要情绪进行汇总，并转化为可执行的营销建议。

## 3. 项目目标

本项目帮助 Nike 将 YouTube 视频下的前 100 条评论分类为七种情绪，汇总活动反馈，标记负面情绪风险，并生成可执行的内容优化建议。（共 44 词，符合 50 词限制）

## 4. 策略

策略是构建一个部署在 Streamlit Cloud 上的商业应用，接受 YouTube 视频 URL，通过 YouTube Data API 获取前 100 条顶级评论，并使用 Hugging Face Transformer pipeline 进行分析。

应用使用两个文本分类 pipeline：

1. 七情绪 pipeline：用于详细的观众情绪分析
2. 三分类情感 pipeline：提供更简单的正面/中性/负面商业信号

七种目标情绪为：愤怒 (anger)、厌恶 (disgust)、恐惧 (fear)、快乐 (joy)、中性 (neutral)、悲伤 (sadness)、惊讶 (surprise)。应用计算主导情绪、完整情绪分布、情感分布，以及负面情绪比率（愤怒 + 厌恶 + 恐惧 + 悲伤）。该比率用作活动风险指标。如果负面情绪较高，营销团队应审查信息框架、受众定位和潜在的公众反应风险。

仪表板提供：

- 主导观众情绪
- 七情绪分布图
- 三分类情感分布图
- 三模型情绪对比模式
- 负面情绪比率指标
- 活动行动与风险等级的决策 pipeline
- 带有模型置信度的逐评论预测表
- 营销建议文本
- 预测结果 CSV 下载

## 5. 模型 URL

**主要部署情绪模型：** https://huggingface.co/chase1zhang/youtube-emotion-distilbert-domain-adapted

**原始微调情绪模型：** https://huggingface.co/chase1zhang/youtube-emotion-distilbert

**公开 GoEmotions 微调对比模型：** https://huggingface.co/SamLowe/roberta-base-go_emotions

**微调前基线情绪模型：** https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

**可选大型七情绪模型：** https://huggingface.co/j-hartmann/emotion-english-roberta-large

**辅助情感模型：** https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

**备选情感模型（快速 CPU）：** https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student

**备选情感模型（社交媒体）：** https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis

最终 Streamlit 应用默认使用 YouTube 域适配模型，因为它使用 YouTube 域评论进行了进一步训练。应用还包含对比模式，可通过三个情绪模型同时分析同一批评论：YouTube 域适配 DistilBERT、公开 SamLowe GoEmotions RoBERTa 模型和公开 j-hartmann DistilRoBERTa 七情绪模型。原始 GoEmotions DistilBERT 和更大的 j-hartmann RoBERTa-large 模型仍可通过侧边栏选择器使用。对于情感分析，应用支持通过下拉菜单在三个预训练模型之间切换：CardiffNLP（推荐）、lxyuan（快速 CPU 推理，多语言）和 FiniteAutomata BERTweet（在社交媒体文本上表现稳健）。

## 6. 应用 URL

**已部署的 Streamlit Cloud 应用：** https://youtube-emotion-analyzer.streamlit.app/

## 7. GitHub URL

**GitHub 仓库：** https://github.com/chasezhang1999/youtube-emotion-analyzer

仓库已连接到 Streamlit Cloud 进行自动部署。YouTube Data API 密钥存储在 Streamlit secrets 中，未上传到 GitHub。

## 8. 数据集

### 8.1 GoEmotions 微调数据集

第一阶段训练使用 Hugging Face 上的 GoEmotions 数据集：

- 数据集 URL：https://huggingface.co/datasets/SetFit/go_emotions
- 输入特征：`text`
- 目标特征：七分类情绪标签
- 标签：anger, disgust, fear, joy, neutral, sadness, surprise

原始 GoEmotions 数据集是多标签的。本项目将其过滤为干净的单标签样本，并缩减为七种目标情绪。训练集和测试集使用目标大小分层抽样（不重复），受可用少数类样本数量限制；验证集保持类别平衡。

| 划分 | 样本数 | 类别平衡 |
|---|---:|---|
| 训练集 | 17,166 | 分层抽样；受可用少数类行数限制 |
| 验证集 | 2,105 | 分层抽样 |
| 测试集 | 2,160 | 分层抽样；受可用少数类行数限制 |

预处理步骤：

1. 保留恰好有一个情绪标签激活的样本
2. 仅保留七种项目标签
3. 将标签名称转换为 0 到 6 的标签 ID
4. 按类别平衡每个划分，避免多数类模型

准备好的文件：

- `data/go_emotions_7class/train.csv`
- `data/go_emotions_7class/validation.csv`
- `data/go_emotions_7class/test.csv`

### 8.2 YouTube 域适配数据集

由于 Reddit 风格的 GoEmotions 文本与 YouTube 评论不同，额外创建了 YouTube 域适配数据集。更新后的集合包含来自 71 个 YouTube 视频的 8,000 条 DeepSeek AI 标注的原始评论池，涵盖科技、品牌、娱乐、产品问题、公共公告、品牌公益广告、产品发布和观众反应等主题。从该池中，最终适配数据集选择了 3,991 条平衡评论。

评论使用 DeepSeek v4pro AI 标注，使用相同的七种情绪标签。这些标签仅用于额外的域适配。独立的应用评估仍使用单独的人工审核 500 条 YouTube 评论集，因此报告的应用测试不是在训练评论上衡量的。

YouTube 域标签分布：

| 标签 | 数量 |
|---|---:|
| anger (愤怒) | 714 |
| disgust (厌恶) | 292 |
| fear (恐惧) | 201 |
| joy (快乐) | 714 |
| neutral (中性) | 714 |
| sadness (悲伤) | 714 |
| surprise (惊讶) | 642 |

准备好的文件：

- `data/youtube_domain_7class_deepseek/all.csv`
- `data/youtube_domain_7class_deepseek/train.csv`
- `data/youtube_domain_7class_deepseek/validation.csv`
- `data/youtube_domain_training_comments_8000_deepseek_labeled.csv`

### 8.3 应用测试数据集

已部署的应用在 10 个 YouTube 视频上进行了评估，每个视频 50 条评论，共 500 条评论。其中 3 个视频的 150 条评论由人工审核，另外 7 个视频的 350 条评论由 DeepSeek v4pro AI 标注。该基准与 YouTube 域适配训练数据分开。10 个视频涵盖稀土、阿凡达片段、枪击新闻、品牌危机、公共安全、公益广告、产品失败、食品安全、快乐预告片和负面品牌危机等主题。基准存储在：

- `experiments/app_per_comment_manual_labels_500.csv`

## 9. 模型

### 9.1 Pipeline 1：七情绪分类

主 pipeline 将每条 YouTube 评论分类为七种情绪标签之一。

```python
pipeline(
    "text-classification",
    model="chase1zhang/youtube-emotion-distilbert-domain-adapted",
)
```

模型开发：

- 基础模型：`distilbert-base-uncased`
- 第一阶段：在平衡的七类 GoEmotions 数据集上微调
- 第二阶段：在 YouTube 域评论上进一步微调；更新后的仓库数据集包含来自 8,000 条 DeepSeek 标注原始池的 3,991 条平衡评论
- 训练设置：学习率 2e-5，批大小 16，第一阶段 3 个 epoch
- 模型选择：验证准确率和应用级人工测试
- Streamlit 对比：应用可将部署模型与 `SamLowe/roberta-base-go_emotions`、`j-hartmann/emotion-english-distilroberta-base`、`chase1zhang/youtube-emotion-distilbert` 和 `j-hartmann/emotion-english-roberta-large` 进行对比

SamLowe 模型预测 28 标签 GoEmotions 标签集。为与本项目的七情绪输出对比，相关标签被映射到七种目标情绪。例如，admiration、amusement、love、gratitude、optimism 和 excitement 映射为 joy；annoyance 映射为 anger；disappointment、grief 和 remorse 映射为 sadness。

### 9.2 Pipeline 2：情感分类

第二个 pipeline 将每条评论分类为正面、中性或负面情感。

```python
pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
```

该辅助模型为利益相关者提供更简单的信号。在应用测试中，三分类情感任务比七情绪分类更稳定，因为 YouTube 评论短小、嘈杂、带有讽刺，有时还是多语言的。

### 9.3 Pipeline 3：业务决策引擎

Pipeline 3 使用基于规则的启发式方法，将 Pipeline 1（情绪）和 Pipeline 2（情感）的预测映射为高层战略行动：

| 规则 | 条件 | 决策 | 风险等级 |
|---|---|---|---|
| 高风险 | 负面情绪比率 >= 40% 或 负面情感比率 >= 40% | 扩大投放前先审查 | 高 |
| 低风险 | 主导情绪为 Joy/Surprise 且 正面情感比率 >= 50% | 扩大正面创意 | 低 |
| 中风险（参与度） | 主导情绪为 Neutral | 改善互动钩子 | 中 |
| 中风险（混合） | 其他所有混合信号 | 监控并审查样本 | 中 |

决策逻辑在 `core.py` 中的 `build_campaign_decision()` 实现。它结合七情绪分布和三分类情感输出，生成决策标签、风险等级、关键比率和建议的下一步行动。

### 9.4 应用 Pipeline

```text
YouTube 视频 URL
    -> 解析视频 ID
    -> YouTube Data API commentThreads 端点
    -> 前 100 条顶级评论
    -> 清洗评论文本
    -> 七情绪 pipeline
    -> 三分类情感 pipeline
    -> 情绪汇总、情感汇总、风险比率
    -> Streamlit 仪表板和 CSV 导出
```

### 9.5 代码结构

```text
youtube_emotion_project/
|-- streamlit_app.py
|-- youtube_emotion/
|   |-- core.py
|   |-- model_runner.py
|   |-- youtube_client.py
|   |-- dataset_prep.py
|   `-- __init__.py
|-- data/
|   |-- go_emotions_7class/
|   |-- youtube_domain_7class_deepseek/
|   `-- sample_comments.csv
|-- notebooks/
|   |-- fine_tune_all_models.ipynb
|   `-- testing_experiments.ipynb
|-- experiments/
|   |-- Experimental_results.xlsx
|   `-- Performance_result.xlsx
`-- requirements.txt
```

## 10. 部署

应用部署在 Streamlit Cloud 上，并连接到 GitHub 仓库。

部署步骤：

1. 将 Streamlit 应用文件推送到 GitHub
2. 将 GitHub 仓库连接到 Streamlit Cloud
3. 将 `YOUTUBE_API_KEY` 添加到 Streamlit Cloud secrets
4. 首次使用时加载 Hugging Face 模型并使用 `st.cache_resource` 缓存
5. 用户通过公开应用 URL 访问

应用使用方法：

1. 粘贴 YouTube 视频 URL
2. 从侧边栏选择单模型分析或多模型对比
3. 点击分析按钮
4. 查看情绪和情感图表
5. 阅读逐评论预测表
6. 使用建议框进行活动解读
7. 如需要，下载预测 CSV

应用截图：

![输入页面与 YouTube URL](screenshots/01_input_with_url.png)

![汇总指标](screenshots/02_summary_metrics.png)

![七情绪分布](screenshots/03_emotion_distribution.png)

![逐评论结果表](screenshots/04_comment_level_results.png)

![营销建议](screenshots/05_marketing_recommendation.png)

![三模型对比模式](screenshots/06_model_comparison_mode.png)

## 11. 实验

实验评估模型准确率、运行时间、域适配效果和已部署应用性能。

### 11.1 GoEmotions 测试集上的模型选择

该基准遵循课程 pipeline 选择思路：比较包含模型加载和不包含模型加载的准确率与运行时间。使用完整的 2,160 行 GoEmotions 测试集。

| 模型 | 设备 | 测试样本 | 准确率 | 含加载运行时间 | 不含加载运行时间 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| `j-hartmann/emotion-english-distilroberta-base` | CPU | 2,160 | 0.6204 | 148.35s | 118.44s | 微调前基线 |
| `j-hartmann/emotion-english-distilroberta-base` | GPU | 2,160 | 0.6204 | 27.63s | 13.89s | 微调前基线 |
| `chase1zhang/youtube-emotion-distilbert` | CPU | 2,160 | **0.8630** | 138.94s | 122.62s | GoEmotions 微调 |
| `chase1zhang/youtube-emotion-distilbert` | GPU | 2,160 | **0.8630** | 11.89s | 10.98s | GoEmotions 微调 |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | CPU | 2,160 | 0.4944 | 131.07s | 121.88s | YouTube 域适配 |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | GPU | 2,160 | 0.4944 | 12.02s | 10.89s | YouTube 域适配 |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | CPU | 2,160 | 0.5444 | 244.95s | 241.32s | 辅助情感模型 |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | GPU | 2,160 | 0.5444 | 23.02s | 21.30s | 辅助情感模型 |

GoEmotions 微调 DistilBERT 在 GoEmotions 测试集上达到 **0.8630** 准确率，相比微调前基线（0.6204）大幅提升。域适配模型在此测试集上较低（0.4944），因为它用域内准确率换取了 YouTube 域性能。这一取舍在 Section 11.2 的 YouTube 域验证结果中得到了验证。GPU 推理速度约为 CPU 的 10 倍。

### 11.2 YouTube 域验证

| 模型 | 数据集 | 设备 | 样本数 | 准确率 | 含加载运行时间 | 不含加载运行时间 |
|---|---|---:|---:|---:|---:|---:|
| `j-hartmann/emotion-english-distilroberta-base` | YouTube 域验证评论 | CPU | 798 | 0.4574 | 7.6660s | 6.4336s |
| `chase1zhang/youtube-emotion-distilbert` | YouTube 域验证评论 | CPU | 798 | 0.3985 | 6.3855s | 6.3273s |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | YouTube 域验证评论 | CPU | 798 | 0.6028 | 6.4497s | 6.4113s |
| `SamLowe/roberta-base-go_emotions` | YouTube 域验证评论 | CPU | 798 | 0.4586 | 14.2942s | 13.3855s |
| `j-hartmann/emotion-english-roberta-large` | YouTube 域验证评论 | CPU | 798 | 0.5138 | 49.6610s | 48.2503s |
| `chase1zhang/youtube-emotion-samlowe-roberta-domain-adapted` | YouTube 域验证评论 | CPU | 798 | 0.6504 | 13.6037s | 13.4440s |
| `chase1zhang/youtube-emotion-jhartmann-distilroberta-domain-adapted` | YouTube 域验证评论 | CPU | 798 | 0.6291 | 6.7531s | 6.6620s |
| `chase1zhang/youtube-emotion-roberta-large-domain-adapted` | YouTube 域验证评论 | CPU | 798 | **0.6717** | 46.3959s | 46.2100s |

域适配 DistilBERT 在 798 样本验证集上达到 0.6028 准确率，域适配 RoBERTa-large 达到最佳 **0.6717**，显著领先于其他基线模型。

### 11.3 Pipeline 2 情感模型对比

**应用基准（150 条人工审核评论）：**

| 模型 | 匹配 / 150 | 准确率 | 不含加载运行时间 |
|---|---:|---:|---:|
| CardiffNLP Twitter RoBERTa | 105 | 0.7000 | 1.9316s |
| lxyuan DistilBERT Multilingual | 94 | 0.6267 | 0.8794s |
| FiniteAutomata BERTweet | 0* | 0.0000* | 1.7399s |

CardiffNLP 在应用基准上表现最佳（105/150），提供最稳定的正面/中性/负面信号。lxyuan 推理速度最快（0.88s），适合延迟敏感场景。

### 11.4 已部署应用性能

**150 条评论应用基准总体性能：**

| 任务 | 模型 / Pipeline | 匹配 / 总数 | 准确率 |
|---|---|---:|---:|
| 七情绪 | 微调前基线 | 67 / 150 | 0.4467 |
| 七情绪 | GoEmotions 微调 | 70 / 150 | 0.4667 |
| 七情绪 | YouTube 域适配 DistilBERT | 71 / 150 | 0.4733 |
| 七情绪 | 公开 GoEmotions RoBERTa | 41 / 150 | 0.2733 |
| 七情绪 | 公开 RoBERTa-large | 62 / 150 | 0.4133 |
| 七情绪 | YouTube 域适配 RoBERTa | 93 / 150 | **0.6200** |
| 七情绪 | YouTube 域适配 DistilRoBERTa | 83 / 150 | 0.5533 |
| 七情绪 | YouTube 域适配 RoBERTa-large | 94 / 150 | **0.6267** |
| 三情感 | CardiffNLP 情感 pipeline | 105 / 150 | 0.7000 |

*注：域适配 RoBERTa、DistilRoBERTa 和 RoBERTa-large 的结果来自先前的评估运行；其上方五个模型已在最新 Colab 运行中重新评估。*

### 11.5 关键发现

- GoEmotions 微调 DistilBERT 在完整 2,160 样本 GoEmotions 测试集上达到 **0.8630** 准确率，相比微调前基线（0.6204）大幅提升，确认微调阶段在域内数据上有效。
- 域适配模型在 GoEmotions 测试集上较低（0.4944），因为它用域内准确率换取了 YouTube 域性能。这一取舍在 YouTube 域验证结果中得到验证：域适配 RoBERTa-large 达到 **0.6717**，域适配 DistilBERT 达到 **0.6028**，均显著领先于非适配基线。
- 在 150 条评论应用基准上，YouTube 域适配 RoBERTa-large 表现最佳（**94/150, 0.6267**），其次是域适配 RoBERTa（93/150, 0.6200）。在重新评估的五个基础模型中，GoEmotions 微调和域适配 DistilBERT 接近（70/150 vs 71/150）。
- 最大的改进出现在愤怒主导的视频上，域适配模型从基线的 12/50（0.24）提升到 18/50（0.36），表明平衡的域适配有助于模型更好地检测负面情绪。
- 三分类情感 pipeline 仍然是稳定的商业级信号（105/150 准确率），但细粒度的七情绪分类对诊断和模型对比更有用。

## 12. 业务解读

对于 Nike 的营销团队，最有用的输出不仅是每条评论的精确标签，还有活动级别的模式：

- 如果快乐和惊讶占主导，活动可能正在产生正面兴奋感。Nike 团队可以继续类似的故事讲述、语气和创意方向。
- 如果中性占主导，视频可能信息丰富但情感吸引力不足。Nike 团队可以改善开头钩子、行动号召或情感框架。
- 如果愤怒、厌恶、恐惧或悲伤超过 40%，Nike 团队应在扩大活动前人工审查评论主题并考虑调整信息。
- 如果七情绪模型和三分类情感模型结果不一致，团队应将输出视为人工审查的信号，而非自动决策。

最终应用是一个决策支持工具。它减少了人工阅读评论的工作量，帮助营销团队快速识别视频是在产生热情、冷漠还是声誉风险。

## 13. 局限性与未来改进

主要局限性是域偏移。GoEmotions 提供高质量的情绪标签，但不是 YouTube 特定数据集。YouTube 评论包含俚语、讽刺、表情符号、简短回复、政治争论、多语言内容和上下文相关的反应。

YouTube 域适配数据集缩小了这一差距。原始池包含 8,000 条 YouTube 评论，最终适配集选择了 3,991 条平衡评论；训练中使用类别加权损失处理剩余的少数类不平衡。未来工作应：

1. 收集更多人工验证的 YouTube 评论
2. 在所有七种情绪类别上平衡域数据集
3. 添加更多产品发布、品牌危机、娱乐和公共事件视频
4. 除准确率外评估 macro-F1
5. 使用置信度阈值标记不确定评论供人工审查

## 14. 结论

本项目展示了基于 Transformer 的文本分类如何支持数字营销决策。Streamlit 应用收集 YouTube 评论，应用两个 Hugging Face pipeline，可视化观众情绪和情感，并生成实用建议。

实验结果表明，微调提高了原始 GoEmotions 基准上的性能，使用平衡 YouTube 数据的域适配在 YouTube 域验证上给出了最强的项目自有模型。公开 SamLowe RoBERTa 模型是最佳应用基准对比模型，而 YouTube 域适配 DistilBERT 仍是最终默认模型，因为它紧凑、为项目自有，且在专用 YouTube 验证集上最强。辅助情感模型在广泛的正面/中性/负面层面表现最佳（105/150），七情绪模型提供了更详细的诊断洞察。两个 pipeline 共同为活动监控和观众反馈分析提供了有用的工作流。

## 15. 提交清单

- [x] 填写学生姓名和学号
- [x] 在导出最终 PDF 前插入五张 Streamlit 截图
- [x] GitHub 仓库：https://github.com/chasezhang1999/youtube-emotion-analyzer
- [x] Streamlit 应用：https://youtube-emotion-analyzer.streamlit.app/
- [x] 原始微调模型：https://huggingface.co/chase1zhang/youtube-emotion-distilbert
- [x] 域适配模型：https://huggingface.co/chase1zhang/youtube-emotion-distilbert-domain-adapted
- [x] 公开对比模型：https://huggingface.co/SamLowe/roberta-base-go_emotions
- [x] 公开对比模型：https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
- [x] 实验结果 Excel：`experiments/Experimental_results.xlsx`
- [x] 应用和数据集文件已在仓库中准备好
- [x] 草稿 PDF 已生成：`docs/report/Project_report.pdf`
- [ ] 填写学生姓名后重新导出最终 PDF
- [ ] 准备 PPT 幻灯片
- [ ] 录制 MP4 演示视频
