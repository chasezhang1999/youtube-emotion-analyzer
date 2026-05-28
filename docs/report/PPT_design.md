# PPT 设计方案 — YouTube Audience Emotion Analyzer

---

## Slide 1: 封面

- **标题：** YouTube Audience Emotion Analyzer for Digital Marketing Campaigns
- **副标题：** ISOM5240 Deep Learning Business Applications with Python
- **姓名：** ZHANG Xinchao (21257618), Yao Ziyue (21260768)
- **日期：** May 2026
- **公司：** InsightWave Digital Marketing Agency

---

## Slide 2: 项目背景与问题

- **标题：** 问题是什么？
- **内容：**
  - YouTube 评论区包含大量用户反馈，但人工阅读效率低、主观、难以规模化
  - 营销团队需要快速了解观众对视频的情绪反应
  - 需要一个自动化工具，将评论转化为可操作的营销建议
- **视觉：** 左侧放一张 YouTube 评论截图，右侧放痛点列表

---

## Slide 3: 解决方案概览

- **标题：** 我们的解决方案
- **内容：**
  - Streamlit Web 应用：输入 YouTube 链接 → 自动分析前 100 条评论
  - 双流水线架构：
    - 七情绪分类（愤怒、厌恶、恐惧、喜悦、中性、悲伤、惊讶）
    - 三分类情感（正面、中性、负面）
  - 自动生成营销建议 + CSV 下载
- **视觉：** 应用架构流程图（YouTube API → 模型 → 仪表板）

---

## Slide 4: 数据集

- **标题：** 数据来源与处理
- **内容：**
  - GoEmotions 数据集：5,000 训练样本，1,000 测试样本，7 类分层采样
  - YouTube 领域适配数据集：8,000 条原始评论池，最终选取 5,000 条，来自 71 个视频
  - 人工审核测试集：150 条评论，3 个视频 × 50 条
  - 重新平衡：最终 5,000 条中 anger/joy/neutral 约 900 条，sadness 802，fear 608，disgust 505，surprise 372；训练继续加入类别加权损失
- **视觉：** 数据集分布柱状图（7 种情绪的数量对比）

---

## Slide 5: 模型开发流程

- **标题：** 两阶段模型训练
- **内容：**
  - 基础模型：`distilbert-base-uncased`
  - 第一阶段：GoEmotions 七情绪调优 → `chase1zhang/youtube-emotion-distilbert`
  - 第二阶段：YouTube 领域适配 → `chase1zhang/youtube-emotion-distilbert-domain-adapted`
  - 关键优化：数据平衡 + class-weighted loss + macro_f1 模型选择
- **视觉：** 两阶段训练流程图

---

## Slide 6: 五模型对比结果

- **标题：** 谁是最好的情绪模型？
- **内容：** 150 条人工审核评论上的准确率对比

| 模型 | 准确率 |
|---|---:|
| **Public GoEmotions RoBERTa** | **0.5333** |
| YouTube-domain adapted | 0.4733 |
| GoEmotions fine-tuned | 0.4667 |
| Pre-tuning baseline | 0.4467 |
| Public RoBERTa-large | 0.4133 |
| Sentiment pipeline | 0.7000 |

- **视觉：** 柱状图，公共最佳模型和自研最终模型分别标注

---

## Slide 7: 逐视频分析

- **标题：** 不同视频类型的表现
- **内容：**
  - 视频 1（中性为主）：三个模型差异不大
  - 视频 2（喜悦为主）：YouTube-domain adapted 与 fine-tuned 持平
  - 视频 3（愤怒为主）：YouTube-domain adapted 从 12/50 提升到 18/50，提升最大
- **视觉：** 三个视频的分组柱状图

---

## Slide 8: 应用演示

- **标题：** Streamlit App 演示
- **内容：**
  - 应用网址：https://youtube-emotion-analyzer.streamlit.app/
  - 功能截图：
    - 输入界面
    - 情绪分布图
    - 模型对比模式
    - 营销建议
- **视觉：** 2-3 张应用截图拼接

---

## Slide 9: 关键发现

- **标题：** 关键发现
- **内容：**
  1. 调优提升 GoEmotions 测试集准确率：0.6680 → 0.7030
  2. YouTube 领域适配在领域验证集排名第一：0.6622
  3. 平衡数据 + 类别加权是解决中性偏差的关键
  4. 愤怒情绪检测提升最大（+50%）
  5. 公共 SamLowe 模型在 app benchmark 中表现最好，说明模型对比模式有实际价值
- **视觉：** 关键数字用大字体突出

---

## Slide 10: 业务价值

- **标题：** 对营销团队意味着什么？
- **内容：**
  - 喜悦 + 惊讶为主 → 继续当前创意方向
  - 中性为主 → 优化标题和情感框架
  - 负面情绪 > 40% → 人工审查，调整信息策略
  - 本工具是决策支持，不是自动决策
- **视觉：** 红绿灯式的情绪指示图

---

## Slide 11: 局限性与未来改进

- **标题：** 局限性与下一步
- **内容：**
  - 当前局限：
    - YouTube 领域数据集仍然较小
    - 讽刺和多语言评论仍然困难
    - 七情绪分类准确率有提升空间
  - 未来方向：
    - 扩大人工标注数据集
    - 评估 macro-F1 指标
    - 增加置信度阈值标记不确定评论
- **视觉：** 左右分栏（局限 vs 改进）

---

## Slide 12: 总结与致谢

- **标题：** 总结
- **内容：**
  - 项目成果：Streamlit App + 5 个模型对比 + 完整实验
  - 最终模型：YouTube-domain adapted DistilBERT，领域验证准确率 0.6622；app benchmark 为 71/150
  - GitHub：https://github.com/chasezhang1999/youtube-emotion-analyzer
  - App：https://youtube-emotion-analyzer.streamlit.app/
  - 感谢教授和助教！
- **视觉：** 简洁总结 + 项目链接二维码

---

## 设计建议

- **配色：** 蓝色系（专业、科技感），搭配红色强调负面情绪
- **字体：** 标题用粗体无衬线字体，正文用常规字体
- **图表：** 使用 Chart.js 或 matplotlib 风格的柱状图/饼图
- **每页文字：** 不超过 6 行，关键词加粗
- **总页数：** 12 页，控制在 10 分钟演讲内
