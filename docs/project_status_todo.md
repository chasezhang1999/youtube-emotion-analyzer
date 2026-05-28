# YouTube 评论情绪分析项目：当前结构与 TODO

最后更新：2026-05-28

这份文档是当前 ISOM5240 Assignment 2 项目的中文工作地图。后续继续开发、训练、改报告或改 PPT 前，先看这份，避免代码、实验结果、报告、Excel 和网页 PPT 的数字不一致。

## 1. 当前项目快照

- GitHub 仓库：https://github.com/chasezhang1999/youtube-emotion-analyzer
- GitHub Pages 网页 PPT：https://chasezhang1999.github.io/youtube-emotion-analyzer/
- Streamlit app 目标地址：https://youtube-emotion-analyzer.streamlit.app/
- 当前项目目录：`youtube_emotion_project/`
- 当前分支：`main`
- 本轮新增/更新重点：YouTube-domain 8,000 条原始池、5,000 条最终 adaptation 数据集、平衡构建脚本、数据规模相关文档。

## 2. 项目目标

项目要做的是一个面向数字营销公司的 YouTube 评论情绪分析系统。用户输入 YouTube 视频链接后，应用自动抓取评论，运行七情绪分类和三分类情感分析，展示图表、评论级预测结果，并给出营销决策建议。

课程作业相关交付物包括：

- Streamlit Web App
- GitHub 仓库
- Hugging Face fine-tuned model
- 模型性能结果 Excel
- 实验结果 Excel
- 项目报告
- Presentation / PPT
- 项目截图和案例演示

## 3. 重要作业文件

### 3.1 当前项目内文件

| 文件 | 状态 | 用途 |
|---|---|---|
| `streamlit_app.py` | 已完成核心功能 | Streamlit 应用入口 |
| `experiments/Experimental_results.xlsx` | 已刷新 | 当前项目实验结果汇总 workbook |
| `experiments/Performance_result.xlsx` | 已生成 | 课程 Performance workbook 的项目内版本 |
| `ppt/index.html` | 已部署到 GitHub Pages | 网页 PPT |
| `docs/report/Project_report_skeleton.md` | 已更新 | 英文报告骨架 |
| `docs/report/Project_report_chinese.md` | 已更新 | 中文报告 |
| `docs/report/PPT_design.md` | 已更新 | PPT 设计说明 |
| `README.md` | 已更新 | 项目说明 |
| `scripts/evaluate_updated_results.py` | 已完成 | 当前可复跑评估脚本 |
| `scripts/build_youtube_domain_balanced_dataset.py` | 已新增 | 抓取、分类、平衡 YouTube-domain 评论 |
| `scripts/train_youtube_domain_all_models.py` | 已新增 | 多模型 YouTube-domain fine-tuning 脚本 |
| `notebooks/fine_tune_youtube_domain_all_models.ipynb` | 已新增 | Colab 多模型调优 notebook |
| `data/youtube_domain_training_comments_8000_assistant_labeled.csv` | 已生成 | 8,000 条 YouTube-domain 标注原始池 |
| `data/youtube_domain_7class_assistant/all.csv` | 已刷新 | 5,000 条最终 YouTube-domain adaptation 数据集 |

### 3.2 项目目录外但属于作业要求的文件

| 文件 | 当前路径 | 当前状态 | 说明 |
|---|---|---|---|
| `Performance_result.xlsx` | `../Performance_result.xlsx` | 仍是课程模板 / 示例内容 | 这是老师要求的 performance result Excel，需要用本项目结果填好 |
| `ModelPerformance_student.ipynb` | `../ModelPerformance_student.ipynb` | 课程示例 Notebook | 当前是 ASR/audio quality 示例，不是本项目内容 |

注意：`../Performance_result.xlsx` 不在当前 git 仓库目录里。项目内已生成 `experiments/Performance_result.xlsx`，方便 git 跟踪和归档；最终打包时如果课程要求使用 Assignment2 根目录文件，再把项目内最终版复制回 `../Performance_result.xlsx`。

## 4. 当前目录结构

```text
youtube_emotion_project/
|-- streamlit_app.py
|   Streamlit 主应用。负责 URL 输入、模型选择、单模型分析、多模型对比、
|   图表、表格、营销建议、决策 pipeline、CSV 导出。
|
|-- youtube_emotion/
|   |-- core.py
|   |   纯逻辑层：YouTube URL 解析、文本清洗、标签归一化、情绪汇总、
|   |   营销建议、campaign decision pipeline。
|   |
|   |-- model_runner.py
|   |   Hugging Face 模型常量、模型选项、pipeline 输出解析。
|   |
|   |-- youtube_client.py
|   |   YouTube Data API 评论抓取。
|   |
|   |-- dataset_prep.py
|   |   GoEmotions 过滤、平衡、分层采样和 split 工具。
|   |
|   `-- __init__.py
|
|-- data/
|   |-- go_emotions_7class/
|   |   |-- train.csv       5,000 条训练数据
|   |   |-- validation.csv  406 条验证数据
|   |   `-- test.csv        1,000 条测试数据
|   |
|   |-- youtube_domain_7class_assistant/
|   |   |-- train.csv       4,000 条训练数据
|   |   |-- validation.csv  1,000 条验证数据
|   |   `-- all.csv         5,000 条总数据
|   |
|   |-- youtube_domain_7class_balanced/
|   |   早期平衡版本 YouTube-domain 数据。
|   |
|   `-- youtube_domain_training_comments_*.csv
|       原始、扩展、assistant-labeled YouTube 评论数据，含 8,000 条原始池。
|
|-- experiments/
|   |-- Experimental_results.xlsx
|   |   当前实验结果 workbook。
|   |
|   |-- Performance_result.xlsx
|   |   课程 Performance workbook 的项目内版本。
|   |
|   |-- experimental_results_template.csv
|   |   GoEmotions 1,000 条测试集模型选择结果。
|   |
|   |-- youtube_domain_validation_performance.csv
|   |   上一轮 YouTube-domain 592 条验证集结果，5,000 条数据刷新后需要重跑。
|   |
|   |-- app_model_comparison_5models.csv
|   |   150 条 app benchmark 简表。
|   |
|   |-- app_performance_model_comparison.csv
|   |   app benchmark 逐视频 + overall 结果。
|   |
|   |-- app_per_comment_manual_labels.csv
|   |   150 条人工审核评论 benchmark 宽表。
|   |
|   |-- app_per_comment_model_predictions.csv
|   |   每条评论、每个模型的预测长表。
|   |
|   |-- app_runtime_summary.csv
|   |   CPU runtime 结果。
|   |
|   `-- model_revision_summary.csv
|       记录每个 Hugging Face 模型的 SHA 和 last modified。
|
|-- scripts/
|   |-- prepare_go_emotions_7class.py
|   |   准备 GoEmotions 七情绪数据。
|   |
|   |-- expand_youtube_domain_dataset.py
|   |   扩展 YouTube-domain 评论数据。
|   |
|   |-- label_youtube_domain_comments.py
|   |   生成 assistant-labeled YouTube-domain split。
|   |
|   |-- build_youtube_domain_balanced_dataset.py
|   |   抓取、分类、平衡 YouTube-domain 评论，生成 8,000 原始池和 5,000 最终数据集。
|   |
|   |-- review_app_manual_labels.py
|   |   审核和修正 150 条 app benchmark 人工标签。
|   |
|   |-- evaluate_app_manual_benchmark.py
|   |   早期 app-only benchmark 脚本。
|   |
|   |-- evaluate_updated_results.py
|   |   当前主要评估脚本：GoEmotions test、YouTube-domain validation、
|   |   app benchmark、runtime、model revision、CSV 刷新。
|   |
|   `-- train_youtube_domain_all_models.py
|       多模型 YouTube-domain fine-tuning 脚本。
|
|-- notebooks/
|   |-- Fine_tune_Model.ipynb
|   |-- fine_tune_go_emotions_distilbert.ipynb
|   |   当前 DistilBERT 两阶段训练 Colab notebook。
|   |
|   |-- fine_tune_youtube_domain_all_models.ipynb
|   |   Colab 多模型 YouTube-domain 调优 notebook。
|   |
|   `-- testing_experiments.ipynb
|
|-- docs/
|   |-- report/
|   |   |-- Project_report_skeleton.md
|   |   |-- Project_report_chinese.md
|   |   `-- PPT_design.md
|   |
|   |-- fine_tuning_steps.md
|   |-- final_submission_checklist.md
|   |-- presentation_outline.md
|   |-- youtube_domain_annotation_guide.md
|   `-- project_status_todo.md
|
|-- ppt/
|   |-- index.html
|   `-- images/
|       网页 PPT 使用的 Streamlit 截图。
|
|-- tests/
|   单元测试：core、dataset prep、model runner、默认模型、YouTube client、
|   YouTube-domain labeling。
|
|-- requirements.txt
|   Streamlit Cloud 运行依赖。
|
|-- requirements_train.txt
|   Colab / 训练依赖。
|
`-- runtime.txt
```

## 5. 当前应用功能

已实现：

- 输入 YouTube 视频 URL。
- 抓取最多 100 条 top-level comments。
- 七情绪分类：
  - anger
  - disgust
  - fear
  - joy
  - neutral
  - sadness
  - surprise
- 三分类情感分析：
  - positive
  - neutral
  - negative
- 单模型分析模式。
- 多模型对比模式。
- 情绪分布图、摘要指标、评论级预测表。
- 营销建议。
- 决策 pipeline：
  - scale positive creative
  - improve engagement hook
  - review before scaling
  - monitor and review samples
- CSV 导出。

当前模型选择包括：

- `chase1zhang/youtube-emotion-distilbert-domain-adapted`
- `chase1zhang/youtube-emotion-distilbert`
- `SamLowe/roberta-base-go_emotions`
- `j-hartmann/emotion-english-distilroberta-base`
- `j-hartmann/emotion-english-roberta-large`
- `cardiffnlp/twitter-roberta-base-sentiment-latest`

## 6. 当前模型状态

### 6.1 项目自研 Hugging Face 模型

| 模型 | URL | 最新评估 SHA | Hugging Face last modified |
|---|---|---|---|
| GoEmotions fine-tuned DistilBERT | https://huggingface.co/chase1zhang/youtube-emotion-distilbert | `751f3c11a47be9b4621602d0b4048d39e382bb35` | 2026-05-27 08:37:07 UTC |
| YouTube-domain adapted DistilBERT | https://huggingface.co/chase1zhang/youtube-emotion-distilbert-domain-adapted | `7ace4f7edaa72eccd65574a5a517ae4ce92aec26` | 2026-05-27 09:56:25 UTC |

### 6.2 公共对比模型

| 模型 | 用途 |
|---|---|
| `SamLowe/roberta-base-go_emotions` | 公共 GoEmotions RoBERTa，对 app benchmark 表现最好 |
| `j-hartmann/emotion-english-distilroberta-base` | 预调优七情绪 baseline |
| `j-hartmann/emotion-english-roberta-large` | 更大 RoBERTa 七情绪 baseline |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 三分类情感辅助 pipeline |

## 7. 当前数据集规模

| 数据集 | 文件 | 数据行数 |
|---|---|---:|
| GoEmotions train | `data/go_emotions_7class/train.csv` | 5,000 |
| GoEmotions validation | `data/go_emotions_7class/validation.csv` | 406 |
| GoEmotions test | `data/go_emotions_7class/test.csv` | 1,000 |
| YouTube-domain train | `data/youtube_domain_7class_assistant/train.csv` | 4,000 |
| YouTube-domain validation | `data/youtube_domain_7class_assistant/validation.csv` | 1,000 |
| YouTube-domain all | `data/youtube_domain_7class_assistant/all.csv` | 5,000 |
| YouTube-domain raw pool | `data/youtube_domain_training_comments_8000_assistant_labeled.csv` | 8,000 |
| App benchmark | `experiments/app_per_comment_manual_labels.csv` | 150 |

注意：CSV 的 `wc -l` 会比数据行多 1，因为包含 header。

重要区分：

- GoEmotions 训练集是 5,000 条，用于通用七情绪 fine-tuning / model selection。
- YouTube-domain raw pool 已扩展到 8,000 条，来自 71 个视频。
- YouTube-domain 最终 adaptation 数据集是 5,000 条；train/validation 为 4,000 / 1,000。
- 新的 5,000 条数据已尽量平衡且不复制样本：anger 905、disgust 505、fear 608、joy 904、neutral 904、sadness 802、surprise 372。
- surprise 和 disgust 仍是相对少数类；当前训练脚本继续使用 class-weighted loss 缓解剩余不平衡。

## 8. 当前评估结果

### 8.1 GoEmotions 1,000 条测试集

| 模型 | 匹配数 | 准确率 |
|---|---:|---:|
| `j-hartmann/emotion-english-distilroberta-base` | 668 / 1,000 | 0.6680 |
| `chase1zhang/youtube-emotion-distilbert` | 703 / 1,000 | 0.7030 |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | 628 / 1,000 | 0.6280 |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 646 / 1,000 | 0.6460 |

解释：

- GoEmotions fine-tuned DistilBERT 在 GoEmotions 测试集上最好。
- YouTube-domain adapted DistilBERT 在 GoEmotions 上下降是正常的，因为它向 YouTube 评论风格适配。

### 8.2 上一轮 YouTube-domain 592 条验证集结果

| 模型 | 匹配数 | 准确率 |
|---|---:|---:|
| `j-hartmann/emotion-english-distilroberta-base` | 279 / 592 | 0.4713 |
| `chase1zhang/youtube-emotion-distilbert` | 343 / 592 | 0.5794 |
| `chase1zhang/youtube-emotion-distilbert-domain-adapted` | 392 / 592 | 0.6622 |
| `SamLowe/roberta-base-go_emotions` | 388 / 592 | 0.6554 |
| `j-hartmann/emotion-english-roberta-large` | 262 / 592 | 0.4426 |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 337 / 592 | 0.5693 |

解释：

- 当前自研 YouTube-domain adapted DistilBERT 在这个验证集上略高于 SamLowe。
- 但这个结果来自上一轮 592 条验证集。当前 YouTube-domain 数据已重建为 5,000 条 all / 4,000 train / 1,000 validation；需要重新训练并刷新评估。
- 这个验证集来自 assistant-labeled YouTube-domain 数据，不等同于最终独立 app benchmark。

### 8.3 150 条 app 人工 benchmark

| 模型 / Pipeline | 匹配数 | 准确率 |
|---|---:|---:|
| Pre-tuning baseline | 67 / 150 | 0.4467 |
| GoEmotions fine-tuned DistilBERT | 70 / 150 | 0.4667 |
| YouTube-domain adapted DistilBERT | 71 / 150 | 0.4733 |
| Public SamLowe GoEmotions RoBERTa | 80 / 150 | 0.5333 |
| Public j-hartmann RoBERTa-large | 62 / 150 | 0.4133 |
| CardiffNLP sentiment pipeline | 105 / 150 | 0.7000 |

当前关键结论：

- 自研 YouTube-domain adapted DistilBERT 是当前最强的项目自研 DistilBERT。
- 但是 SamLowe 公共 RoBERTa 在独立 150 条 app benchmark 上更强。
- 因此下一步应该把所有七情绪基准模型都做同一套 YouTube-domain fine-tuning，然后再公平对比。

## 9. 已完成工作

- [x] 搭建 Streamlit 应用。
- [x] 实现 YouTube API 评论抓取。
- [x] 实现七情绪分类和三分类情感分析。
- [x] 实现营销建议和 campaign decision pipeline。
- [x] 实现多模型对比模式。
- [x] 准备 GoEmotions 七情绪数据集。
- [x] 扩展 GoEmotions train/test 到 5,000 / 1,000。
- [x] 扩展 YouTube-domain assistant-labeled 原始池到 8,000 条。
- [x] 生成 YouTube-domain 5,000 条最终 adaptation 数据集。
- [x] 训练并上传 DistilBERT 原始 fine-tuned 模型。
- [x] 训练并上传 DistilBERT YouTube-domain adapted 模型。
- [x] 建立 150 条 app 人工 benchmark。
- [x] 刷新最新模型结果。
- [x] 生成 `experiments/Experimental_results.xlsx`。
- [x] 生成 `experiments/Performance_result.xlsx`。
- [x] 更新 `docs/final_submission_checklist.md`，纳入 Performance workbook。
- [x] 新增多模型 YouTube-domain fine-tuning 脚本。
- [x] 新增 Colab 多模型调优 notebook。
- [x] 更新 README、英文报告、中文报告、PPT 设计、presentation outline。
- [x] 制作网页 PPT，包含项目截图和案例演示。
- [x] 发布 GitHub Pages。
- [x] 单元测试通过：31 tests OK。

## 10. Performance_result.xlsx 状态

已完成：

- [x] 生成项目内版本：`experiments/Performance_result.xlsx`。
- [x] 使用本项目当前结果填写 Performance sheet。
- [x] Pipeline 1 填七情绪分类模型，因为它与项目目标高度相关，需要 fine-tuning。
- [x] Pipeline 2 填三分类 sentiment pipeline，作为辅助 pipeline。
- [x] Step 1/2a 放入可比 emotion models 和当前 benchmark 结果。
- [x] Step 2b 说明当前模型选择和 caveat：独立 app benchmark 上 SamLowe 仍强于当前 DistilBERT adapted，所以需要做公平的多模型 YouTube-domain adaptation。
- [x] Step 3 纳入 150 条 app-level benchmark。
- [x] 更新 `docs/final_submission_checklist.md`，明确两个 Excel：
  - `Experimental_results.xlsx`
  - `Performance_result.xlsx`

后续仍需：

- [ ] 如果课程最终要求使用 Assignment2 根目录文件，把项目内最终版复制回 `../Performance_result.xlsx`。
- [ ] 多模型 YouTube-domain fine-tuning 完成后，再刷新 `experiments/Performance_result.xlsx`，避免最终 Excel 和新结果不一致。

## 11. 重要 TODO：对所有七情绪基准模型做 YouTube-domain fine-tuning

为什么要做：

- 当前只有 DistilBERT 做了 YouTube-domain adaptation。
- SamLowe 公共 RoBERTa 没有 domain adaptation，却在 150 条 app benchmark 上高于自研 DistilBERT。
- 所以现在的对比不完全公平。
- 下一轮应该把所有七情绪基准模型都用同一套 YouTube-domain 数据调优，再比较 adapted 版本。

建议训练对象：

| 起始模型 | 当前角色 | 下一步输出模型 |
|---|---|---|
| `chase1zhang/youtube-emotion-distilbert` | 当前自研 GoEmotions DistilBERT | 重新训练或保留 domain-adapted DistilBERT |
| `SamLowe/roberta-base-go_emotions` | 当前 app benchmark 最强公共模型 | `youtube-emotion-samlowe-roberta-domain-adapted` |
| `j-hartmann/emotion-english-distilroberta-base` | 预调优公共 baseline | `youtube-emotion-jhartmann-distilroberta-domain-adapted` |
| `j-hartmann/emotion-english-roberta-large` | 更大模型 baseline | `youtube-emotion-roberta-large-domain-adapted` |

不建议把 Cardiff sentiment 当成七情绪模型调优。它应该继续作为三分类辅助 pipeline。

## 12. 多模型调优详细 TODO

### A. 训练管线

- [x] 新增 `scripts/train_youtube_domain_all_models.py`。
- [x] 新增 `notebooks/fine_tune_youtube_domain_all_models.ipynb`，用于一次性训练多个基准模型。
- [x] 所有模型使用同一份数据：
  - train：`data/youtube_domain_7class_assistant/train.csv`
  - validation：`data/youtube_domain_7class_assistant/validation.csv`
- [x] 统一七情绪 label mapping：
  - anger = 0
  - disgust = 1
  - fear = 2
  - joy = 3
  - neutral = 4
  - sadness = 5
  - surprise = 6
- [x] 使用相近训练参数：
  - best model metric：macro-F1
  - class-weighted loss
  - fixed random seed
  - consistent max length
  - early stopping 如可用
- [x] 脚本输出每个 adapted model 的：
  - validation accuracy
  - macro-F1
  - weighted-F1
  - per-class precision / recall / F1
  - runtime / training time
- [ ] 在 Colab GPU 上实际训练，并把每个 adapted model 上传到 Hugging Face。

### B. 评估管线

- [ ] 更新 `scripts/evaluate_updated_results.py`，支持 original vs adapted model family。
- [ ] 增加新 Hugging Face repo 的 revision tracking。
- [ ] 统一评估：
  - GoEmotions 1,000 test
  - YouTube-domain 1,000 validation
  - 150 app benchmark
- [ ] 增加 macro-F1，不只看 accuracy。
- [ ] 重新生成：
  - `experimental_results_template.csv`
  - `youtube_domain_validation_performance.csv`
  - `app_model_comparison_5models.csv` 或新的 expanded comparison CSV
  - `app_performance_model_comparison.csv`
  - `app_per_comment_model_predictions.csv`
  - `app_runtime_summary.csv`
  - `model_revision_summary.csv`
  - `Experimental_results.xlsx`
  - `Performance_result.xlsx`

### C. Streamlit 应用

- [ ] 在 `youtube_emotion/model_runner.py` 加入新的 adapted emotion models。
- [ ] 根据新结果决定默认 emotion model。
- [ ] 更新 comparison mode 默认模型。
- [ ] 保留 CardiffNLP sentiment pipeline。
- [ ] 本地/云端验证模型缓存和运行时间。

### D. 报告和 PPT

- [ ] 更新英文报告。
- [ ] 更新中文报告。
- [ ] 更新 PPT 设计文档。
- [ ] 更新 presentation outline。
- [ ] 更新 README。
- [ ] 更新 final submission checklist。
- [ ] 更新网页 PPT。
- [ ] 如果 app UI 或模型输出截图变化，重新截图。

### E. 验证与部署

- [ ] 跑单元测试：
  - `.venv/bin/python -m unittest discover -s tests -v`
- [ ] 检查 CSV / Excel / 报告 / PPT 数字一致。
- [ ] 本地通过 localhost 检查网页 PPT。
- [ ] commit。
- [ ] push 到 GitHub。
- [ ] 检查 GitHub Pages workflow。
- [ ] 检查线上页面是否包含新结果。

## 13. 建议下一步工作顺序

1. 在 Colab GPU 上运行 `notebooks/fine_tune_youtube_domain_all_models.ipynb`，训练所有 adapted emotion models。
2. 上传新的 Hugging Face repos，并记录每个模型 revision SHA。
3. 更新 `scripts/evaluate_updated_results.py`，纳入 original vs adapted model family。
4. 重新跑完整评估：GoEmotions test、YouTube-domain validation、150 条 app benchmark。
5. 刷新 `Experimental_results.xlsx` 和 `Performance_result.xlsx`。
6. 根据新结果更新 app 默认模型、报告、PPT、README、final checklist。
7. 如果课程最终要求根目录 Excel，把 `experiments/Performance_result.xlsx` 复制回 `../Performance_result.xlsx`。
8. commit、push，并验证 GitHub Pages。

## 14. 注意事项

- 150 条 app benchmark 不能用于训练，只能作为独立测试集。
- YouTube-domain assistant labels 可以用于 domain adaptation，但报告里要说明标签是 assistant-assisted。
- SamLowe 模型输出的是 28 类 GoEmotions 标签，评估时会映射到本项目 7 类。
- RoBERTa-large 建议在 Colab GPU 上训练，本地 CPU 会非常慢。
- 每次重新评估都要记录 Hugging Face model SHA。
- 如果最终模型变更，必须同步更新：
  - `youtube_emotion/model_runner.py`
  - `README.md`
  - `docs/report/*.md`
  - `docs/presentation_outline.md`
  - `docs/final_submission_checklist.md`
  - `ppt/index.html`
  - `experiments/Experimental_results.xlsx`
  - `Performance_result.xlsx`

## 15. 常用命令

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

刷新当前评估结果：

```bash
.venv/bin/python scripts/evaluate_updated_results.py
```

检查多模型训练配置：

```bash
.venv/bin/python scripts/train_youtube_domain_all_models.py --model all --dry-run
```

查看 git 状态：

```bash
git status -sb
git log --oneline -5
```

本地打开网页 PPT：

```bash
cd ppt
python3 -m http.server 8766
```

查看 GitHub Pages workflow：

```bash
gh run list --limit 5
```
