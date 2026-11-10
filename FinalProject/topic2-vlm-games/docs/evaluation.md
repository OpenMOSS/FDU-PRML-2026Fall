# 推理与评测

## 统一入口

在选题目录运行：

```bash
python evaluate.py --model /path/to/model --run-name experiment-01
```

仅允许 Qwen2.5-VL-3B-Instruct 及其训练得到的模型，不允许 7B。3B 基座和由其得到的完整/合并模型使用相同入口，不要求模型目录名称包含原基座名。合并模型无需 `trainer_state.json`，也无需 `--checkpoint-model`。最终提交必须为完整模型；开发期间可用 `--adapter /path/to/adapter` 配合对应的 `--model /path/to/base` 自测，不能只提交 adapter。直接挂载 LoRA adapter 的评测结果可能与合并后的模型存在一定差异。

模型目录须包含加载所需的配置、分词器、图像处理器与聊天模板文件，可由统一入口直接评测。

| 参数 | 作用 |
| --- | --- |
| `--games sokoban maze` | 指定游戏，默认三个游戏 |
| `--gpu ID` | 单个设备编号或 UUID；默认遵循现有 GPU 分配 |
| `--batch-size 64` | 每批提交的题数 |
| `--max-num-seqs 64` | vLLM 最大并发序列数 |
| `--gpu-memory-utilization 0.9` | 引擎显存使用比例 |
| `--max-model-len 8192` | 输入与输出的总上下文容量，输出上限仍为 4096 |
| `--output-dir outputs` | 结果根目录 |
| `--run-name NAME` | 独立运行目录名，省略时按时间命名 |
| `--limit 2` | 每游戏少量题目试运行，不能作为完整成绩 |
| `--dry-run` | 校验数据并展示命令，不加载模型 |

三个游戏在同一张卡上依次推理，然后自动计分。固定 temperature=0、最多4096个新token、原聊天模板及统一补充指令；不能通过底层命令改变评测协议。

## 提示词与输入

```bash
python evaluation/infer.py preview --games sokoban maze snake --count 1
```

此命令展示模型实际收到的文本，无需 GPU。完整提示词由 `evaluation/infer.py` 的 `build_prompt(sample, "qwen-json")` 构建；不同题目的选项标签和答案形式由同一套规则识别。额外指令要求逐步推理，并以含 `answer` 字段的 JSON 给出最终答案。

评测模型只接收图片及问题文本（包括原有选项）和补充指令，不接收参考答案、analysis 或状态 JSON。模型自己的聊天模板由加载的模型目录提供。

## 结果与恢复

```text
outputs/<run-name>/
  protocol.json               # 固定协议、测试数据和模型文件身份
  manifest.json               # 环境与推理配置
  launcher.log                # 推理日志
  games/<game>.jsonl          # 逐题原始回答
  timing.jsonl                # 模型加载和批次生成耗时
  evaluation/summary.json
  evaluation/by_game.csv
  evaluation/<game>/samples.jsonl
  evaluation/<game>/errors.jsonl
  evaluation/report.md
  completion.json             # 推理和计分成功后生成
```

运行时可查看 `launcher.log`。中断后使用相同参数和运行名重启，已完成批次会保留；模型内容、资源参数或评测代码发生变化时应使用新运行名，不混用结果。权重会计算内容哈希，启动前有一定磁盘读取开销。运行期间不要修改模型文件。

各游戏分别报告准确率，全部所选题目进入分母。缺失或无法解析的答案不会被剔除；截断样本也不自动丢弃，按实际输出计分。计分使用规则提取和题型对应的答案比较，并非要求整个回答是严格 JSON 才能判对；可识别的代码围栏内 JSON 等输出仍由原规则处理。最终行 JSON 合规率、答案可提取率、截断率属于不同统计。

结果中的总体准确率仅是汇总指标，不代表课程评分公式。即使使用贪婪解码，硬件、批量或数值计算也可能带来少量重复评测差异，应保留实际配置及逐题结果。
