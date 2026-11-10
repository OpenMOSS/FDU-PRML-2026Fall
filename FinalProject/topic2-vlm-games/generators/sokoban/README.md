# Sokoban 参考数据生成器

在选题目录运行：

```bash
python generators/sokoban/main.py --boards 100 --output my-data/sokoban
python generators/state_guard.py my-data/sokoban/data.json
```

依赖 NumPy、Pillow、Matplotlib，见根目录 `requirements-generation.txt`。素材按代码位置解析，无需手动复制。输出目录应为新的或空目录。

`--boards` 表示尝试生成的棋盘数量，不是问答条数。程序按原逻辑生成不同尺寸和箱子数量的棋盘；单箱棋盘包含最多六类问题，多箱棋盘包含两类。超时或无法生成的题目会被记录并跳过，因此最终题数不固定。

```text
my-data/sokoban/
  data.json
  images/
  states/
  generation-report.json
```

题型包括玩家位置、移动后的玩家或箱子位置、距离、到达目标所需步数及路径等。步数按每次移动计算，包含行走和推箱。

默认加载 `test_data/excluded-states.txt`，在渲染和出题前拒绝相同的完整测试状态。修改生成器时须保留该检查。问题和分析从棋盘状态生成；完整状态文件不应在后续处理时丢弃。

代码入口为 `main.py`，游戏与题目逻辑在 `sokoban.py`，图像生成在 `textured_sokoban.py` 和 `sokoban_texture_handler.py`。完整参数见 `python generators/sokoban/main.py --help`。

生成器代码的许可证见 [LICENSE](../LICENSE)。
