# Maze 参考数据生成器

需要 JDK 17 或更高版本。在选题目录运行：

```bash
bash generators/maze/generate.sh my-data/maze 60 30 10
python generators/state_guard.py my-data/maze/data.json
```

四个参数依次为输出目录、小型、中型、大型棋盘数量，允许某档为0。默认尺寸分别为9×9、11×11、13×13，每个棋盘通常生成六道问题；这些参数不是题型配额。

红色圆点是玩家，绿色格子是终点，蓝色格子是墙，白色格子可通行。题型包括玩家或终点定位、可移动方向、移动后位置、路径与转弯次数。

输出包含 `data.json`、`images/` 和 `states/`，使用新的输出目录。默认读取 `test_data/excluded-states.txt`，排除完整测试状态。`generate.sh` 编译 `src/main/java/`，使用 `lib/gson.jar` 离线运行；编译产物写入本目录的 `build/`。

也可使用保留的 Gradle 工程。在本生成器目录运行：

```bash
./gradlew run --args="maze_dataset 60 30 10 ../../test_data/excluded-states.txt"
```

该方式首次运行可能下载 Gradle 和依赖。五个参数不可省略；不能绕过测试状态排重。

生成器代码的许可证见 [LICENSE](../LICENSE)，附带依赖的说明见 [lib/NOTICE.txt](lib/NOTICE.txt)。
