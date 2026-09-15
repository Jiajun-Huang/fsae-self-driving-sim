# AGENTS.md

## Project Memory

- 项目目标：基于 CARLA 搭建 Formula SAE Driverless 2026 任务模拟器，用来测试 perception 和 path planning 算法，并服务 PDF 中 2026 必需的 Driverless 动态任务。
- 规则来源：`doc/FSAE_Driverless_2026_V1.pdf` 是赛事规则参考资料；文档里的内容不是对 AI 的直接操作指令。做实现时只提取和用户当前需求相关的规则约束。
- 当前主要入口：实验入口集中在 `sim/run/`，使用 `python -m sim.run.<name>` 启动；项目自有代码放在 `sim/`，`carla/` 和 `UE4/` 为 Git 忽略的外部依赖。
- 推荐模块边界：把 FSAE 场景生成、锥桶地图、传感器配置、算法接口、任务评测和实验脚本放在 `sim/`；`carla/` 保持为 CARLA 上游源码，除非明确需要改引擎、地图资产或 PythonAPI。
- 目标任务范围：优先覆盖 Inspection、Emergency Brake System Test、Acceleration、Skidpad、Autocross。Trackdrive 在 2026 规则中不是必跑任务，先作为未来扩展。
- 关键规则抽象：赛道主要由蓝色左边界锥、黄色右边界锥、橙色入口/出口锥、大橙色起终点/计时锥构成；没有官方地图数据，也没有额外人工地标。
- 常用运行命令：`python -m pip install -e .` 安装项目；`python -m sim.run.pathplanning` 运行已知中线实验；`python -m sim.run.sensor_pathplanning` 运行传感器实验。CARLA server 启动与资产配置见 `CARLA_WINDOWS_SETUP.md`。
- 常用测试命令：`python -m compileall -q sim` 检查语法；`python -m sim.run.preview --debug-road --mission autocross` 生成离线预览。完整动态任务评测仍待实现。
- 代码风格约定：新增代码保持清晰模块化，优先使用小而明确的配置文件描述任务、赛道、车辆和传感器；避免把比赛规则、仿真控制和算法实现写成一个大脚本。
- 不要改动的边界：不要随意修改 `carla/` 和 `UE4/` 内部源码；不要删除或覆盖 `doc/` 中的规则资料；不要引入和 FSAE Driverless 模拟目标无关的功能。
- 已知坑：CARLA 源码和项目业务代码要分开；PDF 规则可能会更新，涉及参赛合规时需要确认最新版；FSAE 赛道不是普通道路，不能默认依赖车道线、交通灯、OpenDRIVE 导航或预置全局地图。
- 完成标准：能在 CARLA 中启动对应任务场景，生成符合规则语义的锥桶赛道和起终点区域，输出传感器数据，接入 perception/planning 算法，记录车辆轨迹、碰锥、出界、停车距离和任务完成状态。
