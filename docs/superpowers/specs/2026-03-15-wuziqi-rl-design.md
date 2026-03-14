# 五子棋RL训练平台设计

## 1. 项目概述

五子棋（Gomoku/五目並べ）是一种双人对弈策略游戏，需要在15x15的棋盘上率先连成五子。

**游戏变体：** 自由五子棋（Free-style Gomoku），无禁手限制。任何位置均可落子，连成五子即获胜。

本项目旨在构建一个完整的五子棋平台，支持：
- 完整的五子棋规则引擎
- 基于Gymnasium的RL训练环境
- PyGame人机对战UI
- 多级难度AI对战
- 完整的评测框架

## 2. 架构设计

### 2.1 分层架构

```
wuziqi/
├── wuziqi_core/       # 核心游戏引擎（无外部依赖）
│   ├── board.py       # 棋盘逻辑与状态
│   ├── player.py      # 玩家类
│   ├── game.py        # 游戏主逻辑
│   └── ai.py         # 简单AI（MCTS/规则）
│
├── wuziqi_gym/       # Gymnasium RL环境
│   ├── env.py         # Gymnasium环境实现
│   ├── obs_space.py   # 观察空间
│   └── action_space.py # 动作空间
│
├── wuziqi_ui/        # PyGame UI
│   ├── main.py        # 游戏主界面
│   ├── renderer.py    # 渲染器
│   └── controller.py  # 输入控制
│
└── trainer/          # RL训练与评测
    ├── train.py       # 训练入口
    ├── config.py      # 训练配置
    └── evaluate.py    # 评测框架
```

### 2.2 依赖关系

- `wuziqi_core`: 纯Python，无外部依赖
- `wuziqi_gym`: 依赖 core + gymnasium
- `wuziqi_ui`: 依赖 core + pygame
- `trainer`: 依赖 gym + stable-baselines3

## 3. 核心游戏引擎 (wuziqi_core)

### 3.1 棋盘状态

```python
class Board:
    SIZE = 15  # 15x15棋盘
    EMPTY = 0
    BLACK = 1  # 先手
    WHITE = 2  # 后手

    def __init__(self):
        self.grid = [[self.EMPTY] * self.SIZE for _ in range(self.SIZE)]
        self.current_player = self.BLACK
        self.move_count = 0

    def place(self, x: int, y: int) -> bool:
        """落子，返回是否成功"""
        if not self.is_valid_position(x, y):
            return False
        if self.grid[y][x] != self.EMPTY:
            return False
        self.grid[y][x] = self.current_player
        self.move_count += 1
        self._switch_player()
        return True

    def is_valid_position(self, x: int, y: int) -> bool:
        """检查坐标是否有效"""
        return 0 <= x < self.SIZE and 0 <= y < self.SIZE

    def get_winner(self) -> int:
        """检查是否有五连，返回获胜方: 0:无, 1:黑, 2:白"""
        # 检查四个方向：横、竖、斜、反对角
        for y in range(self.SIZE):
            for x in range(self.SIZE):
                if self.grid[y][x] == self.EMPTY:
                    continue
                player = self.grid[y][x]
                # 水平
                if x + 4 < self.SIZE and all(self.grid[y][x+i] == player for i in range(5)):
                    return player
                # 垂直
                if y + 4 < self.SIZE and all(self.grid[y+i][x] == player for i in range(5)):
                    return player
                # 对角线
                if x + 4 < self.SIZE and y + 4 < self.SIZE and all(self.grid[y+i][x+i] == player for i in range(5)):
                    return player
                # 反对角线
                if x - 4 >= 0 and y + 4 < self.SIZE and all(self.grid[y+i][x-i] == player for i in range(5)):
                    return player
        return 0

    def is_full(self) -> bool:
        """检查棋盘是否已满"""
        return self.move_count >= self.SIZE * self.SIZE

    def is_game_over(self) -> bool:
        """检查游戏是否结束（分出胜负或平局）"""
        return self.get_winner() != 0 or self.is_full()

    def get_valid_moves(self) -> list[tuple[int, int]]:
        """获取所有合法落子位置"""
        return [(x, y) for y in range(self.SIZE) for x in range(self.SIZE) if self.grid[y][x] == self.EMPTY]

    def _switch_player(self):
        """切换当前玩家"""
        self.current_player = self.BLACK if self.current_player == self.WHITE else self.WHITE

    def reset(self):
        """重置棋盘"""
        self.grid = [[self.EMPTY] * self.SIZE for _ in range(self.SIZE)]
        self.current_player = self.BLACK
        self.move_count = 0
```

### 3.2 胜负判定

- 横向、纵向、斜向任意方向连成5子即为获胜
- 棋盘下满为平局

### 3.3 简单AI

| 难度 | 实现方式 |
|------|----------|
| 入门(Lv1) | 随机落子 + 基础防守（阻挡三连） |
| 简单(Lv2) | 评估函数：进攻/防守权重 |
| 中等(Lv3) | Minimax + Alpha-Beta剪枝（深度3-5） |
| 困难(Lv4) | MCTS蒙特卡洛树搜索（模拟1000+次） |
| 大师(Lv5) | PPO训练模型（需训练后解锁） |

**说明：** 简单AI使用规则算法，无需训练。PPO模型需要单独训练。

## 4. Gymnasium环境 (wuziqi_gym)

### 4.1 观察空间

使用Box空间，形状为 (3, 15, 15) - 3通道CNN输入：

| 通道 | 描述 | 值范围 |
|------|------|--------|
| 通道0 | 己方棋子位置 | 0或1 |
| 通道1 | 对方棋子位置 | 0或1 |
| 通道2 | 空白位置 | 0或1 |

**说明：** 使用3通道表示更利于CNN特征提取。

### 4.2 动作空间

- **Discrete** 空间：225 (15x15个位置)
- 必须实现 `legal_actions_mask()` 方法，返回当前合法落子位置

**动作映射：**
- action 0-224 对应棋盘位置 (x, y)
- 行优先顺序: action = y * 15 + x
- 例如: action 0 = (0,0), action 15 = (0,15), action 16 = (1,0)

### 4.3 训练模式

采用 **Self-Play（自我对弈）** 模式：
- 对战双方使用同一策略网络
- 每局随机选择先手/后手
- 奖励在游戏结束时给出（win/lose/draw）
- 训练过程中模型与自身对弈，逐步提升

### 4.4 奖励设计

| 奖励 | 值 | 描述 |
|------|-----|------|
| win | +1.0 | 获胜 |
| lose | -1.0 | 失败 |
| draw | 0.0 | 平局 |
| step | -0.001 | 每步轻微惩罚（鼓励快速获胜） |

### 4.5 环境接口

```python
class WuziqiEnv(gym.Env):
    def reset(self, seed=None) -> tuple[np.ndarray, dict]:
        """重置环境，返回初始观察和info"""
        ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步，返回 (obs, reward, terminated, truncated, info)"""
        # action: 0-224 对应棋盘位置
        # 如果动作非法，返回负奖励并继续
        ...

    def render(self): ...
    def close(self): ...
    def legal_actions(self) -> list[int]:       # 返回当前合法动作列表
    def legal_actions_mask(self) -> np.ndarray:  # 返回225维布尔掩码
```

## 5. PyGame UI (wuziqi_ui)

### 5.1 界面布局

```
┌─────────────────────────────────────────┐
│  五子棋 v1.0            难度: 困难      │
├─────────────────────────────────────────┤
│                                         │
│            ┌───────────────┐            │
│            │               │            │
│            │    棋盘       │            │
│            │   (15x15)     │            │
│            │               │            │
│            └───────────────┘            │
│                                         │
│    黑方: 玩家(你)    vs    白方: AI     │
│                                         │
│         [新游戏]  [悔棋]  [认输]         │
└─────────────────────────────────────────┘
```

### 5.2 交互功能

- 鼠标左键点击落子
- 最后落子位置高亮（红色标记）
- 获胜/平局提示对话框
- 难度选择下拉菜单
- AI思考中提示

### 5.3 难度选择

| 难度 | 说明 |
|------|------|
| 入门 | 随机+基础防守 |
| 简单 | 评估函数 |
| 中等 | Minimax |
| 困难 | MCTS |
| 大师 | PPO训练模型 |

## 6. RL训练 (trainer)

### 6.1 训练配置

```python
@dataclass
class TrainConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 100_000
    max_steps: int = 200  # 单局最大步数（防止超长对局）
    env_make_kwargs: dict = None
    model_kwargs: dict = None
    save_freq: int = 10000
    eval_freq: int = 50000
```

### 6.2 训练流程

1. 初始化环境（双人对战）
2. 自我对弈收集经验
3. PPO更新策略网络
4. 定期评估 vs 随机AI（每eval_freq步）
5. 保存最佳模型（基于胜率）

**模型保存：**
- 保存路径: `./models/ppo_wuziqi.zip`
- 保存标准: 评估胜率超过90%时保存为最佳模型
- TensorBoard日志: `./logs/`

### 6.3 模型架构

```python
# 神经网络架构：卷积神经网络 (CNN)
class WuziqiPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层处理棋盘状态
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # 策略头：输出每个位置的概率
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 15 * 15, 512),
            nn.ReLU(),
            nn.Linear(512, 225)  # 15x15=225个落子位置
        )
        # 价值头：评估局面优劣
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 15 * 15, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
```

### 6.4 默认超参数

```python
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}
```

### 6.5 模型选择

- PPO (推荐)
- A2C
- DQN

## 7. 评测框架 (evaluate)

### 7.1 评测功能

| 功能 | 描述 |
|------|------|
| 对局评测 | AI vs 随机/规则AI |
| 胜率统计 | 计算各难度胜率 |
| 对局历史 | 控制台输出简要结果 |

**注意：** 排行榜功能作为Phase 2扩展。

### 7.2 评测接口

```python
def evaluate(model, opponent, num_games: int = 100) -> dict:
    """评测AI模型

    Args:
        model: 训练的模型 (str: 模型路径 或 object: 模型对象)
        opponent: 对手类型 ("random", "rule", "mcts")
        num_games: 对局数量

    Returns:
        {"win_rate": float, "wins": int, "losses": int, "draws": int}
    """
    wins = losses = draws = 0
    for i in range(num_games):
        result = play_game(model, opponent, first_player=(i % 2))
        if result == 1:  # AI获胜
            wins += 1
        elif result == -1:  # AI失败
            losses += 1
        else:  # 平局
            draws += 1

    return {
        "win_rate": wins / num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws
    }
```

### 7.3 使用示例

```bash
# 评测训练好的模型 vs 随机AI
python -m trainer.evaluate --model models/ppo_wuziqi --opponent random --games 100

# 评测模型 vs 规则AI
python -m trainer.evaluate --model models/ppo_wuziqi --opponent rule --games 50

# 评测不同难度AI
python -m trainer.evaluate --model models/ppo_wuziqi --opponent mcts --games 20
```

## 8. 实施计划

### Phase 1: 核心引擎
- 棋盘实现
- 落子逻辑
- 胜负判定
- 基础AI

### Phase 2: RL环境
- Gymnasium接口
- 观察/动作空间
- 奖励函数

### Phase 3: UI
- PyGame棋盘渲染
- 鼠标交互
- 难度选择

### Phase 4: 训练与评测
- PPO训练脚本
- 模型评估
- 对局历史记录

## 9. 扩展性

- 核心引擎独立，可直接用于其他项目
- 环境接口符合Gymnasium标准，易于接入新算法
- UI层可替换为Web/终端
- 可添加AlphaZero风格的MCTS+神经网络
