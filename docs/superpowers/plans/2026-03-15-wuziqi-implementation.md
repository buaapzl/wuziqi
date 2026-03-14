# 五子棋RL训练平台实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建完整的五子棋RL训练平台，包含核心引擎、RL环境、PyGame UI、多级难度AI对战、训练与评测框架

**Architecture:** 分层架构 - 核心引擎(wuziqi_core)无外部依赖，RL环境依赖gymnasium，UI依赖pygame，训练依赖stable-baselines3

**Tech Stack:** Python, PyGame, Gymnasium, stable-baselines3, PyTorch

---

## Chunk 1: 项目初始化与核心引擎 (wuziqi_core)

### Task 1: 初始化项目结构

**Files:**
- Create: `/Users/rr2017/code/wuziqi/wuziqi_core/__init__.py`
- Create: `/Users/rr2017/code/wuziqi/wuziqi_core/board.py`
- Create: `/Users/rr2017/code/wuziqi/wuziqi_core/player.py`
- Create: `/Users/rr2017/code/wuziqi/wuziqi_core/game.py`
- Create: `/Users/rr2017/code/wuziqi/wuziqi_core/ai.py`
- Test: `/Users/rr2017/code/wuziqi/tests/wuziqi_core/__init__.py`
- Test: `/Users/rr2017/code/wuziqi/tests/wuziqi_core/test_board.py`

- [ ] **Step 1: 创建项目目录结构**

```bash
mkdir -p /Users/rr2017/code/wuziqi/{wuziqi_core,wuziqi_gym,wuziqi_ui,trainer,tests/wuziqi_core}
touch /Users/rr2017/code/wuziqi/wuziqi_core/__init__.py
touch /Users/rr2017/code/wuziqi/tests/wuziqi_core/__init__.py
```

- [ ] **Step 2: 编写 Board 类测试**

```python
# /Users/rr2017/code/wuziqi/tests/wuziqi_core/test_board.py
import pytest
import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')
from wuziqi_core.board import Board

def test_board_init():
    board = Board()
    assert board.SIZE == 15
    assert board.EMPTY == 0
    assert board.BLACK == 1
    assert board.WHITE == 2
    assert board.current_player == Board.BLACK
    assert board.move_count == 0

def test_is_valid_position():
    board = Board()
    assert board.is_valid_position(0, 0) == True
    assert board.is_valid_position(14, 14) == True
    assert board.is_valid_position(-1, 0) == False
    assert board.is_valid_position(0, 15) == False

def test_place_stone():
    board = Board()
    assert board.place(7, 7) == True
    assert board.grid[7][7] == Board.BLACK
    assert board.move_count == 1
    assert board.current_player == Board.WHITE

def test_place_invalid_position():
    board = Board()
    assert board.place(-1, 0) == False
    assert board.place(0, 15) == False

def test_place_occupied():
    board = Board()
    board.place(7, 7)
    assert board.place(7, 7) == False

def test_get_winner_none():
    board = Board()
    assert board.get_winner() == 0

def test_get_winner_horizontal():
    board = Board()
    # Black places 5 in a row horizontally
    for i in range(5):
        board.place(i, 7)
    assert board.get_winner() == Board.BLACK

def test_get_winner_vertical():
    board = Board()
    # Black places 5 in a row vertically
    for i in range(5):
        board.place(7, i)
    assert board.get_winner() == Board.BLACK

def test_get_winner_diagonal():
    board = Board()
    # Black places 5 in a row diagonally
    for i in range(5):
        board.place(7 + i, 7 + i)
    assert board.get_winner() == Board.BLACK

def test_get_winner_anti_diagonal():
    board = Board()
    # Black places 5 in a row anti-diagonally
    for i in range(5):
        board.place(10 - i, 7 + i)
    assert board.get_winner() == Board.BLACK

def test_is_full():
    board = Board()
    assert board.is_full() == False

def test_is_game_over_win():
    board = Board()
    for i in range(5):
        board.place(i, 7)
    assert board.is_game_over() == True

def test_is_game_over_draw():
    board = Board()
    # Fill the board (225 positions)
    for y in range(15):
        for x in range(15):
            board.place(x, y)
    assert board.is_game_over() == True

def test_get_valid_moves():
    board = Board()
    moves = board.get_valid_moves()
    assert len(moves) == 225
    board.place(7, 7)
    moves = board.get_valid_moves()
    assert len(moves) == 224

def test_reset():
    board = Board()
    board.place(7, 7)
    board.reset()
    assert board.grid[7][7] == Board.EMPTY
    assert board.current_player == Board.BLACK
    assert board.move_count == 0
```

- [ ] **Step 3: 运行测试验证失败**

Run: `cd /Users/rr2017/code/wuziqi && python -m pytest tests/wuziqi_core/test_board.py -v`
Expected: FAIL - ModuleNotFoundError or test failures

- [ ] **Step 4: 实现 Board 类**

```python
# /Users/rr2017/code/wuziqi/wuziqi_core/board.py
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

    def get_valid_moves(self) -> list:
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

- [ ] **Step 5: 运行测试验证通过**

Run: `cd /Users/rr2017/code/wuziqi && python -m pytest tests/wuziqi_core/test_board.py -v`
Expected: PASS (15 tests)

- [ ] **Step 6: 提交代码**

```bash
cd /Users/rr2017/code/wuziqi && git init && git add -A && git commit -m "feat: add Board class with game logic"
```

---

### Task 2: 实现 Player 类

**Files:**
- Create: `/Users/rr2017/code/wuziqi/wuziqi_core/player.py`
- Test: `/Users/rr2017/code/wuziqi/tests/wuziqi_core/test_player.py`

- [ ] **Step 1: 编写 Player 类测试**

```python
# /Users/rr2017/code/wuziqi/tests/wuziqi_core/test_player.py
import pytest
import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')
from wuziqi_core.player import Player
from wuziqi_core.board import Board

def test_player_init():
    player = Player("TestPlayer", Board.BLACK)
    assert player.name == "TestPlayer"
    assert player.color == Board.BLACK

def test_player_default_name():
    player = Player(color=Board.WHITE)
    assert player.name == "Player"
```

- [ ] **Step 2: 实现 Player 类**

```python
# /Users/rr2017/code/wuziqi/wuziqi_core/player.py
class Player:
    def __init__(self, name: str = "Player", color: int = Board.BLACK):
        self.name = name
        self.color = color

# Add import at top
from wuziqi_core.board import Board
```

- [ ] **Step 3: 运行测试并提交**

Run: `cd /Users/rr2017/code/wuziqi && python -m pytest tests/wuziqi_core/test_player.py -v`
Commit: `git add -A && git commit -m "feat: add Player class"`
```

---

### Task 3: 实现 Game 类

**Files:**
- Modify: `/Users/rr2017/code/wuziqi/wuziqi_core/game.py`
- Test: `/Users/rr2017/code/wuziqi/tests/wuziqi_core/test_game.py`

- [ ] **Step 1: 编写 Game 类测试**

```python
# /Users/rr2017/code/wuziqi/tests/wuziqi_core/test_game.py
import pytest
import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')
from wuziqi_core.game import Game
from wuziqi_core.board import Board

def test_game_init():
    game = Game()
    assert game.board is not None
    assert game.players[Board.BLACK] is not None
    assert game.players[Board.WHITE] is not None

def test_game_current_player():
    game = Game()
    assert game.get_current_player() == Board.BLACK

def test_game_make_move():
    game = Game()
    result = game.make_move(7, 7)
    assert result == True
    assert game.board.grid[7][7] == Board.BLACK

def test_game_make_move_invalid():
    game = Game()
    result = game.make_move(-1, 0)
    assert result == False

def test_game_is_over():
    game = Game()
    # Win by placing 5 in a row
    for i in range(5):
        game.make_move(i, 7)
    assert game.is_over() == True

def test_game_get_winner():
    game = Game()
    for i in range(5):
        game.make_move(i, 7)
    assert game.get_winner() == Board.BLACK
```

- [ ] **Step 2: 实现 Game 类**

```python
# /Users/rr2017/code/wuziqi/wuziqi_core/game.py
from wuziqi_core.board import Board
from wuziqi_core.player import Player

class Game:
    def __init__(self, player_black: Player = None, player_white: Player = None):
        self.board = Board()
        self.players = {
            Board.BLACK: player_black or Player("Black", Board.BLACK),
            Board.WHITE: player_white or Player("White", Board.WHITE),
        }

    def get_current_player(self) -> Player:
        return self.players[self.board.current_player]

    def make_move(self, x: int, y: int) -> bool:
        """执行落子，返回是否成功"""
        return self.board.place(x, y)

    def is_over(self) -> bool:
        return self.board.is_game_over()

    def get_winner(self) -> int:
        return self.board.get_winner()

    def reset(self):
        self.board.reset()
```

- [ ] **Step 3: 运行测试并提交**

Run: `cd /Users/rr2017/code/wuziqi && python -m pytest tests/wuziqi_core/test_game.py -v`
Commit: `git add -A && git commit -m "feat: add Game class"`
```

---

### Task 4: 实现 AI 类 (5个难度)

**Files:**
- Create: `/Users/rr2017/code/wuziqi/wuziqi_core/ai.py`
- Test: `/Users/rr2017/code/wuziqi/tests/wuziqi_core/test_ai.py`

- [ ] **Step 1: 编写 AI 测试**

```python
# /Users/rr2017/code/wuziqi/tests/wuziqi_core/test_ai.py
import pytest
import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')
from wuziqi_core.ai import RandomAI, RuleAI, MinimaxAI, MCTSAgent
from wuziqi_core.board import Board
from wuziqi_core.game import Game

def test_random_ai_select():
    ai = RandomAI()
    board = Board()
    move = ai.select(board, Board.BLACK)
    assert board.is_valid_position(*move) == True
    # Should be empty position
    assert board.grid[move[1]][move[0]] == Board.EMPTY

def test_random_ai_consistency():
    ai = RandomAI()
    board = Board()
    # After placing at 7,7, AI should not select occupied
    board.place(7, 7)
    move = ai.select(board, Board.WHITE)
    assert move != (7, 7)

def test_rule_ai_select():
    ai = RuleAI()
    board = Board()
    move = ai.select(board, Board.BLACK)
    assert board.is_valid_position(*move) == True

def test_minimax_ai_select():
    ai = MinimaxAI(depth=2)
    board = Board()
    move = ai.select(board, Board.BLACK)
    assert board.is_valid_position(*move) == True

def test_mcts_ai_select():
    ai = MCTSAgent(simulations=100)
    board = Board()
    move = ai.select(board, Board.BLACK)
    assert board.is_valid_position(*move) == True

def test_ai_level_mapping():
    from wuziqi_core.ai import create_ai
    ai1 = create_ai(1)  # Random
    ai2 = create_ai(2)  # Rule
    ai3 = create_ai(3)  # Minimax
    ai4 = create_ai(4)  # MCTS
    assert ai1 is not None
    assert ai2 is not None
    assert ai3 is not None
    assert ai4 is not None
```

- [ ] **Step 2: 实现 AI 类**

```python
# /Users/rr2017/code/wuziqi/wuziqi_core/ai.py
import random
from typing import Tuple, Optional
from wuziqi_core.board import Board

class RandomAI:
    """Lv1: 随机落子 + 基础防守"""
    def select(self, board: Board, color: int) -> Tuple[int, int]:
        # 先检查是否有必防的点（四连被堵）
        for y in range(board.SIZE):
            for x in range(board.SIZE):
                if board.grid[y][x] != Board.EMPTY:
                    continue
                # 模拟落子后检查是否形成四连
                board.grid[y][x] = color
                if self._check_four_in_row(board, x, y, color):
                    board.grid[y][x] = Board.EMPTY
                    return (x, y)
                board.grid[y][x] = Board.EMPTY

        # 没有必防的点，随机落子
        moves = board.get_valid_moves()
        return random.choice(moves) if moves else (7, 7)

    def _check_four_in_row(self, board: Board, x: int, y: int, color: int) -> bool:
        # 检查4个方向是否有四连
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 正方向
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if not board.is_valid_position(nx, ny) or board.grid[ny][nx] != color:
                    break
                count += 1
            # 反方向
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if not board.is_valid_position(nx, ny) or board.grid[ny][nx] != color:
                    break
                count += 1
            if count >= 4:
                return True
        return False


class RuleAI:
    """Lv2: 评估函数 - 进攻/防守权重"""
    def select(self, board: Board, color: int) -> Tuple[int, int]:
        opponent = Board.WHITE if color == Board.BLACK else Board.BLACK

        best_score = float('-inf')
        best_move = None

        for x, y in board.get_valid_moves():
            attack_score = self._evaluate_position(board, x, y, color)
            defense_score = self._evaluate_position(board, x, y, opponent)
            total_score = attack_score + defense_score * 1.1  # 防守优先

            if total_score > best_score:
                best_score = total_score
                best_move = (x, y)

        return best_move or (7, 7)

    def _evaluate_position(self, board: Board, x: int, y: int, color: int) -> float:
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            open_ends = 0

            # 正方向
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if not board.is_valid_position(nx, ny):
                    break
                if board.grid[ny][nx] == color:
                    count += 1
                elif board.grid[ny][nx] == Board.EMPTY:
                    open_ends += 1
                    break
                else:
                    break

            # 反方向
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if not board.is_valid_position(nx, ny):
                    break
                if board.grid[ny][nx] == color:
                    count += 1
                elif board.grid[ny][nx] == Board.EMPTY:
                    open_ends += 1
                    break
                else:
                    break

            # 计分
            if count >= 5:
                score += 100000
            elif count == 4 and open_ends == 2:
                score += 10000
            elif count == 4 and open_ends == 1:
                score += 1000
            elif count == 3 and open_ends == 2:
                score += 1000
            elif count == 3 and open_ends == 1:
                score += 100
            elif count == 2 and open_ends == 2:
                score += 100

        return score


class MinimaxAI:
    """Lv3: Minimax + Alpha-Beta剪枝"""
    def __init__(self, depth: int = 3):
        self.depth = depth

    def select(self, board: Board, color: int) -> Tuple[int, int]:
        best_score = float('-inf')
        best_move = None

        moves = board.get_valid_moves()
        # 限制搜索节点
        if len(moves) > 30:
            moves = moves[:30]

        for x, y in moves:
            board.grid[y][x] = color
            score = self._minimax(board, self.depth - 1, float('-inf'), float('inf'), False, color)
            board.grid[y][x] = Board.EMPTY

            if score > best_score:
                best_score = score
                best_move = (x, y)

        return best_move or (7, 7)

    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, is_maximizing: bool, original_color: int) -> float:
        opponent = Board.WHITE if original_color == Board.BLACK else Board.BLACK
        current_color = opponent if is_maximizing else original_color

        winner = board.get_winner()
        if winner == original_color:
            return 100000
        if winner == opponent:
            return -100000
        if board.is_full() or depth == 0:
            return self._evaluate_board(board, original_color)

        moves = board.get_valid_moves()
        if len(moves) > 25:
            moves = moves[:25]

        if is_maximizing:
            max_eval = float('-inf')
            for x, y in moves:
                board.grid[y][x] = current_color
                eval = self._minimax(board, depth - 1, alpha, beta, False, original_color)
                board.grid[y][x] = Board.EMPTY
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for x, y in moves:
                board.grid[y][x] = current_color
                eval = self._minimax(board, depth - 1, alpha, beta, True, original_color)
                board.grid[y][x] = Board.EMPTY
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_board(self, board: Board, color: int) -> float:
        rule_ai = RuleAI()
        score = 0
        for y in range(board.SIZE):
            for x in range(board.SIZE):
                if board.grid[y][x] == color:
                    score += rule_ai._evaluate_position(board, x, y, color)
                elif board.grid[y][x] != Board.EMPTY:
                    score -= rule_ai._evaluate_position(board, x, y, Board.WHITE if color == Board.BLACK else Board.BLACK)
        return score


class MCTSAgent:
    """Lv4: MCTS蒙特卡洛树搜索"""
    def __init__(self, simulations: int = 1000):
        self.simulations = simulations

    def select(self, board: Board, color: int) -> Tuple[int, int]:
        # 简化实现：使用随机模拟
        best_move = None
        best_wins = 0

        moves = board.get_valid_moves()
        # 只考虑靠近已有棋子的位置
        moves = self._filter_nearby_moves(board, moves)

        for x, y in moves[:15]:  # 限制候选数
            wins = 0
            for _ in range(self.simulations // len(moves[:15])):
                board.grid[y][x] = color
                result = self._simulate(board, color)
                if result == color:
                    wins += 1
                board.grid[y][x] = Board.EMPTY

            if wins > best_wins:
                best_wins = wins
                best_move = (x, y)

        return best_move or random.choice(moves)

    def _filter_nearby_moves(self, board: Board, moves: list) -> list:
        # 只返回靠近已有棋子的位置
        nearby = []
        for x, y in moves:
            has_nearby = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if board.is_valid_position(nx, ny) and board.grid[ny][nx] != Board.EMPTY:
                        has_nearby = True
                        break
                if has_nearby:
                    break
            if has_nearby or board.move_count < 2:  # 前两步全局随机
                nearby.append((x, y))
        return nearby if nearby else moves[:30]

    def _simulate(self, board: Board, original_color: int) -> int:
        current = Board.WHITE if original_color == Board.BLACK else Board.BLACK
        while not board.is_game_over():
            moves = board.get_valid_moves()
            if not moves:
                break
            x, y = random.choice(moves)
            board.grid[y][x] = current
            current = Board.WHITE if current == Board.BLACK else Board.BLACK
        return board.get_winner()


def create_ai(level: int):
    """工厂函数：创建指定难度的AI"""
    if level == 1:
        return RandomAI()
    elif level == 2:
        return RuleAI()
    elif level == 3:
        return MinimaxAI(depth=3)
    elif level == 4:
        return MCTSAgent(simulations=1000)
    else:
        return None  # Lv5 需要PPO模型
```

- [ ] **Step 3: 运行测试并提交**

Run: `cd /Users/rr2017/code/wuziqi && python -m pytest tests/wuziqi_core/test_ai.py -v`
Commit: `git add -A && git commit -m "feat: add AI classes (Random, Rule, Minimax, MCTS)"`
```

---

## Chunk 2: Gymnasium RL环境 (wuziqi_gym)

### Task 5: 实现 Gymnasium 环境

**Files:**
- Create: `/Users/rr2017/code/wuziqi/wuziqi_gym/__init__.py`
- Create: `/Users/rr2017/code/wuziqi/wuziqi_gym/env.py`
- Test: `/Users/rr2017/code/wuziqi/tests/wuziqi_gym/__init__.py`
- Test: `/Users/rr2017/code/wuziqi/tests/wuziqi_gym/test_env.py`

- [ ] **Step 1: 创建测试**

```python
# /Users/rr2017/code/wuziqi/tests/wuziqi_gym/test_env.py
import pytest
import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')
import numpy as np
from wuziqi_gym.env import WuziqiEnv

def test_env_init():
    env = WuziqiEnv()
    assert env.board is not None

def test_env_reset():
    env = WuziqiEnv()
    obs, info = env.reset()
    assert obs.shape == (3, 15, 15)
    assert obs.dtype == np.float32

def test_env_step():
    env = WuziqiEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(112)  # center position (7,7)
    assert obs.shape == (3, 15, 15)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

def test_env_illegal_action():
    env = WuziqiEnv()
    env.reset()
    # First action at 7,7
    env.step(112)
    # Try illegal action at same position
    obs, reward, terminated, truncated, info = env.step(112)
    assert reward < 0  # Negative reward for illegal move

def test_env_legal_actions_mask():
    env = WuziqiEnv()
    env.reset()
    mask = env.legal_actions_mask()
    assert mask.shape == (225,)
    assert mask.sum() == 225  # All positions valid initially

def test_env_win():
    env = WuziqiEnv()
    obs, info = env.reset()
    # Black plays 5 in a row at row 7
    for i in range(5):
        action = i * 15 + 7  # (i, 7)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            assert reward == 1.0
            break
```

- [ ] **Step 2: 实现 WuziqiEnv 类**

```python
# /Users/rr2017/code/wuziqi/wuziqi_gym/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from wuziqi_core.board import Board


class WuziqiEnv(gym.Env):
    """五子棋Gymnasium环境"""

    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.board = Board()

        # 观察空间: 3通道 x 15 x 15
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 15, 15), dtype=np.float32
        )

        # 动作空间: 离散空间 0-224
        self.action_space = spaces.Discrete(225)

        self._current_player = Board.BLACK

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = Board()
        self._current_player = Board.BLACK
        return self._get_observation(), {}

    def step(self, action: int):
        x, y = action % 15, action // 15

        # 检查非法动作
        if not self.board.is_valid_position(x, y) or self.board.grid[y][x] != Board.EMPTY:
            # 非法动作返回负奖励
            return self._get_observation(), -0.1, False, False, {'illegal': True}

        # 执行落子
        self.board.place(x, y)

        # 检查游戏结束
        terminated = self.board.is_game_over()

        # 计算奖励
        reward = 0.0
        if terminated:
            winner = self.board.get_winner()
            if winner == Board.BLACK and self._current_player == Board.BLACK:
                reward = 1.0
            elif winner == Board.WHITE and self._current_player == Board.WHITE:
                reward = 1.0
            elif winner != 0:
                reward = -1.0
            else:
                reward = 0.0  # 平局
        else:
            # 每步轻微惩罚
            reward = -0.001

        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self) -> np.ndarray:
        """生成3通道观察"""
        obs = np.zeros((3, 15, 15), dtype=np.float32)
        player = self._current_player
        opponent = Board.WHITE if player == Board.BLACK else Board.BLACK

        for y in range(15):
            for x in range(15):
                if self.board.grid[y][x] == player:
                    obs[0, y, x] = 1.0
                elif self.board.grid[y][x] == opponent:
                    obs[1, y, x] = 1.0
                else:
                    obs[2, y, x] = 1.0

        return obs

    def legal_actions(self) -> list:
        return [y * 15 + x for x, y in self.board.get_valid_moves()]

    def legal_actions_mask(self) -> np.ndarray:
        mask = np.zeros(225, dtype=np.float32)
        for x, y in self.board.get_valid_moves():
            mask[y * 15 + x] = 1.0
        return mask

    def render(self):
        if self.render_mode == 'human':
            print(self.board.grid)

    def close(self):
        pass
```

- [ ] **Step 3: 更新环境以支持 self-play**

修改 env.py 中的 reset 和 step 方法，添加 player_color 参数：

```python
def reset(self, seed=None, options=None, player_color=None):
    super().reset(seed=seed)
    self.board = Board()
    self._player_color = player_color if player_color else Board.BLACK
    self._current_player = Board.BLACK
    return self._get_observation(), {}

def step(self, action: int):
    # ... 现有代码 ...
    # 更新当前玩家
    self._current_player = Board.WHITE if self._current_player == Board.BLACK else Board.BLACK
```

- [ ] **Step 4: 运行测试并提交**

Run: `cd /Users/rr2017/code/wuziqi && python -m pytest tests/wuziqi_gym/test_env.py -v`
Commit: `git add -A && git commit -m "feat: add WuziqiEnv Gymnasium environment"`
```

---

## Chunk 3: PyGame UI (wuziqi_ui)

### Task 6: 实现 PyGame UI

**Files:**
- Create: `/Users/rr2017/code/wuziqi/wuziqi_ui/__init__.py`
- Create: `/Users/rr2017/code/wuziqi/wuziqi_ui/main.py`
- Create: `/Users/rr2017/code/wuziqi/wuziqi_ui/renderer.py`
- Create: `/Users/rr2017/code/wuziqi/wuziqi_ui/controller.py`

- [ ] **Step 1: 创建基础 UI 结构**

```python
# /Users/rr2017/code/wuziqi/wuziqi_ui/__init__.py
from wuziqi_ui.main import main

__all__ = ['main']
```

```python
# /Users/rr2017/code/wuziqi/wuziqi_ui/renderer.py
import pygame
import sys
from wuziqi_core.board import Board


class Renderer:
    """五子棋渲染器"""

    CELL_SIZE = 40
    BOARD_SIZE = 15 * CELL_SIZE
    WINDOW_SIZE = BOARD_SIZE + 200  # 额外空间给UI

    BLACK_COLOR = (0, 0, 0)
    WHITE_COLOR = (255, 255, 255)
    BOARD_COLOR = (222, 184, 135)  # 木纹色
    LINE_COLOR = (0, 0, 0)
    HIGHLIGHT_COLOR = (255, 0, 0)
    BG_COLOR = (240, 240, 240)
    BUTTON_COLOR = (100, 150, 200)
    BUTTON_HOVER = (120, 170, 220)
    TEXT_COLOR = (50, 50, 50)

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.BOARD_SIZE + 80))
        pygame.display.set_caption("五子棋 v1.0")
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        self.buttons = [
            {'rect': pygame.Rect(20, self.BOARD_SIZE + 20, 80, 40), 'text': '新游戏'},
            {'rect': pygame.Rect(120, self.BOARD_SIZE + 20, 80, 40), 'text': '悔棋'},
            {'rect': pygame.Rect(220, self.BOARD_SIZE + 20, 80, 40), 'text': '认输'},
        ]
        self.difficulty_buttons = [
            {'rect': pygame.Rect(320, self.BOARD_SIZE + 20, 60, 40), 'text': '入门'},
            {'rect': pygame.Rect(390, self.BOARD_SIZE + 20, 60, 40), 'text': '简单'},
            {'rect': pygame.Rect(460, self.BOARD_SIZE + 20, 60, 40), 'text': '中等'},
            {'rect': pygame.Rect(530, self.BOARD_SIZE + 20, 60, 40), 'text': '困难'},
        ]

    def draw_board(self, board: Board, last_move=None):
        self.screen.fill(self.BG_COLOR)

        # 绘制棋盘背景
        pygame.draw.rect(self.screen, self.BOARD_COLOR,
                        (50, 30, self.BOARD_SIZE, self.BOARD_SIZE))

        # 绘制网格线
        for i in range(15):
            # 横线
            pygame.draw.line(self.screen, self.LINE_COLOR,
                           (50, 30 + i * self.CELL_SIZE),
                           (50 + self.BOARD_SIZE, 30 + i * self.CELL_SIZE), 1)
            # 竖线
            pygame.draw.line(self.screen, self.LINE_COLOR,
                           (50 + i * self.CELL_SIZE, 30),
                           (50 + i * self.CELL_SIZE, 30 + self.BOARD_SIZE), 1)

        # 绘制天元和星位
        star_points = [(3, 3), (11, 3), (7, 7), (3, 11), (11, 11)]
        for x, y in star_points:
            pygame.draw.circle(self.screen, self.LINE_COLOR,
                             (50 + x * self.CELL_SIZE, 30 + y * self.CELL_SIZE), 4)

        # 绘制棋子
        for y in range(15):
            for x in range(15):
                if board.grid[y][x] != Board.EMPTY:
                    center = (50 + x * self.CELL_SIZE, 30 + y * self.CELL_SIZE)
                    color = self.BLACK_COLOR if board.grid[y][x] == Board.BLACK else self.WHITE_COLOR
                    pygame.draw.circle(self.screen, color, center, 16)

        # 高亮最后落子位置
        if last_move:
            x, y = last_move
            pygame.draw.circle(self.screen, self.HIGHLIGHT_COLOR,
                             (50 + x * self.CELL_SIZE, 30 + y * self.CELL_SIZE), 20, 2)

        # 绘制按钮
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons + self.difficulty_buttons:
            color = self.BUTTON_HOVER if button['rect'].collidepoint(mouse_pos) else self.BUTTON_COLOR
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=5)
            text = self.font.render(button['text'], True, self.WHITE_COLOR)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)

        # 绘制玩家信息
        info_text = self.font.render("黑方: 玩家(你)  vs  白方: AI", True, self.TEXT_COLOR)
        self.screen.blit(info_text, (50, self.BOARD_SIZE - 20))

        # 绘制难度标签
        diff_text = self.small_font.render("难度:", True, self.TEXT_COLOR)
        self.screen.blit(diff_text, (320, self.BOARD_SIZE - 20))

        pygame.display.flip()

    def draw_message(self, message: str):
        text = self.font.render(message, True, (255, 0, 0))
        text_rect = text.get_rect(center=(self.WINDOW_SIZE // 2, self.BOARD_SIZE // 2))
        pygame.draw.rect(self.screen, self.WHITE_COLOR, text_rect.inflate(20, 10))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(2000)

    def close(self):
        pygame.quit()
```

```python
# /Users/rr2017/code/wuziqi/wuziqi_ui/controller.py
import pygame
from wuziqi_core.game import Game
from wuziqi_core.ai import create_ai


class Controller:
    """游戏控制器"""

    def __init__(self, game: Game, ai, renderer):
        self.game = game
        self.ai = ai
        self.renderer = renderer
        self.last_move = None
        self.difficulty = 3  # 默认中等
        self.game_over = False

    def handle_click(self, pos: tuple) -> bool:
        """处理鼠标点击，返回是否需要重绘"""
        x, y = pos[0], pos[1]

        # 检查按钮点击
        for button in self.renderer.buttons:
            if button['rect'].collidepoint(pos):
                if button['text'] == '新游戏':
                    self.game.reset()
                    self.last_move = None
                    self.game_over = False
                    return True
                elif button['text'] == '悔棋':
                    # 简化：重新开始
                    self.game.reset()
                    self.last_move = None
                    self.game_over = False
                    return True
                elif button['text'] == '认输':
                    self.game_over = True
                    return True

        # 检查难度按钮
        difficulties = {'入门': 1, '简单': 2, '中等': 3, '困难': 4}
        for button in self.renderer.difficulty_buttons:
            if button['rect'].collidepoint(pos):
                if button['text'] in difficulties:
                    self.difficulty = difficulties[button['text']]
                    self.ai = create_ai(self.difficulty)
                    return True

        # 检查棋盘点击
        if 50 <= x <= 50 + self.renderer.BOARD_SIZE and \
           30 <= y <= 30 + self.renderer.BOARD_SIZE:
            board_x = (x - 50) // self.renderer.CELL_SIZE
            board_y = (y - 30) // self.renderer.CELL_SIZE

            if self.game.board.is_valid_position(board_x, board_y) and \
               self.game.board.grid[board_y][board_x] == Board.EMPTY:
                return board_x, board_y

        return None

    def ai_move(self):
        """AI落子"""
        if self.game_over or self.game.board.is_game_over():
            return

        # 创建当前玩家颜色的AI
        ai = create_ai(self.difficulty)
        if ai:
            move = ai.select(self.game.board, self.game.board.current_player)
            self.game.make_move(*move)
            self.last_move = move

    def update(self):
        """更新游戏状态"""
        # AI移动
        if not self.game_over and not self.game.board.is_game_over():
            if self.game.board.current_player != Board.BLACK:  # AI是白方
                self.ai_move()

        # 检查游戏结束
        if self.game.board.is_game_over():
            self.game_over = True
            winner = self.game.board.get_winner()
            if winner == Board.BLACK:
                self.renderer.draw_message("恭喜！你赢了！")
            elif winner == Board.WHITE:
                self.renderer.draw_message("AI获胜！")
            else:
                self.renderer.draw_message("平局！")
```

```python
# /Users/rr2017/code/wuziqi/wuziqi_ui/main.py
import pygame
from wuziqi_core.game import Game
from wuziqi_core.board import Board
from wuziqi_core.ai import create_ai
from wuziqi_ui.renderer import Renderer
from wuziqi_ui.controller import Controller


def main():
    """主函数"""
    game = Game()
    renderer = Renderer()
    ai = create_ai(3)  # 默认中等难度
    controller = Controller(game, ai, renderer)

    running = True
    while running:
        # 绘制
        controller.renderer.draw_board(game.board, controller.last_move)

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                result = controller.handle_click(event.pos)
                if isinstance(result, tuple):
                    x, y = result
                    if game.make_move(x, y):
                        controller.last_move = (x, y)

        # 更新
        controller.update()

        if controller.game_over:
            pygame.time.wait(1000)
            game.reset()
            controller.last_move = None
            controller.game_over = False

    renderer.close()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 修复导入问题**

在 controller.py 中添加必要的导入:

```python
# 在文件顶部添加
from wuziqi_core.board import Board
```

- [ ] **Step 3: 提交代码**

Commit: `git add -A && git commit -m "feat: add PyGame UI with board rendering and AI gameplay"`
```

---

## Chunk 4: RL训练与评测 (trainer)

### Task 7: 实现训练脚本

**Files:**
- Create: `/Users/rr2017/code/wuziqi/trainer/__init__.py`
- Create: `/Users/rr2017/code/wuziqi/trainer/config.py`
- Create: `/Users/rr2017/code/wuziqi/trainer/train.py`
- Create: `/Users/rr2017/code/wuziqi/trainer/evaluate.py`

- [ ] **Step 1: 创建配置类**

```python
# /Users/rr2017/code/wuziqi/trainer/config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 100_000
    max_steps: int = 200
    env_make_kwargs: Optional[dict] = None
    model_kwargs: Optional[dict] = None
    save_freq: int = 10000
    eval_freq: int = 50000
    save_path: str = "./models"
    log_dir: str = "./logs"


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

- [ ] **Step 2: 创建训练脚本**

```python
# /Users/rr2017/code/wuziqi/trainer/train.py
import os
import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')

from stable_baselines3 import PPO, A2C
from wuziqi_gym.env import WuziqiEnv
from trainer.config import TrainConfig


def train(config: TrainConfig):
    """训练主函数"""
    # 创建环境
    env = WuziqiEnv()

    # 创建模型
    model_kwargs = config.model_kwargs.copy() if config.model_kwargs else {}
    policy = model_kwargs.pop('policy', 'MlpPolicy')

    if config.algorithm == "PPO":
        model = PPO(policy, env, verbose=1, **model_kwargs)
    elif config.algorithm == "A2C":
        model = A2C(policy, env, verbose=1, **model_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # 训练
    model.learn(total_timesteps=config.total_timesteps)

    # 保存
    os.makedirs(config.save_path, exist_ok=True)
    model.save(os.path.join(config.save_path, f"{config.algorithm}_wuziqi"))

    return model


if __name__ == '__main__':
    config = TrainConfig(
        algorithm="PPO",
        total_timesteps=100000,
    )
    train(config)
```

- [ ] **Step 3: 创建评测脚本**

```python
# /Users/rr2017/code/wuziqi/trainer/evaluate.py
import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')

import argparse
import numpy as np
from stable_baselines3 import PPO
from wuziqi_core.board import Board
from wuziqi_core.ai import RandomAI, RuleAI, MCTSAgent, create_ai
from wuziqi_gym.env import WuziqiEnv


def play_game(model, opponent, first_player=0):
    """执行一局游戏，返回胜者"""
    env = WuziqiEnv()
    obs, _ = env.reset()

    current_player = first_player
    max_steps = 200
    steps = 0

    while not env.board.is_game_over() and steps < max_steps:
        if current_player == 1:  # AI (model)
            action, _ = model.predict(obs, deterministic=True)
        else:  # 对手
            if opponent == "random":
                ai = RandomAI()
                move = ai.select(env.board, env.board.current_player)
                action = move[0] + move[1] * 15
            elif opponent == "rule":
                ai = RuleAI()
                move = ai.select(env.board, env.board.current_player)
                action = move[0] + move[1] * 15
            elif opponent == "mcts":
                ai = MCTSAgent(simulations=500)
                move = ai.select(env.board, env.board.current_player)
                action = move[0] + move[1] * 15
            else:
                # 随机
                moves = env.legal_actions()
                action = np.random.choice(moves)

        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated:
            return 1 if reward > 0 else -1 if reward < 0 else 0

        current_player = 1 - current_player
        steps += 1

    return 0  # 平局


def evaluate(model_path, opponent, num_games=100):
    """评测模型

    Args:
        model_path: 模型路径
        opponent: 对手类型 ("random", "rule", "mcts")
        num_games: 对局数量

    Returns:
        dict: 评测结果
    """
    # 加载模型
    if model_path:
        model = PPO.load(model_path)
    else:
        print("Warning: No model loaded, using random baseline")
        model = None

    wins = losses = draws = 0

    for i in range(num_games):
        first_player = i % 2
        result = play_game(model, opponent, first_player)

        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            print(f"Played {i + 1}/{num_games} games")

    return {
        "win_rate": wins / num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Wuziqi AI')
    parser.add_argument('--model', type=str, default='./models/ppo_wuziqi',
                       help='Path to trained model')
    parser.add_argument('--opponent', type=str, default='random',
                       choices=['random', 'rule', 'mcts'],
                       help='Opponent type')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games to play')

    args = parser.parse_args()

    result = evaluate(args.model, args.opponent, args.games)

    print(f"\n=== Evaluation Results ===")
    print(f"Opponent: {args.opponent}")
    print(f"Games: {args.games}")
    print(f"Wins: {result['wins']}")
    print(f"Losses: {result['losses']}")
    print(f"Draws: {result['draws']}")
    print(f"Win Rate: {result['win_rate']:.2%}")
```

- [ ] **Step 4: 提交代码**

Commit: `git add -A && git commit -m "feat: add trainer and evaluation framework"`
```

---

## Chunk 5: 集成测试与依赖配置

### Task 8: 创建 requirements.txt 和最终测试

**Files:**
- Create: `/Users/rr2017/code/wuziqi/requirements.txt`
- Create: `/Users/rr2017/code/wuziqi/pyproject.toml` 或 `/Users/rr2017/code/wuziqi/setup.py`

- [ ] **Step 1: 创建依赖文件**

```bash
# /Users/rr2017/code/wuziqi/requirements.txt
gymnasium>=0.29.0
pygame>=2.5.0
stable-baselines3>=2.0.0
torch>=2.0.0
numpy>=1.24.0
pytest>=7.0.0
```

- [ ] **Step 2: 运行完整测试**

```bash
cd /Users/rr2017/code/wuziqi && pip install -r requirements.txt
python -m pytest tests/ -v
```

- [ ] **Step 3: 提交**

Commit: `git add -A && git commit -m "chore: add dependencies and requirements"`
```

---

## 使用说明

### 运行游戏 UI
```bash
cd /Users/rr2017/code/wuziqi
pip install -r requirements.txt
python -m wuziqi_ui.main
```

### 训练模型
```bash
python -m trainer.train
```

### 评测模型
```bash
# vs 随机AI
python -m trainer.evaluate --model models/ppo_wuziqi --opponent random --games 100

# vs 规则AI
python -m trainer.evaluate --model models/ppo_wuziqi --opponent rule --games 50

# vs MCTS
python -m trainer.evaluate --model models/ppo_wuziqi --opponent mcts --games 20
```

### 运行测试
```bash
pytest tests/ -v
```
