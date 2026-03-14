import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import importlib.util

# Load Board from wuziqi_core
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_board_file = os.path.join(_project_root, 'wuziqi_core', 'board.py')
spec = importlib.util.spec_from_file_location("board", _board_file)
board_module = importlib.util.module_from_spec(spec)
sys.modules['wuziqi_core.board'] = board_module
spec.loader.exec_module(board_module)
Board = board_module.Board


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
        self.np_random_ = np.random.default_rng(seed)
        self.board = Board()
        self._current_player = Board.BLACK
        return self._get_observation(), {}

    def step(self, action: int):
        # 检查游戏是否已结束
        if self.board.is_game_over():
            return self._get_observation(), 0, True, False, {'error': 'Game already over'}

        x, y = action % 15, action // 15

        # 检查非法动作
        if not self.board.is_valid_position(x, y) or self.board.grid[y][x] != Board.EMPTY:
            # 非法动作返回负奖励
            return self._get_observation(), -0.1, False, False, {'illegal': True}

        # 执行落子
        self.board.place(x, y)

        # 保存当前玩家（在切换之前）用于奖励计算
        current_player_before_switch = self._current_player

        # 更新当前玩家
        self._current_player = self.board.current_player

        # 检查游戏结束
        terminated = self.board.is_game_over()

        # 计算奖励
        reward = 0.0
        if terminated:
            winner = self.board.get_winner()
            if winner == Board.BLACK and current_player_before_switch == Board.BLACK:
                reward = 1.0
            elif winner == Board.WHITE and current_player_before_switch == Board.WHITE:
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
