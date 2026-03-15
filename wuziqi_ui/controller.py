import pygame
import os
from wuziqi_core.game import Game
from wuziqi_core.board import Board
from wuziqi_core.ai import create_ai

# 尝试导入PPO模型
try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False


class Controller:
    """游戏控制器"""

    def __init__(self, game: Game, ai, renderer):
        self.game = game
        self.ai = ai
        self.renderer = renderer
        self.last_move = None
        self.difficulty = 3  # 默认中等
        self.game_over = False
        self.move_history = []  # 记录每一步的棋盘状态用于悔棋
        self.player_just_moved = False  # 标记玩家是否刚落子
        self.ppo_model = None  # PPO模型
        self._load_ppo_model()

    def _load_ppo_model(self):
        """加载PPO模型"""
        if not PPO_AVAILABLE:
            return
        # 查找模型文件
        model_paths = [
            "./models/ppo_wuziqi.zip",
            "./models/PPO_wuziqi.zip",
        ]
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.ppo_model = PPO.load(path)
                    print(f"已加载PPO模型: {path}")
                    break
                except Exception as e:
                    print(f"加载模型失败 {path}: {e}")

    def handle_click(self, pos: tuple) -> bool:
        """处理鼠标点击，返回是否需要重绘"""
        x, y = pos[0], pos[1]

        # 检查按钮点击
        for button in self.renderer.buttons:
            if button['rect'].collidepoint(pos):
                if button['text'] == '新游戏':
                    self.game.reset()
                    self.move_history = []  # 清空悔棋历史
                    self.last_move = None
                    self.game_over = False
                    return True
                elif button['text'] == '悔棋':
                    # 实际悔棋：恢复到上一步状态
                    if len(self.move_history) >= 2:
                        # 移除最近两步（玩家和AI）
                        self.move_history.pop()
                        if self.move_history:
                            last_state = self.move_history[-1]
                            self.game.board.grid = [row[:] for row in last_state]
                            self.game.board.current_player = Board.BLACK
                            self.last_move = None
                            self.game_over = False
                    return True
                elif button['text'] == '认输':
                    self.game_over = True
                    return True

        # 检查难度按钮
        difficulties = {'入门': 1, '简单': 2, '中等': 3, '困难': 4, '大师': 5}
        for button in self.renderer.difficulty_buttons:
            if button['rect'].collidepoint(pos):
                if button['text'] in difficulties:
                    self.difficulty = difficulties[button['text']]
                    self.renderer.selected_difficulty = self.difficulty  # 更新选中状态
                    if self.difficulty < 5:
                        self.ai = create_ai(self.difficulty)
                    return True

        # 检查棋盘点击 - 使用更精确的落点检测
        board_left = 50
        board_top = 30
        board_right = board_left + self.renderer.BOARD_SIZE
        board_bottom = board_top + self.renderer.BOARD_SIZE

        if board_left <= x < board_right and board_top <= y < board_bottom:
            # 计算最近的交叉点
            board_x = int(round((x - board_left) / self.renderer.CELL_SIZE))
            board_y = int(round((y - board_top) / self.renderer.CELL_SIZE))

            # 确保在有效范围内
            board_x = max(0, min(14, board_x))
            board_y = max(0, min(14, board_y))

            if self.game.board.is_valid_position(board_x, board_y) and \
               self.game.board.grid[board_y][board_x] == Board.EMPTY:
                return board_x, board_y

        return None

    def ai_move(self):
        """AI落子"""
        if self.game_over or self.game.board.is_game_over():
            return

        move = None

        # 如果是大师难度，尝试使用PPO模型
        if self.difficulty == 5:
            if self.ppo_model is not None:
                # 使用PPO模型预测动作
                # 需要将棋盘状态转换为模型输入格式
                try:
                    obs = self._get_obs()
                    action, _ = self.ppo_model.predict(obs, deterministic=True)
                    # 将action转换为(x, y)
                    board_x = action % 15
                    board_y = action // 15
                    # 检查是否合法
                    if self.game.board.is_valid_position(board_x, board_y) and \
                       self.game.board.grid[board_y][board_x] == Board.EMPTY:
                        move = (board_x, board_y)
                except Exception as e:
                    print(f"PPO模型预测失败: {e}")

        # 如果没有使用PPO或PPO失败，使用规则AI
        if move is None:
            ai = create_ai(4)  # 使用MCTS作为后备
            if ai:
                move = ai.select(self.game.board, self.game.board.current_player)

        if move:
            # 保存当前棋盘状态用于悔棋
            self.move_history.append([row[:] for row in self.game.board.grid])
            self.game.make_move(*move)
            self.last_move = move

    def _get_obs(self):
        """获取PPO模型的观察输入"""
        import numpy as np
        obs = np.zeros((3, 15, 15), dtype=np.float32)
        player = self.game.board.current_player
        opponent = Board.WHITE if player == Board.BLACK else Board.BLACK

        for y in range(15):
            for x in range(15):
                if self.game.board.grid[y][x] == player:
                    obs[0, y, x] = 1.0
                elif self.game.board.grid[y][x] == opponent:
                    obs[1, y, x] = 1.0
                else:
                    obs[2, y, x] = 1.0
        return obs

    def update(self):
        """更新游戏状态"""
        # 如果玩家刚落子，先清除标记，让玩家看到自己的落子
        if self.player_just_moved:
            self.player_just_moved = False
            return

        # AI移动
        if not self.game_over and not self.game.board.is_game_over():
            if self.game.board.current_player != Board.BLACK:  # AI是白方
                self.ai_move()
