import pygame
from wuziqi_core.game import Game
from wuziqi_core.board import Board
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
        self.move_history = []  # 记录每一步的棋盘状态用于悔棋
        self.player_just_moved = False  # 标记玩家是否刚落子

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
        difficulties = {'入门': 1, '简单': 2, '中等': 3, '困难': 4}
        for button in self.renderer.difficulty_buttons:
            if button['rect'].collidepoint(pos):
                if button['text'] in difficulties:
                    self.difficulty = difficulties[button['text']]
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

        # 创建当前玩家颜色的AI
        ai = create_ai(self.difficulty)
        if ai:
            move = ai.select(self.game.board, self.game.board.current_player)
            # 保存当前棋盘状态用于悔棋
            self.move_history.append([row[:] for row in self.game.board.grid])
            self.game.make_move(*move)
            self.last_move = move

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
