import pygame
import sys
from wuziqi_core.board import Board


class Renderer:
    """五子棋渲染器"""

    CELL_SIZE = 40
    BOARD_SIZE = 15 * CELL_SIZE
    WINDOW_SIZE = BOARD_SIZE + 200  # 额外空间给UI
    WINDOW_HEIGHT = BOARD_SIZE + 80  # 底部空间给按钮

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
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_HEIGHT))
        pygame.display.set_caption("五子棋 v1.0")

        # 使用支持中文的字体 (macOS优先)
        import platform
        if platform.system() == "Darwin":
            # macOS 系统字体
            try:
                self.font = pygame.font.Font("/System/Library/Fonts/STHeiti Medium.ttc", 32)
                self.small_font = pygame.font.Font("/System/Library/Fonts/STHeiti Medium.ttc", 24)
            except:
                try:
                    self.font = pygame.font.Font("/System/Library/Fonts/Supplemental/Songti.ttc", 32)
                    self.small_font = pygame.font.Font("/System/Library/Fonts/Supplemental/Songti.ttc", 24)
                except:
                    self.font = pygame.font.Font(None, 32)
                    self.small_font = pygame.font.Font(None, 24)
        else:
            # Windows/Linux 系统字体
            try:
                self.font = pygame.font.Font("SimHei", 32)
                self.small_font = pygame.font.Font("SimHei", 24)
            except:
                try:
                    self.font = pygame.font.Font("Microsoft YaHei", 32)
                    self.small_font = pygame.font.Font("Microsoft YaHei", 24)
                except:
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
            {'rect': pygame.Rect(600, self.BOARD_SIZE + 20, 60, 40), 'text': '大师'},
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
