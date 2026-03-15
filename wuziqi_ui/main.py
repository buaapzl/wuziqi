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

    waiting_for_ai = False  # 标记是否在等待AI落子

    running = True
    while running:
        # 绘制
        controller.renderer.draw_board(game.board, controller.last_move)

        # 如果正在等待AI，不处理鼠标点击
        if not waiting_for_ai:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    result = controller.handle_click(event.pos)
                    if isinstance(result, tuple):
                        x, y = result
                        if game.make_move(x, y):
                            # 保存当前棋盘状态用于悔棋
                            controller.move_history.append([row[:] for row in game.board.grid])
                            controller.last_move = (x, y)

                            # 检查游戏是否结束
                            if game.board.is_game_over():
                                controller.game_over = True
                            else:
                                # 玩家成功落子，现在等待AI
                                waiting_for_ai = True

        # AI落子
        if waiting_for_ai and not controller.game_over:
            if game.board.current_player == Board.WHITE:
                controller.ai_move()
                waiting_for_ai = False

                # 检查游戏是否结束
                if game.board.is_game_over():
                    controller.game_over = True

        # 处理游戏结束
        if controller.game_over:
            winner = game.board.get_winner()
            if winner == Board.BLACK:
                renderer.draw_message("恭喜！你赢了！")
            elif winner == Board.WHITE:
                renderer.draw_message("AI获胜！")
            else:
                renderer.draw_message("平局！")

            # 重置游戏
            game.reset()
            controller.move_history = []
            controller.last_move = None
            controller.game_over = False
            waiting_for_ai = False

    renderer.close()


if __name__ == '__main__':
    main()
