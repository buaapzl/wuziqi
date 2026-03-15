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
                        # 保存当前棋盘状态用于悔棋
                        controller.move_history.append([row[:] for row in game.board.grid])
                        controller.last_move = (x, y)
                        controller.player_just_moved = True  # 标记玩家刚落子

                        # 检查玩家是否获胜
                        if game.board.is_game_over():
                            controller.game_over = True
                            winner = game.board.get_winner()
                            if winner == Board.BLACK:
                                renderer.draw_message("恭喜！你赢了！")
                            elif winner == Board.WHITE:
                                renderer.draw_message("AI获胜！")
                            else:
                                renderer.draw_message("平局！")

        # 更新（AI移动）
        if not controller.game_over:
            controller.update()

        if controller.game_over:
            pygame.time.wait(1000)
            game.reset()
            controller.move_history = []  # 清空悔棋历史
            controller.last_move = None
            controller.game_over = False

    renderer.close()


if __name__ == '__main__':
    main()
