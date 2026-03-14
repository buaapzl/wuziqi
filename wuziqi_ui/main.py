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
