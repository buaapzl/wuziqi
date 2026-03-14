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
