from wuziqi_core.board import Board


class Player:
    def __init__(self, name: str = "Player", color: int = Board.BLACK):
        self.name = name
        self.color = color
