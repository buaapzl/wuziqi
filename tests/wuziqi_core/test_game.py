import pytest
import sys
import os
import importlib.util

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

# Import board module using importlib to ensure it works with pytest
_board_file = os.path.join(_project_root, 'wuziqi_core', 'board.py')
spec = importlib.util.spec_from_file_location("board", _board_file)
board_module = importlib.util.module_from_spec(spec)
sys.modules['wuziqi_core.board'] = board_module
spec.loader.exec_module(board_module)
Board = board_module.Board

# Import player module
_player_file = os.path.join(_project_root, 'wuziqi_core', 'player.py')
spec = importlib.util.spec_from_file_location("player", _player_file)
player_module = importlib.util.module_from_spec(spec)
sys.modules['wuziqi_core.player'] = player_module
spec.loader.exec_module(player_module)
Player = player_module.Player

# Import game module
_game_file = os.path.join(_project_root, 'wuziqi_core', 'game.py')
spec = importlib.util.spec_from_file_location("game", _game_file)
game_module = importlib.util.module_from_spec(spec)
sys.modules['wuziqi_core.game'] = game_module
spec.loader.exec_module(game_module)
Game = game_module.Game


def test_game_init():
    game = Game()
    assert game.board is not None
    assert game.players[Board.BLACK] is not None
    assert game.players[Board.WHITE] is not None

def test_game_current_player():
    game = Game()
    assert game.get_current_player().color == Board.BLACK

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
    # Skip WHITE's turns
    for i in range(5):
        game.make_move(i, 7)  # BLACK
        if i < 4:
            game.make_move(0, 14 - i)  # WHITE (skip turn)
    assert game.is_over() == True

def test_game_get_winner():
    game = Game()
    # Skip WHITE's turns
    for i in range(5):
        game.make_move(i, 7)  # BLACK
        if i < 4:
            game.make_move(0, 14 - i)  # WHITE (skip turn)
    assert game.get_winner() == Board.BLACK
