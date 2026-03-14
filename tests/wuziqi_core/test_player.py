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


def test_player_init():
    player = Player("TestPlayer", Board.BLACK)
    assert player.name == "TestPlayer"
    assert player.color == Board.BLACK


def test_player_default_name():
    player = Player(color=Board.WHITE)
    assert player.name == "Player"
