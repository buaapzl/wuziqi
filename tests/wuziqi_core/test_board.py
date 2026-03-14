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

def test_board_init():
    board = Board()
    assert board.SIZE == 15
    assert board.EMPTY == 0
    assert board.BLACK == 1
    assert board.WHITE == 2
    assert board.current_player == Board.BLACK
    assert board.move_count == 0

def test_is_valid_position():
    board = Board()
    assert board.is_valid_position(0, 0) == True
    assert board.is_valid_position(14, 14) == True
    assert board.is_valid_position(-1, 0) == False
    assert board.is_valid_position(0, 15) == False

def test_place_stone():
    board = Board()
    assert board.place(7, 7) == True
    assert board.grid[7][7] == Board.BLACK
    assert board.move_count == 1
    assert board.current_player == Board.WHITE

def test_place_invalid_position():
    board = Board()
    assert board.place(-1, 0) == False
    assert board.place(0, 15) == False

def test_place_occupied():
    board = Board()
    board.place(7, 7)
    assert board.place(7, 7) == False

def test_get_winner_none():
    board = Board()
    assert board.get_winner() == 0

def test_get_winner_horizontal():
    board = Board()
    # Black places 5 in a row horizontally
    # Skip WHITE's turns by making WHITE place elsewhere
    for i in range(5):
        board.place(i, 7)  # BLACK
        if i < 4:
            board.place(0, 14 - i)  # WHITE (skip turn)

    assert board.get_winner() == Board.BLACK

def test_get_winner_vertical():
    board = Board()
    # Black places 5 in a row vertically
    # Skip WHITE's turns by making WHITE place elsewhere
    for i in range(5):
        board.place(7, i)  # BLACK
        if i < 4:
            board.place(i + 1, 0)  # WHITE (skip turn)

    assert board.get_winner() == Board.BLACK

def test_get_winner_diagonal():
    board = Board()
    # Black places 5 in a row diagonally
    # Skip WHITE's turns by making WHITE place elsewhere
    for i in range(5):
        board.place(7 + i, 7 + i)  # BLACK
        if i < 4:
            board.place(0, i)  # WHITE (skip turn)

    assert board.get_winner() == Board.BLACK

def test_get_winner_anti_diagonal():
    board = Board()
    # Black places 5 in a row anti-diagonally
    # Skip WHITE's turns by making WHITE place elsewhere
    for i in range(5):
        board.place(10 - i, 7 + i)  # BLACK
        if i < 4:
            board.place(14 - i, 0)  # WHITE (skip turn)

    assert board.get_winner() == Board.BLACK

def test_is_full():
    board = Board()
    assert board.is_full() == False

def test_is_game_over_win():
    board = Board()
    # Skip WHITE's turns
    for i in range(5):
        board.place(i, 7)  # BLACK
        if i < 4:
            board.place(0, 14 - i)  # WHITE (skip turn)

    assert board.is_game_over() == True

def test_is_game_over_draw():
    board = Board()
    # Fill the board (225 positions)
    for y in range(15):
        for x in range(15):
            board.place(x, y)
    assert board.is_game_over() == True

def test_get_valid_moves():
    board = Board()
    moves = board.get_valid_moves()
    assert len(moves) == 225
    board.place(7, 7)
    moves = board.get_valid_moves()
    assert len(moves) == 224

def test_reset():
    board = Board()
    board.place(7, 7)
    board.reset()
    assert board.grid[7][7] == Board.EMPTY
    assert board.current_player == Board.BLACK
    assert board.move_count == 0
