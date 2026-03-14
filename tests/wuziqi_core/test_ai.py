# /Users/rr2017/code/wuziqi/tests/wuziqi_core/test_ai.py
import pytest
import sys
import os
import importlib.util

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

# Import modules using importlib to ensure they work with pytest
_board_file = os.path.join(_project_root, 'wuziqi_core', 'board.py')
_board_spec = importlib.util.spec_from_file_location("board", _board_file)
board_module = importlib.util.module_from_spec(_board_spec)
sys.modules['wuziqi_core.board'] = board_module
_board_spec.loader.exec_module(board_module)
Board = board_module.Board

# Now import AI module
_ai_file = os.path.join(_project_root, 'wuziqi_core', 'ai.py')
_ai_spec = importlib.util.spec_from_file_location("ai", _ai_file)
ai_module = importlib.util.module_from_spec(_ai_spec)
sys.modules['wuziqi_core.ai'] = ai_module
_ai_spec.loader.exec_module(ai_module)

RandomAI = ai_module.RandomAI
RuleAI = ai_module.RuleAI
MinimaxAI = ai_module.MinimaxAI
MCTSAgent = ai_module.MCTSAgent
create_ai = ai_module.create_ai


def test_random_ai_select():
    ai = RandomAI()
    board = Board()
    move = ai.select(board, Board.BLACK)
    assert board.is_valid_position(*move) == True
    # Should be empty position
    assert board.grid[move[1]][move[0]] == Board.EMPTY


def test_random_ai_consistency():
    ai = RandomAI()
    board = Board()
    # After placing at 7,7, AI should not select occupied
    board.place(7, 7)
    move = ai.select(board, Board.WHITE)
    assert move != (7, 7)


def test_rule_ai_select():
    ai = RuleAI()
    board = Board()
    move = ai.select(board, Board.BLACK)
    assert board.is_valid_position(*move) == True


def test_minimax_ai_select():
    ai = MinimaxAI(depth=2)
    board = Board()
    move = ai.select(board, Board.BLACK)
    assert board.is_valid_position(*move) == True


def test_mcts_ai_select():
    ai = MCTSAgent(simulations=100)
    board = Board()
    move = ai.select(board, Board.BLACK)
    assert board.is_valid_position(*move) == True


def test_ai_level_mapping():
    ai1 = create_ai(1)  # Random
    ai2 = create_ai(2)  # Rule
    ai3 = create_ai(3)  # Minimax
    ai4 = create_ai(4)  # MCTS
    assert ai1 is not None
    assert ai2 is not None
    assert ai3 is not None
    assert ai4 is not None
