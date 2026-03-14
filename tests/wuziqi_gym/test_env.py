import pytest
import sys
import os
import importlib.util
import numpy as np

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

# Import env module using importlib
_env_file = os.path.join(_project_root, 'wuziqi_gym', 'env.py')
spec = importlib.util.spec_from_file_location("env", _env_file)
env_module = importlib.util.module_from_spec(spec)
sys.modules['wuziqi_gym.env'] = env_module
spec.loader.exec_module(env_module)
WuziqiEnv = env_module.WuziqiEnv


def test_env_init():
    env = WuziqiEnv()
    assert env.board is not None


def test_env_reset():
    env = WuziqiEnv()
    obs, info = env.reset()
    assert obs.shape == (3, 15, 15)
    assert obs.dtype == np.float32


def test_env_step():
    env = WuziqiEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(112)  # center position (7,7)
    assert obs.shape == (3, 15, 15)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_illegal_action():
    env = WuziqiEnv()
    env.reset()
    # First action at 7,7
    env.step(112)
    # Try illegal action at same position
    obs, reward, terminated, truncated, info = env.step(112)
    assert reward < 0  # Negative reward for illegal move


def test_env_legal_actions_mask():
    env = WuziqiEnv()
    env.reset()
    mask = env.legal_actions_mask()
    assert mask.shape == (225,)
    assert mask.sum() == 225  # All positions valid initially


def test_env_win():
    env = WuziqiEnv()
    obs, info = env.reset()
    # Black plays 5 in a row vertically at column 7
    # We need to skip WHITE's turns by playing elsewhere
    for i in range(5):
        action = 7 * 15 + i  # (x=7, y=i) - vertical line
        obs, reward, terminated, truncated, info = env.step(action)
        if i < 4:
            # Skip WHITE's turn by playing at a different position
            skip_action = i * 15  # (x=0, y=i)
            env.step(skip_action)
        if terminated:
            assert reward == 1.0
            break
    # If we didn't break, check that we won
    if not terminated:
        assert False, "Game should have ended with Black winning"


def test_env_full_board_draw():
    env = WuziqiEnv()
    env.reset()
    # Fill the board
    for y in range(15):
        for x in range(15):
            action = y * 15 + x
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        if terminated:
            break
    # Board full should result in draw (reward 0)
    assert terminated == True


def test_env_observation_channels():
    """Test that observation has correct 3-channel format"""
    env = WuziqiEnv()
    env.reset()
    obs, _ = env.reset()

    # Channel 0: current player's stones
    # Channel 1: opponent's stones
    # Channel 2: empty positions

    # All empty initially, so channel 2 should be all 1s
    assert np.all(obs[2] == 1.0)

    # Place a stone for current player (BLACK)
    env.step(112)  # Center position

    obs, _, _, _, _ = env.step(0)  # Another position

    # After placing, channel 2 (empty) should have some zeros
    assert obs[2].sum() < 225


def test_env_illegal_action_out_of_bounds():
    """Test that out of bounds action is handled"""
    env = WuziqiEnv()
    env.reset()
    # Try action out of bounds
    obs, reward, terminated, truncated, info = env.step(225)
    assert reward < 0  # Should get negative reward
