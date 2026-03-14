import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')

import argparse
import numpy as np
from stable_baselines3 import PPO
from wuziqi_core.board import Board
from wuziqi_core.ai import RandomAI, RuleAI, MCTSAgent, create_ai
from wuziqi_gym.env import WuziqiEnv


def play_game(model, opponent, first_player=0):
    """执行一局游戏，返回胜者"""
    env = WuziqiEnv()
    obs, _ = env.reset()

    current_player = first_player
    max_steps = 200
    steps = 0

    while not env.board.is_game_over() and steps < max_steps:
        if current_player == 1:  # AI (model)
            action, _ = model.predict(obs, deterministic=True)
        else:  # 对手
            if opponent == "random":
                ai = RandomAI()
                move = ai.select(env.board, env.board.current_player)
                action = move[0] + move[1] * 15
            elif opponent == "rule":
                ai = RuleAI()
                move = ai.select(env.board, env.board.current_player)
                action = move[0] + move[1] * 15
            elif opponent == "mcts":
                ai = MCTSAgent(simulations=500)
                move = ai.select(env.board, env.board.current_player)
                action = move[0] + move[1] * 15
            else:
                # 随机
                moves = env.legal_actions()
                action = np.random.choice(moves)

        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated:
            return 1 if reward > 0 else -1 if reward < 0 else 0

        current_player = 1 - current_player
        steps += 1

    return 0  # 平局


def evaluate(model_path, opponent, num_games=100):
    """评测模型

    Args:
        model_path: 模型路径
        opponent: 对手类型 ("random", "rule", "mcts")
        num_games: 对局数量

    Returns:
        dict: 评测结果
    """
    # 加载模型
    if model_path:
        model = PPO.load(model_path)
    else:
        print("Warning: No model loaded, using random baseline")
        model = None

    wins = losses = draws = 0

    for i in range(num_games):
        first_player = i % 2
        result = play_game(model, opponent, first_player)

        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            print(f"Played {i + 1}/{num_games} games")

    return {
        "win_rate": wins / num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Wuziqi AI')
    parser.add_argument('--model', type=str, default='./models/ppo_wuziqi',
                       help='Path to trained model')
    parser.add_argument('--opponent', type=str, default='random',
                       choices=['random', 'rule', 'mcts'],
                       help='Opponent type')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games to play')

    args = parser.parse_args()

    result = evaluate(args.model, args.opponent, args.games)

    print(f"\n=== Evaluation Results ===")
    print(f"Opponent: {args.opponent}")
    print(f"Games: {args.games}")
    print(f"Wins: {result['wins']}")
    print(f"Losses: {result['losses']}")
    print(f"Draws: {result['draws']}")
    print(f"Win Rate: {result['win_rate']:.2%}")
