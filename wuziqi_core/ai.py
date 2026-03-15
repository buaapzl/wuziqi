# /Users/rr2017/code/wuziqi/wuziqi_core/ai.py
import random
from typing import Tuple, Optional
from wuziqi_core.board import Board


class RandomAI:
    """Lv1: 随机落子 + 基础防守"""
    def select(self, board: Board, color: int) -> Tuple[int, int]:
        # 先检查是否有必防的点（四连被堵）
        for y in range(board.SIZE):
            for x in range(board.SIZE):
                if board.grid[y][x] != Board.EMPTY:
                    continue
                # 模拟落子后检查是否形成四连
                board.grid[y][x] = color
                if self._check_four_in_row(board, x, y, color):
                    board.grid[y][x] = Board.EMPTY
                    return (x, y)
                board.grid[y][x] = Board.EMPTY

        # 没有必防的点，随机落子
        moves = board.get_valid_moves()
        return random.choice(moves) if moves else (7, 7)

    def _check_four_in_row(self, board: Board, x: int, y: int, color: int) -> bool:
        # 检查4个方向是否有四连
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 正方向
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if not board.is_valid_position(nx, ny) or board.grid[ny][nx] != color:
                    break
                count += 1
            # 反方向
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if not board.is_valid_position(nx, ny) or board.grid[ny][nx] != color:
                    break
                count += 1
            if count >= 4:
                return True
        return False


class RuleAI:
    """Lv2: 评估函数 - 进攻/防守权重"""
    def select(self, board: Board, color: int) -> Tuple[int, int]:
        opponent = Board.WHITE if color == Board.BLACK else Board.BLACK

        best_score = float('-inf')
        best_move = None

        for x, y in board.get_valid_moves():
            attack_score = self._evaluate_position(board, x, y, color)
            defense_score = self._evaluate_position(board, x, y, opponent)
            total_score = attack_score + defense_score * 1.1  # 防守优先

            if total_score > best_score:
                best_score = total_score
                best_move = (x, y)

        return best_move or (7, 7)

    def _evaluate_position(self, board: Board, x: int, y: int, color: int) -> float:
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            open_ends = 0

            # 正方向
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if not board.is_valid_position(nx, ny):
                    break
                if board.grid[ny][nx] == color:
                    count += 1
                elif board.grid[ny][nx] == Board.EMPTY:
                    open_ends += 1
                    break
                else:
                    break

            # 反方向
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if not board.is_valid_position(nx, ny):
                    break
                if board.grid[ny][nx] == color:
                    count += 1
                elif board.grid[ny][nx] == Board.EMPTY:
                    open_ends += 1
                    break
                else:
                    break

            # 计分
            if count >= 5:
                score += 100000
            elif count == 4 and open_ends == 2:
                score += 10000
            elif count == 4 and open_ends == 1:
                score += 1000
            elif count == 3 and open_ends == 2:
                score += 1000
            elif count == 3 and open_ends == 1:
                score += 100
            elif count == 2 and open_ends == 2:
                score += 100

        return score


class MinimaxAI:
    """Lv3: Minimax + Alpha-Beta剪枝"""
    def __init__(self, depth: int = 3):
        self.depth = depth

    def select(self, board: Board, color: int) -> Tuple[int, int]:
        best_score = float('-inf')
        best_move = None

        moves = board.get_valid_moves()
        # 限制搜索节点
        if len(moves) > 30:
            moves = moves[:30]

        for x, y in moves:
            board.grid[y][x] = color
            score = self._minimax(board, self.depth - 1, float('-inf'), float('inf'), False, color)
            board.grid[y][x] = Board.EMPTY

            if score > best_score:
                best_score = score
                best_move = (x, y)

        return best_move or (7, 7)

    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, is_maximizing: bool, original_color: int) -> float:
        opponent = Board.WHITE if original_color == Board.BLACK else Board.BLACK
        current_color = opponent if is_maximizing else original_color

        winner = board.get_winner()
        if winner == original_color:
            return 100000
        if winner == opponent:
            return -100000
        if board.is_full() or depth == 0:
            return self._evaluate_board(board, original_color)

        moves = board.get_valid_moves()
        if len(moves) > 25:
            moves = moves[:25]

        if is_maximizing:
            max_eval = float('-inf')
            for x, y in moves:
                board.grid[y][x] = current_color
                eval = self._minimax(board, depth - 1, alpha, beta, False, original_color)
                board.grid[y][x] = Board.EMPTY
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for x, y in moves:
                board.grid[y][x] = current_color
                eval = self._minimax(board, depth - 1, alpha, beta, True, original_color)
                board.grid[y][x] = Board.EMPTY
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_board(self, board: Board, color: int) -> float:
        rule_ai = RuleAI()
        score = 0
        for y in range(board.SIZE):
            for x in range(board.SIZE):
                if board.grid[y][x] == color:
                    score += rule_ai._evaluate_position(board, x, y, color)
                elif board.grid[y][x] != Board.EMPTY:
                    score -= rule_ai._evaluate_position(board, x, y, Board.WHITE if color == Board.BLACK else Board.BLACK)
        return score


class MCTSAgent:
    """Lv4: MCTS蒙特卡洛树搜索"""
    def __init__(self, simulations: int = 1000):
        self.simulations = simulations

    def select(self, board: Board, color: int) -> Tuple[int, int]:
        # 简化实现：使用随机模拟
        best_move = None
        best_wins = 0

        moves = board.get_valid_moves()
        # 只考虑靠近已有棋子的位置
        moves = self._filter_nearby_moves(board, moves)

        for x, y in moves[:15]:  # 限制候选数
            wins = 0
            for _ in range(self.simulations // len(moves[:15])):
                board.grid[y][x] = color
                result = self._simulate(board, color)
                if result == color:
                    wins += 1
                board.grid[y][x] = Board.EMPTY

            if wins > best_wins:
                best_wins = wins
                best_move = (x, y)

        return best_move or random.choice(moves)

    def _filter_nearby_moves(self, board: Board, moves: list) -> list:
        # 只返回靠近已有棋子的位置
        nearby = []
        for x, y in moves:
            has_nearby = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if board.is_valid_position(nx, ny) and board.grid[ny][nx] != Board.EMPTY:
                        has_nearby = True
                        break
                if has_nearby:
                    break
            if has_nearby or board.move_count < 2:  # 前两步全局随机
                nearby.append((x, y))
        return nearby if nearby else moves[:30]

    def _simulate(self, board: Board, original_color: int) -> int:
        # 保存棋盘副本，模拟结束后恢复
        grid_backup = [row[:] for row in board.grid]
        move_count_backup = board.move_count

        current = Board.WHITE if original_color == Board.BLACK else Board.BLACK
        while not board.is_game_over():
            moves = board.get_valid_moves()
            if not moves:
                break
            x, y = random.choice(moves)
            board.grid[y][x] = current
            board.move_count += 1
            current = Board.WHITE if current == Board.BLACK else Board.BLACK

        winner = board.get_winner()

        # 恢复棋盘状态
        board.grid = grid_backup
        board.move_count = move_count_backup

        return winner


def create_ai(level: int):
    """工厂函数：创建指定难度的AI"""
    if level == 1:
        return RandomAI()
    elif level == 2:
        return RuleAI()
    elif level == 3:
        return MinimaxAI(depth=3)
    elif level == 4:
        return MCTSAgent(simulations=100)  # 减少模拟次数以加快速度
    else:
        return None  # Lv5 需要PPO模型
