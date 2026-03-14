class Board:
    SIZE = 15  # 15x15棋盘
    EMPTY = 0
    BLACK = 1  # 先手
    WHITE = 2  # 后手

    def __init__(self):
        self.grid = [[self.EMPTY] * self.SIZE for _ in range(self.SIZE)]
        self.current_player = self.BLACK
        self.move_count = 0

    def place(self, x: int, y: int) -> bool:
        """落子，返回是否成功"""
        if not self.is_valid_position(x, y):
            return False
        if self.grid[y][x] != self.EMPTY:
            return False
        self.grid[y][x] = self.current_player
        self.move_count += 1
        self._switch_player()
        return True

    def is_valid_position(self, x: int, y: int) -> bool:
        """检查坐标是否有效"""
        return 0 <= x < self.SIZE and 0 <= y < self.SIZE

    def get_winner(self) -> int:
        """检查是否有五连，返回获胜方: 0:无, 1:黑, 2:白"""
        for y in range(self.SIZE):
            for x in range(self.SIZE):
                if self.grid[y][x] == self.EMPTY:
                    continue
                player = self.grid[y][x]
                # 水平
                if x + 4 < self.SIZE and all(self.grid[y][x+i] == player for i in range(5)):
                    return player
                # 垂直
                if y + 4 < self.SIZE and all(self.grid[y+i][x] == player for i in range(5)):
                    return player
                # 对角线
                if x + 4 < self.SIZE and y + 4 < self.SIZE and all(self.grid[y+i][x+i] == player for i in range(5)):
                    return player
                # 反对角线
                if x - 4 >= 0 and y + 4 < self.SIZE and all(self.grid[y+i][x-i] == player for i in range(5)):
                    return player
        return 0

    def is_full(self) -> bool:
        """检查棋盘是否已满"""
        return self.move_count >= self.SIZE * self.SIZE

    def is_game_over(self) -> bool:
        """检查游戏是否结束（分出胜负或平局）"""
        return self.get_winner() != 0 or self.is_full()

    def get_valid_moves(self) -> list:
        """获取所有合法落子位置"""
        return [(x, y) for y in range(self.SIZE) for x in range(self.SIZE) if self.grid[y][x] == self.EMPTY]

    def _switch_player(self):
        """切换当前玩家"""
        self.current_player = self.BLACK if self.current_player == self.WHITE else self.WHITE

    def reset(self):
        """重置棋盘"""
        self.grid = [[self.EMPTY] * self.SIZE for _ in range(self.SIZE)]
        self.current_player = self.BLACK
        self.move_count = 0
