import random
import matplotlib.pyplot as plt
import numpy as np

class BattleshipIncrementalSampler:
    def __init__(self, rows, cols, ships, no_touch=False, sample_size=5000):
        self.rows = rows
        self.cols = cols
        self.ships = ships
        self.no_touch = no_touch
        self.sample_size = sample_size
        self.hits = set()
        self.misses = set()
        self.sunk_cells = set()
        self.ships_remaining = {}
        for s in ships:
            self.ships_remaining[s] = self.ships_remaining.get(s, 0) + 1
        self.samples = []
        self._generate_samples()

    def _generate_samples(self):
        self.samples = []
        attempts = 0
        while len(self.samples) < self.sample_size and attempts < self.sample_size*10:
            attempts += 1
            grid = [[0]*self.cols for _ in range(self.rows)]
            fleet = []
            valid = True
            for ship_length, count in self.ships_remaining.items():
                for _ in range(count):
                    placed = False
                    for _ in range(50):
                        orient = random.choice(['H','V'])
                        if orient == 'H':
                            r = random.randint(0, self.rows-1)
                            c = random.randint(0, self.cols-ship_length)
                            if all(grid[r][c+i]==0 for i in range(ship_length)):
                                if self.no_touch and any(self._adjacent_filled(grid, r, c+i) for i in range(ship_length)):
                                    continue
                                for i in range(ship_length):
                                    grid[r][c+i] = 1
                                fleet.append([(r, c+i) for i in range(ship_length)])
                                placed = True
                                break
                        else:
                            r = random.randint(0, self.rows-ship_length)
                            c = random.randint(0, self.cols-1)
                            if all(grid[r+i][c]==0 for i in range(ship_length)):
                                if self.no_touch and any(self._adjacent_filled(grid, r+i, c) for i in range(ship_length)):
                                    continue
                                for i in range(ship_length):
                                    grid[r+i][c] = 1
                                fleet.append([(r+i, c) for i in range(ship_length)])
                                placed = True
                                break
                    if not placed:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                self.samples.append(fleet)

    def _adjacent_filled(self, grid, r, c):
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] != 0:
                        return True
        return False

    def _filter_samples(self):
        new_samples = []
        for fleet in self.samples:
            valid = True
            for r,c in self.hits:
                if not any((r,c) in ship for ship in fleet):
                    valid = False
                    break
            for r,c in self.misses:
                if any((r,c) in ship for ship in fleet):
                    valid = False
                    break
            for length, remaining in self.ships_remaining.items():
                count_in_sample = sum(1 for ship in fleet if len(ship)==length)
                if count_in_sample < remaining:
                    valid = False
                    break
            if valid:
                new_samples.append(fleet)
        self.samples = new_samples
        if len(self.samples) < self.sample_size:
            self._generate_samples()

    def mark_hit(self, r, c):
        self.hits.add((r, c))
        self._filter_samples()

    def mark_miss(self, r, c):
        self.misses.add((r, c))
        self._filter_samples()

    def mark_sunk(self, length, count=1, cells=None):
        if length in self.ships_remaining:
            self.ships_remaining[length] = max(0, self.ships_remaining[length]-count)
            if cells:
                for cell in cells:
                    self.sunk_cells.add(tuple(cell))
            print(f"標記 {length} 長度船沉沒 {count} 艘，剩餘 {self.ships_remaining[length]} 艘")
            self._filter_samples()
        else:
            print(f"警告: 船長 {length} 不存在")

    def probability_map(self):
        prob_map = [[0]*self.cols for _ in range(self.rows)]
        for fleet in self.samples:
            for ship in fleet:
                for r,c in ship:
                    prob_map[r][c] += 1
        total = len(self.samples)
        if total>0:
            for r in range(self.rows):
                for c in range(self.cols):
                    prob_map[r][c] /= total
        return np.array(prob_map), total

    def suggest_next(self, k=5, margin=1):
        prob_map, _ = self.probability_map()
        candidates = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in self.hits and (r,c) not in self.misses:
                    candidates.append(((r,c), prob_map[r][c]))
        if self.no_touch:
            filtered = []
            for (r,c), prob in candidates:
                too_close = any(abs(r-sr)<=margin and abs(c-sc)<=margin for sr,sc in self.sunk_cells)
                if not too_close:
                    filtered.append(((r,c), prob))
            candidates = filtered
        candidates.sort(key=lambda x: -x[1])
        return candidates[:k]

    def show_cells(self):
        grid = np.ones((self.rows, self.cols, 3))
        for r,c in self.misses:
            grid[r,c] = [0.8,0.8,0.8]
        for r,c in self.hits:
            grid[r,c] = [1,0,0]
        for r,c in self.sunk_cells:
            grid[r,c] = [0,0,0]
        plt.figure(figsize=(6,6))
        plt.imshow(grid, origin='upper')
        plt.title("命中(紅), 落空(灰), 沉船(黑)")
        plt.grid(True, color='blue', linewidth=0.5)
        plt.xticks(range(self.cols))
        plt.yticks(range(self.rows))
        plt.show()

    def clear_cell(self, r, c):
        self.hits.discard((r,c))
        self.misses.discard((r,c))
        self.sunk_cells.discard((r,c))
        print(f"已清除格子 ({r},{c}) 狀態")
        self._generate_samples()

    def set_cell(self, r, c, status):
        self.clear_cell(r,c)
        if status=="hit":
            self.mark_hit(r,c)
        elif status=="miss":
            self.mark_miss(r,c)
        elif status=="sunk":
            self.sunk_cells.add((r,c))
        else:
            print("無效狀態")
        self._generate_samples()

    def show_probability_map(self, prob_map=None):
        if prob_map is None:
            prob_map, total = self.probability_map()
        else:
            total = len(self.samples)
        plt.figure(figsize=(6,6))
        plt.imshow(prob_map, cmap='hot', origin='upper')
        plt.colorbar(label='機率')
        plt.title(f'Probability map, Number of samples={total}')
        plt.grid(True, color='blue', linewidth=0.5)
        plt.xticks(range(self.cols))
        plt.yticks(range(self.rows))
        for r,c in self.hits:
            plt.text(c, r, 'X', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
        for r,c in self.misses:
            plt.text(c, r, '•', color='black', ha='center', va='center', fontsize=10)
        for r,c in self.sunk_cells:
            plt.text(c, r, 'S', color='blue', ha='center', va='center', fontsize=12, fontweight='bold')
        plt.show()

# === 互動模式 ===
if __name__ == "__main__":
    rows, cols = 6, 6
    ships = [4,3,2]
    sampler = BattleshipIncrementalSampler(rows, cols, ships, no_touch=True, sample_size=20000)

    print("=== Battleship AI Interactive Mode ===")
    print("指令: hit r c | miss r c | sunk length [count] | suggest [k] | show | quit")

    while True:
        try:
            cmd = input("> ").strip().split()
            if not cmd:
                continue
            action = cmd[0].lower()
            if action == "quit":
                break
            elif action == "hit" and len(cmd)==3:
                r,c = int(cmd[1]), int(cmd[2])
                sampler.mark_hit(r,c)
                print(f"已標記命中 ({r},{c})")
            elif action == "miss" and len(cmd)==3:
                r,c = int(cmd[1]), int(cmd[2])
                sampler.mark_miss(r,c)
                print(f"已標記落空 ({r},{c})")
            elif action == "sunk" and len(cmd)>=2:
                length = int(cmd[1])
                count = int(cmd[2]) if len(cmd)==3 else 1
                sampler.mark_sunk(length, count)
            elif action == "suggest":
                k = int(cmd[1]) if len(cmd)>1 else 5
                suggestions = sampler.suggest_next(k)
                for pos, prob in suggestions:
                    print(f"{pos} -> {prob:.3f}")
            elif action == "show":
                prob_map, total = sampler.probability_map()
                print(f"樣本數: {total}")
                sampler.show_probability_map(prob_map)
            else:
                print("無效指令")
        except Exception as e:
            print(f"錯誤: {e}")
