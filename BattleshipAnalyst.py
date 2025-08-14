import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class BattleshipCoreSampler:
    def __init__(self, rows, cols, ships, sample_size=20000, no_touch=True):
        self.rows = rows
        self.cols = cols
        self.no_touch = no_touch
        self.fleet_spec = Counter(ships) if not isinstance(ships, dict) else Counter(ships)
        self.sample_size = sample_size
        self.hits = set()
        self.misses = set()
        self.samples = []
        self._generate_samples()

    def _adjacent_filled(self, grid, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] != 0:
                        return True
        return False

    def _try_place_fleet_once(self):
        grid = [[0] * self.cols for _ in range(self.rows)]
        fleet = []
        for length, count in self.fleet_spec.items():
            for _ in range(count):
                placed = False
                for _ in range(100):
                    horizontal = random.choice([True, False])
                    if horizontal:
                        r = random.randint(0, self.rows - 1)
                        c = random.randint(0, self.cols - length)
                        cells = [(r, c + i) for i in range(length)]
                        if all(grid[r][c + i] == 0 for i in range(length)):
                            if self.no_touch and any(self._adjacent_filled(grid, r, c + i) for i in range(length)):
                                continue
                            for i in range(length):
                                grid[r][c + i] = 1
                            fleet.append(cells)
                            placed = True
                            break
                    else:
                        r = random.randint(0, self.rows - length)
                        c = random.randint(0, self.cols - 1)
                        cells = [(r + i, c) for i in range(length)]
                        if all(grid[r + i][c] == 0 for i in range(length)):
                            if self.no_touch and any(self._adjacent_filled(grid, r + i, c) for i in range(length)):
                                continue
                            for i in range(length):
                                grid[r + i][c] = 1
                            fleet.append(cells)
                            placed = True
                            break
                if not placed:
                    return None
        return fleet

    def _fleet_cells(self, fleet):
        s = set()
        for ship in fleet:
            s.update(ship)
        return s

    def _is_fleet_valid_under_constraints(self, fleet):
        cells = self._fleet_cells(fleet)
        if not self.hits.issubset(cells):
            return False
        if any(m in cells for m in self.misses):
            return False
        return True

    def _generate_samples(self):
        self.samples = []
        attempts = 0
        limit = self.sample_size * 10
        while len(self.samples) < self.sample_size and attempts < limit:
            attempts += 1
            fleet = self._try_place_fleet_once()
            if fleet is None:
                continue
            if not self._is_fleet_valid_under_constraints(fleet):
                continue
            self.samples.append(fleet)

    def mark_hit(self, r, c):
        self.hits.add((r, c))
        self._regenerate_to_fill()

    def mark_miss(self, r, c):
        self.misses.add((r, c))
        self._regenerate_to_fill()

    def mark_sunk(self, length, vertical, start_r, start_c):
        coords = []
        for i in range(length):
            if vertical:
                coords.append((start_r + i, start_c))
            else:
                coords.append((start_r, start_c + i))
        # 先加入命中格
        for r, c in coords:
            self.hits.add((r, c))
        # no_touch 下標記周圍為落空
        if self.no_touch:
            for r, c in coords:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in coords:
                            self.misses.add((nr, nc))
        # 過濾樣本：去掉含有沉船位置之外的 hits/misses 不符的 sample
        self._regenerate_to_fill(sunk_coords=coords)

    def _regenerate_to_fill(self, sunk_coords=None):
        new_samples = []
        for f in self.samples:
            valid = True
            fleet_cells = self._fleet_cells(f)
            if not self.hits.issubset(fleet_cells):
                valid = False
            if any(m in fleet_cells for m in self.misses):
                valid = False
            # 如果傳入 sunk_coords，剔除與沉船重疊不正確的 sample
            if sunk_coords:
                for ship in f:
                    if any(cell not in self.hits for cell in ship) and any(cell in sunk_coords for cell in ship):
                        valid = False
                        break
            if valid:
                new_samples.append(f)
        self.samples = new_samples
        attempts = 0
        limit = self.sample_size * 10
        while len(self.samples) < self.sample_size and attempts < limit:
            attempts += 1
            fleet = self._try_place_fleet_once()
            if fleet is None:
                continue
            fleet_cells = self._fleet_cells(fleet)
            if not self.hits.issubset(fleet_cells):
                continue
            if any(m in fleet_cells for m in self.misses):
                continue
            self.samples.append(fleet)

    def probability_map(self):
        counts = [[0] * self.cols for _ in range(self.rows)]
        for fleet in self.samples:
            for ship in fleet:
                for (r, c) in ship:
                    if (r, c) not in self.hits:
                        counts[r][c] += 1
        total = len(self.samples)
        if total == 0:
            return np.zeros((self.rows, self.cols)), 0
        prob = np.array([[counts[r][c] / total for c in range(self.cols)] for r in range(self.rows)])
        return prob, total

    def suggest_next(self, k=10):
        prob, _ = self.probability_map()
        cand = [((r, c), prob[r][c]) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.hits and (r, c) not in self.misses]
        cand.sort(key=lambda x: -x[1])
        return cand[:k]

    def show(self):
        prob, total = self.probability_map()
        plt.figure(figsize=(6,6))
        plt.imshow(prob, cmap='hot', origin='upper')
        plt.colorbar(label='Probability')
        plt.title(f'Probability Hot Map, Number of Samples={total}')
        for r, c in self.hits:
            plt.text(c, r, 'X', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
        for r, c in self.misses:
            plt.text(c, r, '•', color='white', ha='center', va='center', fontsize=10)
        plt.grid(True, color='blue', linewidth=0.5)
        plt.xticks(range(self.cols))
        plt.yticks(range(self.rows))
        plt.show()

# 互動模式
if __name__ == "__main__":
    sampler = BattleshipCoreSampler(rows=8, cols=8, ships=[5,3,3,2,2], sample_size=20000, no_touch=True)
    print(f"起始樣本數: {len(sampler.samples)}")
    while True:
        cmd = input("輸入指令(hit r c / miss r c / sunk len v(h) r c / show / suggest (num) / quit): ").strip().split()
        if not cmd:
            continue
        if cmd[0] == "quit":
            break
        elif cmd[0] == "hit" and len(cmd) == 3:
            r, c = int(cmd[1]), int(cmd[2])
            sampler.mark_hit(r, c)
            print("已標記命中", (r, c))
        elif cmd[0] == "miss" and len(cmd) == 3:
            r, c = int(cmd[1]), int(cmd[2])
            sampler.mark_miss(r, c)
            print("已標記落空", (r, c))
        elif cmd[0] == "sunk" and len(cmd) == 5:
            length = int(cmd[1])
            vertical = cmd[2].lower() == 'v'
            r, c = int(cmd[3]), int(cmd[4])
            sampler.mark_sunk(length, vertical, r, c)
            print(f"已標記沉船 長度{length} {'垂直' if vertical else '水平'} 起點({r},{c})")
        elif cmd[0] == "show":
            sampler.show()
        elif cmd[0] == "suggest":
            k = int(cmd[1]) if len(cmd) > 1 else 10  # 預設 5
            suggestions = sampler.suggest_next(k)
            for pos, p in suggestions:
                print(pos, f"{p:.4f}")
        else:
            print("未知指令")
