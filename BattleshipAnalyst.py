import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

class BattleshipCoreSampler:
    def __init__(self, rows, cols, ships, sample_size_constant=5000, no_touch=False):
        self.rows = rows
        self.cols = cols
        self.no_touch = no_touch
        self.fleet_spec = Counter(ships)
        self.base_constant = sample_size_constant  # 基準 sample 數量
        self.hits = set()
        self.misses = set()
        self.sunk = set()
        self.samples = []
        self.mode = "Hunt mode"
        self._fill_samples()

    def _diagonal_coords(self, coords): # 回傳對角的4個座標
        #print(coords)
        for r, c in coords:
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        yield nr, nc

    def _adjacent8_filled(self, grid, r, c): # 檢查周圍8格是否有任何標記
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] != 0:
                        return True
        return False

    def _compute_max_lengths(self):
        """計算每個格子往右、往下最大可行長度"""
        right_max = [[0] * self.cols for _ in range(self.rows)]
        down_max = [[0] * self.cols for _ in range(self.rows)]

        # 水平方向
        for r in range(self.rows):
            run = 0
            for c in reversed(range(self.cols)):
                if (r, c) in self.misses or (r, c) in self.hits:
                    run = 0
                else:
                    run += 1
                right_max[r][c] = run

        # 垂直方向
        for c in range(self.cols):
            run = 0
            for r in reversed(range(self.rows)):
                if (r, c) in self.misses or (r, c) in self.hits:
                    run = 0
                else:
                    run += 1
                down_max[r][c] = run

        # 存起來，方便快速查詢
        self.right_max = right_max
        self.down_max = down_max

    def _try_place_fleet_once(self):
        grid = [[0] * self.cols for _ in range(self.rows)]
        fleet = []

        # 將 hits 填 1，misses 填 2，sunk 填 1
        for r, c in self.hits:
            grid[r][c] = 1
        for r, c in self.misses:
            grid[r][c] = -1
        for r, c in self.sunk:
            grid[r][c] = 1
        
        def surrounding(cells, rows, cols):
            """回傳 cells 周圍格子 (含對角)，不超出邊界"""
            result = set()
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result.add((nr, nc))
            # 排除原格子
            result.difference_update(cells)
            return list(result)

        def can_place(cells):
            return all(0 <= rr < self.rows and 0 <= cc < self.cols and grid[rr][cc] == 0 for rr, cc in cells)

        def occupy(cells):
            for rr, cc in cells:
                grid[rr][cc] = 1
            if self.no_touch:
                surround = surrounding(cells, self.rows, self.cols)
                for r, c in surround:
                    grid[r][c] -= 1

        def release(cells):
            for rr, cc in cells:
                grid[rr][cc] = 0
            if self.no_touch:
                surround = surrounding(cells, self.rows, self.cols)
                for r, c in surround:
                    grid[r][c] += 1

        # 將船展平成單一列表以支援局部回溯
        fleet_order = []
        for length, count in self.fleet_spec.items():
            fleet_order.extend([length] * count)

        index = 0
        while index < len(fleet_order):
            length = fleet_order[index]
            placed = False
            candidates = []

            # 遍歷 grid，只看空格
            for r in range(self.rows):
                for c in range(self.cols):
                    if grid[r][c] != 0:
                        continue
                    # 水平
                    cells_h = [(r, c+i) for i in range(length)]
                    if can_place(cells_h):
                        candidates.append(cells_h)
                    # 垂直
                    cells_v = [(r+i, c) for i in range(length)]
                    if can_place(cells_v):
                        candidates.append(cells_v)

            random.shuffle(candidates)
            for cells in candidates:
                occupy(cells)
                fleet.append(cells)
                placed = True
                index += 1  # 成功放船，進入下一艘
                break

            if not placed:
                # 放失敗，僅回溯上一艘船
                if fleet:
                    release(fleet[-1])
                    fleet.pop()
                    index -= 1  # 回到上一艘船重新嘗試
                else:
                    # 第一艘船就放不下，完全失敗
                    return None


        return fleet


    def _fleet_cells(self, fleet):
        s = set()
        for ship in fleet:
            s.update(ship)
        return s

    def _is_fleet_valid_under_constraints(self, fleet, new_point=None, is_hit=True, fast=True):
        """
        檢查 fleet 是否符合目前的 hits/misses 限制

        Args:
            fleet: 船隊配置 (list of ships)
            new_point: (r, c) 最新更新的點
            is_hit: bool, True=hit, False=miss
            fast: 如果 True，只檢查 new_point；否則檢查所有 hits/misses
        """
        cells = self._fleet_cells(fleet)
        #print(cells)
        if fast: # 只檢查新點 
            if is_hit: # 如果新點是hit
                return new_point in cells
            else: # 如果新點是miss
                return new_point not in cells
        else:
            if not self.hits.issubset(cells): # 檢查已記錄的hits是否全部在fleet裡
                return False
            if not self.sunk.issubset(cells): # 檢查已記錄的sunks是否全部在fleet裡
                return False
            if any(m in cells for m in self.misses): # 檢查已記錄的misses是否有在fleet裡
                return False
            return True
    
    def _remove_samples(self, new_point=None, is_hit=True, fast=True):
        before = len(self.samples)
        self.samples = [s for s in self.samples if self._is_fleet_valid_under_constraints(s, new_point, is_hit, fast)]
        after = len(self.samples)
        removed = before - after
        if removed > 0:
            print(f"[DEBUG] Removed {removed} invalid samples ({after} left)")
        
    def _dynamic_sample_size(self):
        remaining_cells = self.rows * self.cols - len(self.hits) - len(self.misses)
        remaining_ships = []
        for length, count in self.fleet_spec.items():
            remaining_ships += [length]*count
        if not remaining_ships or remaining_cells <= 0:
            return 5
        sum_squares = sum(l**2 for l in remaining_ships)
        size = int(self.base_constant * remaining_cells / sum_squares)
        return max(size, 5)

    def _fill_samples(self, debug=True):
        #self._compute_max_lengths()  # 更新統計
        sample_target = self._dynamic_sample_size()
        attempts = 0
        #success = 0
        limit = sample_target * 1 
        while len(self.samples) < sample_target and attempts < limit:
            attempts += 1
            fleet = self._try_place_fleet_once()
            if fleet is None:
                continue
            #if not self._is_fleet_valid_under_constraints(fleet, fast=False):
                #continue
            #success += 1
            self.samples.append(fleet)

        if debug:
            #ratio = success / attempts if attempts else 0
            print(f"Totally {len(self.samples)} samples")

    def mark_hit(self, r, c):
        #print(self.samples)
        self.hits.add((r, c))
        self._remove_samples((r, c), is_hit=True, fast=True)
        corner = self._diagonal_coords([(r, c)])
        for p in corner:
            self.misses.add(p) # set, 重複加入不影響
            self._remove_samples(p, is_hit=False, fast=True)
        self._fill_samples()

    def mark_miss(self, r, c):
        self.misses.add((r, c))
        self._remove_samples((r, c), is_hit=False, fast=True)

    def mark_sunk(self, length, vertical, start_r, start_c):
        # 1. 更新 fleet_spec
        if length in self.fleet_spec and self.fleet_spec[length] > 0:
            self.fleet_spec[length] -= 1
            if self.fleet_spec[length] == 0:
                del self.fleet_spec[length]
        else:
            print(f"[警告] 沒有可用的長度 {length} 船可以標記沉沒！")

        # 2. 計算這艘船的格子
        coords = [(start_r + i, start_c) if vertical else (start_r, start_c + i) for i in range(length)]

        # 3. 更新 hits → sunk
        for r, c in coords:
            self.hits.discard((r, c))   # 從 hits 移除
            self.sunk.add((r, c))       # 加到 sunk
            self._remove_samples((r, c), is_hit=True, fast=True)

        # 4. no_touch 模式 → 標記周圍為 miss
        if self.no_touch:
            for r, c in coords:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in coords:
                            self.misses.add((nr, nc))
                            self._remove_samples((nr, nc), is_hit=False, fast=True)

        # 5. 重新填樣本
        self._fill_samples()


    def probability_map(self):
        counts = [[0] * self.cols for _ in range(self.rows)]
        for fleet in self.samples:
            for ship in fleet:
                for (r, c) in ship:
                    if (r, c) not in self.hits and (r, c) not in self.sunk:  # <--- 跳過 sunk
                        counts[r][c] += 1
        total = len(self.samples)
        if total == 0:
            return np.zeros((self.rows, self.cols)), 0
        prob = np.array([[counts[r][c] / total for c in range(self.cols)] for r in range(self.rows)])
        return prob, total

    def suggest_next(self, k=5):
        prob, _ = self.probability_map()
        cand = [((r, c), prob[r][c]) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.hits and (r, c) not in self.misses]
        cand.sort(key=lambda x: -x[1])
        return cand[:k]

    def suggest_smart(self, k=5):
        prob, _ = self.probability_map()

        # --- Target mode ---
        active_hits = self.hits - self.sunk  # 只考慮未沉的命中
        if active_hits:
            self.mode = "Target mode"
            cand = []
            for (r, c) in active_hits:
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if (nr, nc) not in self.hits and (nr, nc) not in self.misses and (nr, nc) not in self.sunk:
                            cand.append(((nr, nc), prob[nr][nc]))
            if cand:
                cand.sort(key=lambda x: -x[1])
                return cand[:k]

        # --- Hunt mode ---
        self.mode = "Hunt mode"
        cand = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.hits or (r, c) in self.misses or (r, c) in self.sunk:
                    continue
                p = prob[r][c]
                if (r + c) % 2 == 0:
                    p *= 1.2
                cand.append(((r, c), p))

        cand.sort(key=lambda x: -x[1])
        return cand[:k]

    def show(self):
        print(self.samples)
        # 計算 base counts
        counts = np.zeros((self.rows, self.cols), dtype=int)
        for fleet in self.samples:
            for ship in fleet:
                for (r, c) in ship:
                    if (r, c) not in self.hits and (r, c) not in self.sunk:
                        counts[r, c] += 1
        total_samples = len(self.samples)
        base_prob = counts / total_samples if total_samples > 0 else counts

        # 判斷 mode
        active_hits = self.hits - self.sunk
        mode = "Target mode" if active_hits else "Hunt mode"

        # Hunt 修正的機率（checkerboard 加權）
        hunt_prob = base_prob.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.hits or (r, c) in self.misses or (r, c) in self.sunk:
                    hunt_prob[r, c] = 0
                elif (r + c) % 2 == 0:
                    hunt_prob[r, c] *= 1.2

        sample_target = self._dynamic_sample_size()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 左圖：基本 Hot Map
        im0 = axes[0].imshow(base_prob, cmap='hot', origin='upper')
        axes[0].set_title(f'Base Hot Map | Samples={len(self.samples)}/{sample_target}')
        fig.colorbar(im0, ax=axes[0])

        # 右圖：Hunt 修正 Hot Map
        im1 = axes[1].imshow(hunt_prob, cmap='hot', origin='upper')
        axes[1].set_title(f'Hunt-corrected Hot Map | {mode}')
        fig.colorbar(im1, ax=axes[1])

        # 標記 hit / miss / sunk 並顯示 sample 次數
        for ax, prob_map in zip(axes, [base_prob, hunt_prob]):
            for r in range(self.rows):
                for c in range(self.cols):
                    if counts[r, c] > 0:
                        ax.text(c, r, f'{counts[r,c]}', color='green', ha='center', va='center', fontsize=8)
            for r, c in self.hits:
                ax.text(c, r, 'H', color='yellow', ha='center', va='center', fontsize=12, fontweight='bold')
            for r, c in self.misses:
                ax.text(c, r, 'M', color='blue', ha='center', va='center', fontsize=10)
            for r, c in self.sunk:
                ax.text(c, r, 'S', color='red', ha='center', va='center', fontsize=12, fontweight='bold')
                rect = patches.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

            ax.set_xticks(range(self.cols))
            ax.set_yticks(range(self.rows))
            ax.grid(True, color='blue', linewidth=0.5)

        plt.show()


"""
1. 改良hunt mode
2. sample方式似乎有問題
"""
if __name__ == "__main__":
    sampler = BattleshipCoreSampler(rows=7, cols=7, ships=[5,4,2], sample_size_constant=3, no_touch=True)
    debug_mode = True
    print(f"起始樣本數: {len(sampler.samples)}")
    while True:
        cmd = input("輸入指令(hit r c | miss r c | sunk length h(v) r c | show | suggest | suggest_smart | debug | quit): ").strip().split()
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
            # 呼叫 suggest_smart 更新 mode
            sampler.suggest_smart()
            sampler.show()
        elif cmd[0] == "suggest":
            k = int(cmd[1]) if len(cmd) > 1 else 5
            suggestions = sampler.suggest_next(k)
            for pos, p in suggestions:
                print(pos, f"{p:.3f}")
        elif cmd[0] == "suggest_smart":
            k = int(cmd[1]) if len(cmd) > 1 else 5
            suggestions = sampler.suggest_smart(k)
            mode = "Target mode" if sampler.hits else "Hunt mode"
            print(f"[{mode}] 建議 {k} 個位置：")
            for pos, p in suggestions:
                print(pos, f"{p:.3f}")
        elif cmd[0] == "debug":
            if len(cmd) > 1 and cmd[1].lower() in ["on", "off"]:
                debug_mode = (cmd[1].lower() == "on")
                print(f"Debug 模式: {'開啟' if debug_mode else '關閉'}")
            else:
                print(f"目前 Debug 模式: {'開啟' if debug_mode else '關閉'}")
        else:
            print("未知指令")

"""
改進方向:
1. 增加更高效的樣本生成算法，減少無效嘗試。應該從有效的格子(形狀奇怪也是)直接生成
2. sunk後應該直接去除船隻，減少生成樣本的數量
"""
