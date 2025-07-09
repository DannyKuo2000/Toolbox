import numpy as np
from scipy.optimize import minimize
from scipy.special import comb
import csv, json

### --------- Part 1: BoX比分機率 ----------
def boX_exact_score_probability(p, n_win, win_score, lose_score):
    total_games = win_score + lose_score
    if win_score < n_win or lose_score >= n_win:
        return 0
    ways = comb(total_games - 1, win_score - 1)
    return ways * (p ** win_score) * ((1 - p) ** lose_score)

### --------- Part 2: Likelihood（支援 log 空間與正則化） ----------
def compute_likelihood(params, matches, team_indices, use_log_space=True, reg_lambda=0.0):
    if use_log_space:
        strengths = np.exp(params)
    else:
        strengths = params

    total_log_likelihood = 0
    for match in matches:
        team1 = match['team1']
        team2 = match['team2']
        winner = match['winner']
        bo_str = match['bo']
        score1 = match['score1']
        score2 = match['score2']

        i, j = team_indices[team1], team_indices[team2]
        s_i, s_j = strengths[i], strengths[j]
        p_i = s_i / (s_i + s_j)

        if winner == team1:
            win_score = score1
            lose_score = score2
            p_win = p_i
        else:
            win_score = score2
            lose_score = score1
            p_win = 1 - p_i
            p_i = 1 - p_i

        if bo_str.lower() == "bo1":
            prob = p_win if (win_score == 1 and lose_score == 0) else 0
        elif bo_str.lower() == "bo3":
            prob = boX_exact_score_probability(p_i, 2, win_score, lose_score)
        elif bo_str.lower() == "bo5":
            prob = boX_exact_score_probability(p_i, 3, win_score, lose_score)
        elif bo_str.lower() == "bo7":
            prob = boX_exact_score_probability(p_i, 4, win_score, lose_score)
        elif bo_str.lower() == "bo9":
            prob = boX_exact_score_probability(p_i, 5, win_score, lose_score)

        else:
            raise ValueError(f"Unsupported match type: {bo_str}")

        total_log_likelihood += np.log(prob + 1e-12)

    if use_log_space:
        reg_term = reg_lambda * np.sum((params - 0) ** 2)  # params 是 log-strengths
    else:
        reg_term = reg_lambda * np.sum((params - 1) ** 2)

    return -total_log_likelihood + reg_term

### --------- Part 3: 強度估計（統一版本） ----------
def estimate_team_strengths(teams, matches, use_log_space=True, reg_lambda=0.01):
    N = len(teams)
    team_indices = {name: i for i, name in enumerate(teams)}

    if use_log_space:
        init_params = np.zeros(N)  # log-strengths 起始點
        result = minimize(
            compute_likelihood,
            init_params,
            args=(matches, team_indices, True, reg_lambda),
            method='L-BFGS-B'
        )
        strengths = np.exp(result.x)
    else:
        init_params = np.ones(N)
        bounds = [(1e-3, None) for _ in range(N)]
        result = minimize(
            compute_likelihood,
            init_params,
            args=(matches, team_indices, False, reg_lambda),
            bounds=bounds,
            method='L-BFGS-B'
        )
        strengths = result.x

    normalized = strengths / np.min(strengths)
    return dict(zip(teams, normalized))

### --------- 輸入資料讀取 ----------
def load_matches_from_csv(filename):
    matches = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        expected_fields = {"team1", "team2", "winner", "bo", "score1", "score2"}
        if set(reader.fieldnames) != expected_fields:
            raise ValueError(f"CSV 欄位名稱錯誤，應該是：{expected_fields}，但讀到的是：{reader.fieldnames}")
        for row in reader:
            matches.append({
                "team1": row["team1"],
                "team2": row["team2"],
                "winner": row["winner"],
                "bo": row["bo"],
                "score1": int(row["score1"]),
                "score2": int(row["score2"]),
            })
    return matches

def load_matches_from_json(filename):
    with open(filename, encoding='utf-8') as f:
        matches = json.load(f)
    return matches

### --------- 主程式 ----------
def main(filename, use_log_space=True, reg_lambda=0.01):
    if filename.endswith(".csv"):
        matches = load_matches_from_csv(filename)
    elif filename.endswith(".json"):
        matches = load_matches_from_json(filename)
    else:
        raise ValueError("Only CSV or JSON supported")

    team_set = set()
    for m in matches:
        team_set.update([m["team1"], m["team2"]])
    teams = sorted(list(team_set))

    strengths = estimate_team_strengths(teams, matches, use_log_space, reg_lambda)

    for team, value in sorted(strengths.items(), key=lambda x: -x[1]):
        print(f"{team}: {value:.3f}")

### --------- 執行入口 ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("用法: python StrengthEstimation.py matches.csv [log/raw] [lambda]")
    else:
        use_log = True if len(sys.argv) < 3 or sys.argv[2].lower() == "log" else False
        reg_lambda = float(sys.argv[3]) if len(sys.argv) == 4 else 0.01
        main(sys.argv[1], use_log, reg_lambda)

### Example
# python StrengthEstimation.py LOL_MSI_2025.csv log 0.05
# python StrengthEstimation.py LOL_MSI_2025.csv log 0.5
# python StrengthEstimation.py LOL_MSI_2025.csv raw
###
