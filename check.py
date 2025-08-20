# verify_gp_best_vs_enum.py
# 用途：把 GP 产出的 summary_best_results.csv 逐条读取，
#      将其中的 "Best_Join_Tree_Str" 用“min 简化规则 + Σ rows_out”重算，
#      并与 "Best_Cost" 对比验证是否一致。

import argparse
import ast
import math
import pandas as pd

# === 读取选择率与基表行数 ===
def load_selectivity_with_rows(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None)
    selectivity_map = {}
    table_rows = {}
    for _, row in df.iterrows():
        t1 = str(row[0]).strip()
        t2 = str(row[1]).strip()
        rows1 = int(row[2])
        rows2 = int(row[3])
        sel = float(row[4])
        key = frozenset([t1, t2])
        selectivity_map[key] = sel
        table_rows[t1] = rows1
        table_rows[t2] = rows2
    return selectivity_map, table_rows

# === NATURAL-join 的“min 简化规则”：跨边所有表对取最小 sel；无则 1.0 ===
def _sel_min_rule(left_subtree, right_subtree, selectivity_map):
    def tabs(t):
        if isinstance(t, str): return [t]
        return tabs(t[1]) + tabs(t[2])
    L = tabs(left_subtree)
    R = tabs(right_subtree)
    vals = []
    for l in L:
        for r in R:
            k = frozenset([l, r])
            if k in selectivity_map:
                vals.append(selectivity_map[k])
    return min(vals) if vals else 1.0

# === 计算 (final_rows, total_generated_rows) ，与全枚举脚本一致 ===
def sum_join_outputs(tree, selectivity_map, table_rows):
    if isinstance(tree, str):
        return table_rows.get(tree, 1), 0.0
    _, left, right = tree
    Lrows, Lsum = sum_join_outputs(left,  selectivity_map, table_rows)
    Rrows, Rsum = sum_join_outputs(right, selectivity_map, table_rows)
    sel = _sel_min_rule(left, right, selectivity_map)
    rows_out  = Lrows * Rrows * sel
    total_sum = Lsum + Rsum + rows_out
    return rows_out, total_sum

def read_gp_summary(summary_csv):
    """
    读取 GP 的 summary_best_results.csv
    期望列: SQL_File, Best_Cost, Best_Join_Tree_Str, Table_Order
    兼容无表头情形（按位置取前三列）
    """
    try:
        df = pd.read_csv(summary_csv)
        cols = [str(c).strip() for c in df.columns]
        df.columns = cols
        if {"SQL_File","Best_Cost","Best_Join_Tree_Str"} <= set(df.columns):
            df["SQL_File"] = df["SQL_File"].astype(str).str.strip()
            return df[["SQL_File","Best_Cost","Best_Join_Tree_Str"]].copy()
    except Exception:
        pass

    # 无表头或表头乱：fallback
    df = pd.read_csv(summary_csv, header=None)
    if df.shape[1] < 3:
        raise SystemExit(f"[ERROR] {summary_csv} 至少需要三列（file, cost, tree）")
    out = df.iloc[:, :3].copy()
    out.columns = ["SQL_File","Best_Cost","Best_Join_Tree_Str"]
    out["SQL_File"] = out["SQL_File"].astype(str).str.strip()
    return out

def main():
    ap = argparse.ArgumentParser(description="Verify GP best cost by recomputing with enum's min-rule + sum of rows_out")
    ap.add_argument("--summary", default="summary_best_results.csv", help="GP 输出的 summary_best_results.csv 路径")
    ap.add_argument("--excel",   default="Join-Selectivities.xlsx", help="Join-Selectivities.xlsx 路径")
    ap.add_argument("--sql",     default=None, help="只验证某一个文件名（如 10a.sql）。不填则验证所有行")
    ap.add_argument("--rel_tol", type=float, default=1e-12, help="浮点相对容差")
    ap.add_argument("--abs_tol", type=float, default=1e-15, help="浮点绝对容差")
    args = ap.parse_args()

    selectivity_map, table_rows = load_selectivity_with_rows(args.excel)
    df = read_gp_summary(args.summary)

    if args.sql:
        df = df[df["SQL_File"] == args.sql]
        if df.empty:
            print(f"[WARN] 在 {args.summary} 中找不到 {args.sql}")
            return

    total = 0
    ok = 0
    mismatches = []

    for _, row in df.iterrows():
        sql_file = str(row["SQL_File"])
        best_cost = float(row["Best_Cost"])
        tup_str = str(row["Best_Join_Tree_Str"])

        try:
            tree = ast.literal_eval(tup_str)
        except Exception as e:
            print(f"[ERROR] 解析 {sql_file} 的树失败：{e}")
            continue

        _, total_sum = sum_join_outputs(tree, selectivity_map, table_rows)

        match = math.isclose(best_cost, total_sum, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
        total += 1
        if match:
            ok += 1
            print(f"[OK] {sql_file}  GP={best_cost:.15g}  Recomputed={total_sum:.15g}")
        else:
            diff = total_sum - best_cost
            mismatches.append((sql_file, best_cost, total_sum, diff))
            print(f"[DIFF] {sql_file}  GP={best_cost:.15g}  Recomputed={total_sum:.15g}  Δ={diff:.15g}")

    print(f"\n验证完成：{ok}/{total} 条一致。")
    if mismatches:
        print("存在不一致的条目（前10条）：")
        for m in mismatches[:10]:
            print(f"  {m[0]}  GP={m[1]:.15g}  Recomputed={m[2]:.15g}  Δ={m[3]:.15g}")

if __name__ == "__main__":
    main()
