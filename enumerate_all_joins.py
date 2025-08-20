# enumerate_min_cost.py
import itertools
import csv
import math
import pandas as pd
import os
from copy import deepcopy
from SQLParser import parse_sql

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

# === 工具：获取子树所有叶子表名（左→右） ===
def get_all_tables(subtree):
    if isinstance(subtree, str):
        return [subtree]
    elif isinstance(subtree, tuple) and subtree[0] == "JOIN":
        return get_all_tables(subtree[1]) + get_all_tables(subtree[2])
    return []

# === NATURAL-join 的“min 简化规则”：跨边所有表对取最小 sel；无则 1.0 ===
def _sel_min_rule(left_subtree, right_subtree, selectivity_map):
    def tabs(t):
        if isinstance(t, str): return [t]
        return tabs(t[1]) + tabs(t[2])
    Ltabs = tabs(left_subtree)
    Rtabs = tabs(right_subtree)
    vals = []
    for l in Ltabs:
        for r in Rtabs:
            k = frozenset([l, r])
            if k in selectivity_map:
                vals.append(selectivity_map[k])
    return min(vals) if vals else 1.0

# === 计算：返回 (final_rows, total_generated_rows) ===
def sum_join_outputs(tree, selectivity_map, table_rows):
    """
    rows_out = Lrows * Rrows * sel
    final_rows: 根节点输出行数
    total_generated_rows: 树内所有 JOIN 的 rows_out 之和（优化/验证用）
    sel 采用 _sel_min_rule（无配对→1.0）
    """
    if isinstance(tree, str):
        return table_rows.get(tree, 1), 0.0

    _, left, right = tree
    Lrows, Lsum = sum_join_outputs(left,  selectivity_map, table_rows)
    Rrows, Rsum = sum_join_outputs(right, selectivity_map, table_rows)

    sel = _sel_min_rule(left, right, selectivity_map)
    rows_out  = Lrows * Rrows * sel
    total_sum = Lsum + Rsum + rows_out
    return rows_out, total_sum

# === 枚举所有“保持叶序”的二叉 JOIN 树（对给定排列） ===
def generate_all_join_trees(tables):
    if len(tables) == 1:
        return [tables[0]]
    trees = []
    # 所有可能的二分点
    for i in range(1, len(tables)):
        lefts = generate_all_join_trees(tables[:i])
        rights = generate_all_join_trees(tables[i:])
        for l in lefts:
            for r in rights:
                trees.append(("JOIN", deepcopy(l), deepcopy(r)))
    return trees

# === 把 JOIN 树转成 'join_tables(...)' 可读字符串 ===
def join_tree_to_str(t):
    if isinstance(t, str):
        return t
    return f"join_tables({join_tree_to_str(t[1])}, {join_tree_to_str(t[2])})"

# === 读取全部 SQL（至少需要 FROM 表列表）===
def load_all_sql_queries(folder_path):
    sql_queries = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".sql"):
            filepath = os.path.join(folder_path, filename)
            try:
                query_info = parse_sql(filepath)
                sql_queries[filename] = query_info
            except Exception as e:
                print(f"[ERROR] Failed to parse {filename}: {e}")
    return sql_queries

# === 主流程：全枚举验证 ===
if __name__ == "__main__":
    selectivity_map, table_rows = load_selectivity_with_rows("Join-Selectivities.xlsx")
    all_queries = load_all_sql_queries("TestQueries")

    summary_results = []

    # 浮点比较容差
    REL_TOL = 1e-12
    ABS_TOL = 1e-15

    for filename, query_info in all_queries.items():
        table_list = list(dict.fromkeys(query_info["FROM"]))

        if len(table_list) < 2:
            print(f"[枚举跳过] {filename}: 表数 < 2")
            continue
        # 为防爆炸，这里限制最多 7 张表；可按需改为 8（会非常慢）
        if len(table_list) > 7:
            print(f"[枚举跳过] {filename}: 表数 = {len(table_list)} (>7)，全枚举代价太大")
            continue

        print(f"开始枚举 Query: {filename}  表: {table_list}")

        all_min_trees = []
        min_cost = float("inf")

        # 枚举所有叶序排列 + 在该叶序下的所有二叉树形
        for perm in itertools.permutations(table_list):
            join_trees = generate_all_join_trees(list(perm))
            for tree in join_trees:
                try:
                    final_rows, total_sum = sum_join_outputs(tree, selectivity_map, table_rows)
                    cost = total_sum  # 以 Σ rows_out 为代价
                    if cost < min_cost and not math.isclose(cost, min_cost, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                        min_cost = cost
                        all_min_trees = [(perm, deepcopy(tree), cost, final_rows)]
                    elif math.isclose(cost, min_cost, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                        all_min_trees.append((perm, deepcopy(tree), cost, final_rows))
                except Exception as e:
                    print(f"[WARN] Tree {tree} error: {e}")

        # 去重（同形同序只保留一次）
        unique_strs = set()
        final_results = []
        for perm, tree, cost, final_rows in all_min_trees:
            s = join_tree_to_str(tree)
            if s not in unique_strs:
                unique_strs.add(s)
                final_results.append((cost, final_rows, s, ",".join(perm)))

        # 输出每个 SQL 的所有最优解
        out_dir = "enumerate_results"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{filename}_all_min_cost_join_trees.csv")
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["MinCost_SumRows", "FinalRows_at_Root", "Join_Tree_Str", "Leaf_Order"])
            for row in final_results:
                writer.writerow(row)

        print(f"Query {filename} 最小总代价(Σ rows_out): {min_cost}")
        print(f"共有 {len(final_results)} 种不同的最优 JOIN 树，已写入 {out_file}\n")

        # 汇总：记录每个 SQL 的最小总代价与对应叶序（取第一条代表）
        if final_results:
            best_row = final_results[0]
            summary_results.append([filename, best_row[0], best_row[1], best_row[3]])

    # 汇总所有 SQL 的最小总代价
    summary_file = "summary_min_cost.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SQL_File", "MinCost_SumRows", "FinalRows_at_Root", "Leaf_Order"])
        for row in summary_results:
            writer.writerow(row)

    print(f"✅ 全部完成。汇总已写入 {summary_file}")
