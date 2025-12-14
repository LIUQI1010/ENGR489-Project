# -*- coding: utf-8 -*-
import os
import random
import operator
import csv
import pandas as pd
from SQLParser import parse_sql
from deap import base, creator, gp, tools, algorithms
from copy import deepcopy

# =========================
#  选择率与行数加载（保持不变）
# =========================
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

# =========================
#  GP 基元：JOIN 树（保持不变）
# =========================
def join_tables(a, b):
    return ("JOIN", a, b)

def create_pset(table_list):
    pset = gp.PrimitiveSet("MAIN", 0)
    pset.addPrimitive(join_tables, 2, name="join_tables")
    for table in table_list:
        pset.addTerminal(table, name=str(table))
    return pset

# 随机生成保持“每表恰好一次”的初始 JOIN 树（随机形状）（保持不变）
def generate_expr_unique_terminals(pset, table_list, rng):
    shuffled = rng.sample(table_list, len(table_list))

    def build_tree(tables):
        if len(tables) == 1:
            return tables[0]
        split = rng.randint(1, len(tables) - 1)
        left = build_tree(tables[:split])
        right = build_tree(tables[split:])
        return ("JOIN", left, right)

    def tree_to_str(t):
        if isinstance(t, str):
            return t
        return f"join_tables({tree_to_str(t[1])}, {tree_to_str(t[2])})"

    tree = build_tree(shuffled)
    expr_str = tree_to_str(tree)
    return gp.PrimitiveTree.from_string(expr_str, pset)


def generate_expr_connected(pset, table_list, allowed_edges, rng):
    # 随机形状使用局部 rng
    base = generate_expr_unique_terminals(pset, table_list, rng)
    shape = parse_gp_str(str(base))
    if len(leaves_inorder(shape)) != len(table_list):
        return base  # 退回原始形状
    # 叶序使用确定性贪心
    order = build_greedy_connected_order(table_list, allowed_edges)
    expr_str = tuple_to_expr_str(fill_leaves_with_order(shape, iter(order)))
    return gp.PrimitiveTree.from_string(expr_str, pset)

# =========================
#  树解析/重建/叶序（保持不变）
# =========================
def parse_gp_str(expr_str: str):
    expr_str = expr_str.strip()
    if not expr_str.startswith("join_tables"):
        return expr_str
    inside = expr_str[len("join_tables("):-1]
    depth = 0
    split_index = None
    for i, ch in enumerate(inside):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            split_index = i
            break
    left_str = inside[:split_index].strip()
    right_str = inside[split_index+1:].strip()
    return ("JOIN", parse_gp_str(left_str), parse_gp_str(right_str))

def tuple_to_expr_str(t):
    if isinstance(t, str):
        return t
    _, l, r = t
    return f"join_tables({tuple_to_expr_str(l)}, {tuple_to_expr_str(r)})"

def leaves_inorder(t):
    if isinstance(t, str):
        return [t]
    return leaves_inorder(t[1]) + leaves_inorder(t[2])

def get_all_tables(subtree):
    if isinstance(subtree, str):
        return [subtree]
    if isinstance(subtree, tuple) and subtree[0] == "JOIN":
        return get_all_tables(subtree[1]) + get_all_tables(subtree[2])
    return []

def fill_leaves_with_order(shape_tuple, order_iter):
    if isinstance(shape_tuple, str):
        return next(order_iter)
    _, l, r = shape_tuple
    return ("JOIN",
            fill_leaves_with_order(l, order_iter),
            fill_leaves_with_order(r, order_iter))

def get_leaf_order_from_ind(individual):
    return leaves_inorder(parse_gp_str(str(individual)))

# =========================
#  变异（保持为“与父旋转”的 mutate2）
# =========================
def mutate_individual(individual, pset, table_list, max_tries=10):
    """
    论文的 mutate2 思想：随机挑一个内部节点，与其父节点做一次旋转（左旋/右旋），
    结构微调，不改变叶集合。失败则原样返回。
    """
    def _get_sub(t, path):
        if not path:
            return t
        _, l, r = t
        return _get_sub(l, path[1:]) if path[0] == 'L' else _get_sub(r, path[1:])

    def _set_sub(t, path, new_sub):
        if not path:
            return new_sub
        _, l, r = t
        if path[0] == 'L':
            return ("JOIN", _set_sub(l, path[1:], new_sub), r)
        else:
            return ("JOIN", l, _set_sub(r, path[1:], new_sub))

    def _list_internal_paths(t, path=()):
        if isinstance(t, str):
            return []
        _, l, r = t
        paths = [path]
        paths += _list_internal_paths(l, path + ('L',))
        paths += _list_internal_paths(r, path + ('R',))
        return paths

    def _rotate_with_parent(t, target_path):
        if not target_path:
            return None  # 根无父
        parent_path = target_path[:-1]
        is_left = (target_path[-1] == 'L')
        parent = _get_sub(t, parent_path)
        if isinstance(parent, str):
            return None
        _, A, B = parent  # parent = ("JOIN", A, B)

        target = A if is_left else B
        if isinstance(target, str):
            return None
        _, x, y = target  # target = ("JOIN", x, y)

        if is_left:
            # 右旋：( (x,y), B ) -> ( x, (y,B) )
            new_parent = ("JOIN", x, ("JOIN", y, B))
        else:
            # 左旋：( A, (x,y) ) -> ( (A,x), y )
            new_parent = ("JOIN", ("JOIN", A, x), y)

        return _set_sub(t, parent_path, new_parent)

    t = parse_gp_str(str(individual))
    internal_paths = [p for p in _list_internal_paths(t) if len(p) > 0]  # 非根
    if not internal_paths:
        return creator.Individual(gp.PrimitiveTree(individual)),

    tries = 0
    while tries < max_tries:
        tries += 1
        p = random.choice(internal_paths)
        new_t = _rotate_with_parent(t, p)
        if new_t is not None and tuple_to_expr_str(new_t) != tuple_to_expr_str(t):
            expr = gp.PrimitiveTree.from_string(tuple_to_expr_str(new_t), pset)
            return creator.Individual(expr),
    return creator.Individual(gp.PrimitiveTree(individual)),

# =========================
#  论文式 φ-交叉：工具函数
# =========================
def leaves_set(node):
    if isinstance(node, str):
        return {node}
    _, l, r = node
    return leaves_set(l) | leaves_set(r)

def postorder_join_descriptors(t):
    """
    返回按后序的 join 描述列表：每个元素是 (Lset, Rset)，
    其中 Lset/Rset 是该 join 左/右子树的叶集合（集合对象、可作匹配键）。
    """
    out = []
    def rec(n):
        if isinstance(n, str):
            return
        _, l, r = n
        rec(l); rec(r)
        out.append((frozenset(leaves_set(l)), frozenset(leaves_set(r))))
    rec(t)
    return out

def descriptors_for_subtree(t):
    return set(postorder_join_descriptors(t))

def phi_rebuild(nodelist, T_set):
    """
    论文 φ 算子：逐步从 T_set 取出与 (Lset,Rset) 匹配的两棵树，组合为新 JOIN 树。
    """
    def keyset(tree):
        return frozenset(leaves_set(tree))

    pool = {}
    def push(tree):
        pool.setdefault(keyset(tree), []).append(tree)
    def pop_exact(key):
        lst = pool.get(key, [])
        if not lst:
            return None
        return lst.pop()

    for t in T_set:
        push(t)

    for (Lset, Rset) in nodelist:
        A = pop_exact(Lset)
        B = pop_exact(Rset)
        if A is None or B is None:
            return None
        push(("JOIN", A, B))

    # 最后应只剩 1 棵树
    leftover = None
    for lst in pool.values():
        for el in lst:
            if leftover is None:
                leftover = el
            else:
                return None
    return leftover

def pick_random_proper_subtree(t):
    subs = []
    def rec(n, is_root):
        if isinstance(n, str):
            return
        _, l, r = n
        if not is_root:
            subs.append(n)
        rec(l, False); rec(r, False)
    rec(t, True)
    if not subs:
        return None
    return random.choice(subs)

# =========================
#  连通/有效性相关（保持不变）
# =========================
def _tabs(node):
    if isinstance(node, str):
        return [node]
    return _tabs(node[1]) + _tabs(node[2])

def build_allowed_edges(selectivity_map, table_list):
    tables = set(table_list)
    return {pair for pair in selectivity_map.keys() if all(t in tables for t in pair)}

def _has_allowed_edge_between_sets(left_tabs, right_tabs, allowed_edges):
    for l in left_tabs:
        for r in right_tabs:
            if frozenset([l, r]) in allowed_edges:
                return True
    return False

def count_disconnected_joins(tree, allowed_edges):
    if isinstance(tree, str):
        return 0
    _, L, R = tree
    left_tabs = _tabs(L); right_tabs = _tabs(R)
    bad = 0 if _has_allowed_edge_between_sets(left_tabs, right_tabs, allowed_edges) else 1
    return bad + count_disconnected_joins(L, allowed_edges) + count_disconnected_joins(R, allowed_edges)

# =========================
#  论文式 φ-交叉（严格版）：始终产出“有效树”
# =========================
def gp_crossover_phi_strict(ind1, ind2, pset, allowed_edges,
                            max_tries=40, min_leaf_count=2):
    """
    论文定义：
      NG1 := φ( postorder(T1) - postorder(S2), {S2} ∪ (leaves(T1) - leaves(S2)) )
      NG2 := φ( postorder(T2) - postorder(S1), {S1} ∪ (leaves(T2) - leaves(S1)) )
    严格性：重复采样 S1,S2，确保 φ 成功且 new 树没有“断连 join”（不产生 cross product）。
    多次失败则返回父代（no-op）。
    """
    T1 = parse_gp_str(str(ind1))
    T2 = parse_gp_str(str(ind2))

    # 叶集合必须一致（你的初始化已保证）
    if frozenset(leaves_inorder(T1)) != frozenset(leaves_inorder(T2)):
        return creator.Individual(gp.PrimitiveTree(ind1)), creator.Individual(gp.PrimitiveTree(ind2))

    def _valid_leafset(tree):
        used = get_all_tables(tree)
        return len(used) == len(set(used)) and set(used) == set(leaves_inorder(T1))

    tries = 0
    while tries < max_tries:
        tries += 1

        S1 = pick_random_proper_subtree(T1)
        S2 = pick_random_proper_subtree(T2)
        if (S1 is None) or (S2 is None):
            continue
        if min_leaf_count is not None:
            if len(leaves_set(S1)) < min_leaf_count or len(leaves_set(S2)) < min_leaf_count:
                continue

        desc_T1 = postorder_join_descriptors(T1)
        desc_S2 = descriptors_for_subtree(S2)
        nodelist1 = [d for d in desc_T1 if d not in desc_S2]
        leaves_T1 = leaves_set(T1)
        leaves_S2 = leaves_set(S2)
        Tset1 = [S2] + [t for t in leaves_T1 - leaves_S2]

        desc_T2 = postorder_join_descriptors(T2)
        desc_S1 = descriptors_for_subtree(S1)
        nodelist2 = [d for d in desc_T2 if d not in desc_S1]
        leaves_T2 = leaves_set(T2)
        leaves_S1 = leaves_set(S1)
        Tset2 = [S1] + [t for t in leaves_T2 - leaves_S1]

        NG1_tuple = phi_rebuild(nodelist1, Tset1)
        NG2_tuple = phi_rebuild(nodelist2, Tset2)

        if (NG1_tuple is None) or (NG2_tuple is None):
            continue
        if not (_valid_leafset(NG1_tuple) and _valid_leafset(NG2_tuple)):
            continue
        # 严格：不允许断连（等价于论文“无 cross product / 无人工 join”的有效性要求）
        if count_disconnected_joins(NG1_tuple, allowed_edges) != 0:
            continue
        if count_disconnected_joins(NG2_tuple, allowed_edges) != 0:
            continue

        expr1 = gp.PrimitiveTree.from_string(tuple_to_expr_str(NG1_tuple), pset)
        expr2 = gp.PrimitiveTree.from_string(tuple_to_expr_str(NG2_tuple), pset)
        return creator.Individual(expr1), creator.Individual(expr2)

    # 多次尝试失败：不做交叉（与论文“保持有效性”的精神一致）
    return creator.Individual(gp.PrimitiveTree(ind1)), creator.Individual(gp.PrimitiveTree(ind2))

# =========================
#  代价构件：选择率乘积 / 不连通惩罚（保持不变）
# =========================
def _crossing_selectivity_product(left_subtree, right_subtree, selectivity_map, allowed_edges):
    L = _tabs(left_subtree); R = _tabs(right_subtree)
    sel = 1.0
    has = False
    for l in L:
        for r in R:
            k = frozenset([l, r])
            if k in allowed_edges:
                has = True
                sel *= selectivity_map.get(k, 1.0)
    return sel if has else None

def _anchor_penalty(left_subtree, right_subtree):
    """ domain-agnostic: 不对特定表做惩罚，返回 1.0 """
    return 1.0

def estimate_cost(tree, selectivity_map, table_rows, allowed_edges=None, disconnect_penalty=1e6):
    if isinstance(tree, str):
        return table_rows.get(tree, 1)
    _, left, right = tree
    Lrows = estimate_cost(left, selectivity_map, table_rows, allowed_edges, disconnect_penalty)
    Rrows = estimate_cost(right, selectivity_map, table_rows, allowed_edges, disconnect_penalty)
    if allowed_edges is None:
        allowed_edges = build_allowed_edges(selectivity_map, _tabs(tree))
    sel = _crossing_selectivity_product(left, right, selectivity_map, allowed_edges)
    if sel is None:
        rows_out = Lrows * Rrows * disconnect_penalty
    else:
        rows_out = Lrows * Rrows * sel
        rows_out *= _anchor_penalty(left, right)
    return rows_out

def sum_join_outputs(tree, selectivity_map, table_rows, allowed_edges=None, disconnect_penalty=1e6):
    """
    返回：
      final_rows            — 子树最终输出行数
      total_generated_rows  — 所有 JOIN 节点 rows_out 之和（作为 fitness）
    """
    if isinstance(tree, str):
        return table_rows.get(tree, 1), 0.0

    _, left, right = tree
    Lrows, Lsum = sum_join_outputs(left,  selectivity_map, table_rows, allowed_edges, disconnect_penalty)
    Rrows, Rsum = sum_join_outputs(right, selectivity_map, table_rows, allowed_edges, disconnect_penalty)

    if allowed_edges is None:
        all_tabs = _tabs(tree)
        allowed_edges = build_allowed_edges(selectivity_map, all_tabs)

    sel = _crossing_selectivity_product(left, right, selectivity_map, allowed_edges)
    if sel is None:
        rows_out = Lrows * Rrows * disconnect_penalty
    else:
        rows_out = Lrows * Rrows * sel
        rows_out *= _anchor_penalty(left, right)

    total_sum = Lsum + Rsum + rows_out
    return rows_out, total_sum

# =========================
#  SQL 解析 / 基数前移近似（保持不变）
# =========================
def load_all_sql_queries(folder_path):
    sql_queries = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".sql"):
            filepath = os.path.join(folder_path, filename)
            try:
                query_info = parse_sql(filepath)  # 需要至少 "FROM"
                sql_queries[filename] = query_info
            except Exception as e:
                print(f"[ERROR] Failed to parse {filename}: {e}")
    return sql_queries

def _collect_condition_text(query_info):
    parts = []
    for key in ("WHERE", "where", "FILTERS", "filters", "CONDITIONS"):
        v = query_info.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(map(str, v))
    for key in ("JOIN", "joins"):
        joins = query_info.get(key)
        if isinstance(joins, list):
            for j in joins:
                on = j.get("on") if isinstance(j, dict) else None
                if on:
                    parts.append(str(on))
    return " ".join(parts)

def apply_base_filters(table_rows, query_info):
    """ 不再基于 WHERE 文本进行行数缩放，直接返回原始行数。 """
    return dict(table_rows)

# =========================
#  连通初始化/修复 相关辅助（保持不变）
# =========================
def build_greedy_connected_order(table_list, allowed_edges, start=None):
    # 按 SQL 解析顺序保序去重
    tabs = list(dict.fromkeys(table_list))
    if not tabs:
        return []

    # 计算度数（稳定遍历）
    deg = {t: 0 for t in tabs}
    for a in tabs:
        for b in tabs:
            if a != b and frozenset([a, b]) in allowed_edges:
                deg[a] += 1

    # 选起点：度数最大；并列按 tabs 顺序打破
    if start in deg:
        cur = start
    else:
        max_deg = max(deg.values()) if deg else 0
        cur = next((t for t in tabs if deg[t] == max_deg), tabs[0])

    used = [cur]
    # remain 用“按 tabs 顺序”的列表，不用 set
    remain = [t for t in tabs if t != cur]

    while remain:
        # 候选按 remain 的稳定顺序
        candidates = [t for t in remain
                      if any(frozenset([t, u]) in allowed_edges for u in used)]
        if candidates:
            # 选连接数最多的；并列保留候选出现顺序
            best_score = -1
            best_idx = 0
            for i, t in enumerate(candidates):
                score = sum(1 for u in used if frozenset([t, u]) in allowed_edges)
                if score > best_score:
                    best_score = score
                    best_idx = i
            next_t = candidates[best_idx]
        else:
            # 以前是 random.choice(...)；现在取第一个，保证确定性
            next_t = remain[0]

        used.append(next_t)
        remain = [t for t in remain if t != next_t]

    return used


def repair_individual_connected(individual, pset, table_list, allowed_edges):
    shape = parse_gp_str(str(individual))
    if isinstance(shape, str):
        return creator.Individual(gp.PrimitiveTree(individual))
    shape_leaves = leaves_inorder(shape)
    if len(shape_leaves) != len(table_list):
        return creator.Individual(gp.PrimitiveTree(individual))
    if count_disconnected_joins(shape, allowed_edges) == 0:
        return creator.Individual(gp.PrimitiveTree(individual))
    order = build_greedy_connected_order(table_list, allowed_edges)
    fixed_tuple = fill_leaves_with_order(shape, iter(order))
    expr = gp.PrimitiveTree.from_string(tuple_to_expr_str(fixed_tuple), pset)
    return creator.Individual(expr)

# =========================
#  优化主流程（保持不变，仅 mate 改为严格 φ-交叉）
# =========================
def optimize_query(query_info, selectivity_map, table_rows, filename="(unknown)",
                   seed=None, summary_file="summary_best_results_seeds.csv", log_dir="logs_seeds"):
    # ---- 控制参数（保持不变） ----
    EARLY_STOP = False
    PATIENCE = 15
    MIN_DELTA_REL = 1e-3
    MIN_DELTA_ABS = 1e-6

    PENALTY_START = 1e5
    PENALTY_END   = 1e6
    ANNEAL_UNTIL_GEN = 40

    POP_SIZE = 50
    NGEN = 50
    ELITE_NUM = 2
    CXPB = 0.3
    MUTPB = 0.6
    TOURN = 2

    if seed is not None:
        random.seed(seed)

    table_list = list(dict.fromkeys(query_info["FROM"]))  # 去重
    if len(table_list) < 2:
        print(f"Query {filename} has less than 2 tables. Skipping.")
        return

    base_rows = apply_base_filters(table_rows, query_info)
    allowed_edges = build_allowed_edges(selectivity_map, table_list)

    pset = create_pset(table_list)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("clone", deepcopy)
    # 初始化：尽量连通（保持不变）
    import hashlib

    # 为每个初始个体分配稳定的局部 RNG：受 (filename, seed, 个体序号) 决定
    _ind_counter = {"i": 0}

    def _stable_subseed(label: str, fname: str, master_seed: int, idx: int) -> int:
        s = f"{label}|{fname}|{master_seed}|{idx}".encode("utf-8")
        h = hashlib.md5(s).digest()  # 稳定哈希，不受 PYTHONHASHSEED 影响
        return int.from_bytes(h[:4], "big")  # 取 32bit 子种子

    def _expr_wrapper():
        i = _ind_counter["i"]
        _ind_counter["i"] += 1
        local_seed = _stable_subseed("INIT", filename, int(seed if seed is not None else 0), i)
        rng = random.Random(local_seed)
        return generate_expr_connected(pset, table_list, allowed_edges, rng)

    toolbox.register("expr", _expr_wrapper)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    current_penalty = PENALTY_START

    def _valid_leafset(tree):
        used = get_all_tables(tree)
        return len(used) == len(table_list) and set(used) == set(table_list)

    def _try_repair_leafset(tree):
        """尽量修复为每表恰好一次；失败返回 None。"""
        try:
            order = build_greedy_connected_order(table_list, allowed_edges)
            fixed = fill_leaves_with_order(tree, iter(order))
            if _valid_leafset(fixed):
                return fixed
            fixed2 = fill_leaves_with_order(tree, iter(table_list))
            if _valid_leafset(fixed2):
                return fixed2
        except Exception:
            pass
        return None

    def eval_cost(individual):
        try:
            tree = parse_gp_str(str(individual))

            if not _valid_leafset(tree):
                repaired = _try_repair_leafset(tree)
                if repaired is None:
                    return (1e300,)
                tree = repaired

            _, total_sum = sum_join_outputs(
                tree, selectivity_map, base_rows,
                allowed_edges=allowed_edges,
                disconnect_penalty=current_penalty
            )
            if not (total_sum == total_sum) or total_sum == float("inf") or total_sum == float("-inf"):
                return (1e300,)
            return (float(total_sum),)
        except Exception as e:
            s = str(individual)
            print(f"[EVAL-ERR] {filename} seed={seed}: {e} | ind[:120]={s[:120]}")
            return (1e300,)

    toolbox.register("evaluate", eval_cost)
    toolbox.register("select", tools.selTournament, tournsize=TOURN)

    # === mate：改为“严格 φ-交叉”，不再回退到 cxOnePoint；失败则 no-op ===
    toolbox.register("mate", gp_crossover_phi_strict, pset=pset, allowed_edges=allowed_edges)

    toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
    toolbox.register("mutate", mutate_individual, pset=pset, table_list=table_list)

    print(f"\n🧪 Optimizing Query: {filename} | Seed: {seed}  (Tables: {table_list})")
    pop = toolbox.population(n=POP_SIZE)

    # 初始化后做一次连通修复 + 合法性保障（保持不变）
    def _legalize(ind):
        try:
            ind2 = repair_individual_connected(ind, pset, table_list, allowed_edges)
        except Exception:
            ind2 = creator.Individual(gp.PrimitiveTree(ind))
        t = parse_gp_str(str(ind2))
        if not _valid_leafset(t):
            return creator.Individual(gp.PrimitiveTree(ind))
        return ind2

    pop = [_legalize(ind) for ind in pop]

    hof = tools.HallOfFame(1)

    # 初评
    fits0 = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits0):
        ind.fitness.values = fit
    hof.update(pop)

    # === 记录 Generation 0（初始化种群）的最佳个体与 cost ===
    log_data = []  # 确保列表已初始化
    # 记录第0代种群中的最优解
    gen0_best = min(pop, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'))
    gen0_best_cost = gen0_best.fitness.values[0] if gen0_best.fitness.valid else toolbox.evaluate(gen0_best)[0]
    raw0 = parse_gp_str(str(gen0_best))
    
    # 与后续 logging 对齐：如叶集合不合法，尝试修复一次
    if not _valid_leafset(raw0):
        fixed0 = _try_repair_leafset(raw0)
        gen0_best_tree = fixed0 if fixed0 is not None else raw0
    else:
        gen0_best_tree = raw0
        
    log_data.append(
        (seed, 0, gen0_best_cost, tuple_to_expr_str(gen0_best_tree), ",".join(get_all_tables(gen0_best_tree)))
    )
    mu = len(pop)
    lmbda = mu

    best_seen = float("inf")
    no_improve = 0

    def _anneal_penalty(gen_idx):
        if gen_idx >= ANNEAL_UNTIL_GEN:
            return PENALTY_END
        t = gen_idx / float(ANNEAL_UNTIL_GEN)
        return PENALTY_START + (PENALTY_END - PENALTY_START) * t

    for gen in range(1, NGEN + 1):
        current_penalty = _anneal_penalty(gen)

        parents = toolbox.select(pop, k=mu)
        offspring = algorithms.varOr(parents, toolbox, lambda_=lmbda, cxpb=CXPB, mutpb=MUTPB)

        # 交叉/变异后：连通修复 + 合法化（保持不变）
        new_offspring = []
        for ind in offspring:
            try:
                ind = repair_individual_connected(ind, pset, table_list, allowed_edges)
            except Exception:
                ind = creator.Individual(gp.PrimitiveTree(ind))
            t = parse_gp_str(str(ind))
            if not _valid_leafset(t):
                fixed = _try_repair_leafset(t)
                if fixed is not None:
                    ind = creator.Individual(gp.PrimitiveTree.from_string(tuple_to_expr_str(fixed), pset))
                else:
                    ind = creator.Individual(gp.PrimitiveTree(ind))
            new_offspring.append(ind)
        offspring = new_offspring

        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop = offspring

        # 精英注入（保持不变）
        if len(hof) > 0:
            inj = min(ELITE_NUM, len(hof), len(pop))
            pop[:inj] = [deepcopy(h) for h in hof[:inj]]

        hof.update(pop)
        
        # 记录当前代种群中的最优解（而不是历史最优）
        current_best = min(pop, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'))
        current_best_cost = current_best.fitness.values[0] if current_best.fitness.valid else toolbox.evaluate(current_best)[0]
        raw = parse_gp_str(str(current_best))
        if not _valid_leafset(raw):
            fixed = _try_repair_leafset(raw)
            current_best_tuple = fixed if fixed is not None else raw
        else:
            current_best_tuple = raw

        current_best_tables = get_all_tables(current_best_tuple)
        log_data.append((seed, gen, current_best_cost, tuple_to_expr_str(current_best_tuple), ",".join(current_best_tables)))

        # Early stopping （保持不变；默认不启用）
        if EARLY_STOP:
            import math
            if best_seen is None or not math.isfinite(best_seen):
                best_seen = current_best_cost
                no_improve = 0
            else:
                threshold = max(MIN_DELTA_ABS, abs(best_seen) * MIN_DELTA_REL)
                improved = current_best_cost < (best_seen - threshold)
                if improved:
                    best_seen = current_best_cost
                    no_improve = 0
                else:
                    MIN_EARLY_GEN = 10
                    if gen >= MIN_EARLY_GEN:
                        no_improve += 1
                        if no_improve >= PATIENCE:
                            print(f"[EARLY-STOP] {filename} seed={seed} at gen {gen}, best={best_seen:.6g}")
                            break

    # 写日志（结构不变）
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{filename}_seed{seed}_log.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "Generation", "Best_Cost", "Best_Join_Tree_Str", "Used_Tables"])
        writer.writerows(log_data)

    # 汇总（结构不变）
    new_summary = not os.path.exists(summary_file)
    best = hof[0]
    raw = parse_gp_str(str(best))
    if not _valid_leafset(raw):
        fixed = _try_repair_leafset(raw)
        best_tree_tuple = fixed if fixed is not None else raw
    else:
        best_tree_tuple = raw
    best_cost = eval_cost(best)[0] if not best.fitness.valid else best.fitness.values[0]
    best_tables = get_all_tables(best_tree_tuple)
    tree_str = tuple_to_expr_str(best_tree_tuple)
    with open(summary_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_summary:
            writer.writerow(["SQL_File", "Seed", "Best_Cost", "Best_Join_Tree_Str", "Table_Order"])
        writer.writerow([filename, seed, best_cost, tree_str, ",".join(best_tables)])

# =========================
#  主执行（输出文件名保持“baseline”）
# =========================
if __name__ == "__main__":
    # 注意：不固定全局随机种子；每次运行都会抽取新的 30 个 seed
    selectivity_map, table_rows = load_selectivity_with_rows("Join-Selectivities.xlsx")
    all_queries = load_all_sql_queries("TestQueries")

    # 使用固定的 seeds_used.txt 文件中的 seed
    seeds = []
    if os.path.exists("seeds_used.txt"):
        with open("seeds_used.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(int(line))
        print(f"使用 seeds_used.txt 中的 {len(seeds)} 个种子")
    else:
        print("错误: seeds_used.txt 文件不存在，请确保该文件存在并包含种子值")
        exit(1)

    # 多 seed × 多查询运行 —— baseline 区分文件名不变
    for filename, query_info in all_queries.items():
        for s in seeds:
            try:
                optimize_query(
                    query_info, selectivity_map, table_rows,
                    filename=filename,
                    seed=s,
                    summary_file="summary_best_results_seeds_baseline.csv",
                    log_dir="logs_seeds_baseline"
                )
            except Exception as e:
                print(f"[BASELINE-ERROR] Optimizing {filename} with seed {s} failed: {e}")