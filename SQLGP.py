import os
import random
import operator
import csv
import pandas as pd
from SQLParser import parse_sql
from deap import base, creator, gp, tools, algorithms
from copy import deepcopy

# =========================
#  选择率与行数加载
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
#  GP 基元：JOIN 树
# =========================
def join_tables(a, b):
    return ("JOIN", a, b)

def create_pset(table_list):
    pset = gp.PrimitiveSet("MAIN", 0)
    pset.addPrimitive(join_tables, 2, name="join_tables")
    for table in table_list:
        pset.addTerminal(table, name=str(table))
    return pset

# 随机生成保持“每表恰好一次”的初始 JOIN 树
def generate_expr_unique_terminals(pset, table_list):
    shuffled = random.sample(table_list, len(table_list))
    def build_tree(tables):
        if len(tables) == 1:
            return tables[0]
        split = random.randint(1, len(tables) - 1)
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

# =========================
#  树解析/重建/叶序
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
#  交叉/变异（你的原逻辑保留）
# =========================
def mutate_individual(individual, pset, table_list, max_tries=10):
    ind = gp.PrimitiveTree(individual)
    internal_idxs = [i for i, node in enumerate(ind) if getattr(node, "arity", 0) > 0]
    if not internal_idxs:
        return creator.Individual(ind),
    idx = random.choice(internal_idxs)
    sl = ind.searchSubtree(idx)
    sub_terms = [node.name for node in ind[sl] if getattr(node, "arity", 0) == 0]
    sub_terms = list(dict.fromkeys(sub_terms))
    if len(sub_terms) <= 1:
        return creator.Individual(ind),
    def build_expr(names):
        if len(names) == 1:
            return names[0]
        split = random.randint(1, len(names) - 1)
        left = build_expr(names[:split])
        right = build_expr(names[split:])
        return f"join_tables({left}, {right})"
    tries = 0
    while tries < max_tries:
        tries += 1
        new_order = random.sample(sub_terms, len(sub_terms))
        if new_order == sub_terms:
            new_order = sub_terms[1:] + sub_terms[:1]
        expr_str = build_expr(new_order)
        new_sub = gp.PrimitiveTree.from_string(expr_str, pset)
        test_ind = gp.PrimitiveTree(ind)
        test_ind[sl] = new_sub
        if str(test_ind) != str(ind):
            ind = test_ind
            break
    return creator.Individual(ind),

def ox_crossover(order1, order2):
    size = len(order1)
    a, b = sorted(random.sample(range(size), 2))
    hole = set(order1[a:b])
    res = [None] * size
    res[a:b] = order1[a:b]
    idx = b
    for elem in order2:
        if elem not in hole:
            if idx >= size:
                idx = 0
            res[idx] = elem
            idx += 1
    return res

def internal_subtrees(expr_tuple):
    lst = []
    def rec(node):
        if isinstance(node, str):
            return
        lst.append(node)
        rec(node[1]); rec(node[2])
    rec(expr_tuple)
    return lst

def pick_random_subtree_span(expr_tuple, *, forbid_full=True):
    internals = internal_subtrees(expr_tuple)
    full_leaves = leaves_inorder(expr_tuple)
    total = len(full_leaves)

    if forbid_full:
        candidates = []
        for sub in internals:
            if len(leaves_inorder(sub)) < total:
                candidates.append(sub)
    else:
        candidates = internals[:]

    if not candidates:
        a = random.randrange(total)
        return (a, a+1), {full_leaves[a]}

    sub = random.choice(candidates)
    sub_leaves = leaves_inorder(sub)
    a = full_leaves.index(sub_leaves[0])
    b = a + len(sub_leaves)
    return (a, b), set(sub_leaves)

def ox_crossover_with_span(order1, order2, span):
    size = len(order1)
    a, b = span
    hole = set(order1[a:b])
    res = [None] * size
    res[a:b] = order1[a:b]
    idx = b
    for e in order2:
        if e not in hole:
            if idx >= size:
                idx = 0
            res[idx] = e
            idx += 1
    return res, span

def enumerate_subtrees_with_paths(t):
    out = []
    def rec(node, path):
        if isinstance(node, str):
            out.append((path, node, frozenset([node])))
            return frozenset([node])
        _, l, r = node
        L = rec(l, path + ('L',))
        R = rec(r, path + ('R',))
        s = L | R
        out.append((path, node, s))
        return s
    rec(t, ())
    return out

def replace_subtree_at_path(t, path, new_subtree):
    if len(path) == 0:
        return new_subtree
    _, l, r = t
    if path[0] == 'L':
        return ("JOIN", replace_subtree_at_path(l, path[1:], new_subtree), r)
    else:
        return ("JOIN", l, replace_subtree_at_path(r, path[1:], new_subtree))

def _crossover_swap_matching_subtrees_with_info(ind1, ind2, pset,
                                                forbid_full=True,
                                                prefer_largest=True,
                                                min_leaf_count=2):
    t1 = parse_gp_str(str(ind1))
    t2 = parse_gp_str(str(ind2))
    leaves1 = frozenset(leaves_inorder(t1))
    leaves2 = frozenset(leaves_inorder(t2))
    if leaves1 != leaves2:
        return None

    subs1 = enumerate_subtrees_with_paths(t1)
    subs2 = enumerate_subtrees_with_paths(t2)

    m1, m2 = {}, {}
    for path, subtree, s in subs1:
        m1.setdefault(s, []).append((path, subtree))
    for path, subtree, s in subs2:
        m2.setdefault(s, []).append((path, subtree))

    candidates = set(m1.keys()) & set(m2.keys())
    if forbid_full:
        candidates.discard(leaves1)
    if min_leaf_count is not None and min_leaf_count > 1:
        candidates = {s for s in candidates if len(s) >= min_leaf_count}
    if not candidates:
        return None

    key = max(candidates, key=lambda s: len(s)) if prefer_largest else random.choice(list(candidates))
    path1, sub1 = random.choice(m1[key])
    path2, sub2 = random.choice(m2[key])

    new_t1 = replace_subtree_at_path(t1, path1, sub2)
    new_t2 = replace_subtree_at_path(t2, path2, sub1)
    expr1 = gp.PrimitiveTree.from_string(tuple_to_expr_str(new_t1), pset)
    expr2 = gp.PrimitiveTree.from_string(tuple_to_expr_str(new_t2), pset)
    child1 = creator.Individual(expr1)
    child2 = creator.Individual(expr2)
    return child1, child2, set(key)

def crossover_matching_or_ox(ind1, ind2, pset, table_list,
                             *,
                             forbid_full=True,
                             prefer_largest=True,
                             min_leaf_count=2,
                             forbid_full_subtree=True):
    try:
        res = _crossover_swap_matching_subtrees_with_info(
            ind1, ind2, pset,
            forbid_full=forbid_full,
            prefer_largest=prefer_largest,
            min_leaf_count=min_leaf_count
        )
        if res is not None:
            c1, c2, _ = res
            return c1, c2
    except Exception:
        pass
    return crossover_individuals(ind1, ind2, pset, table_list, forbid_full_subtree=forbid_full_subtree)

def crossover_individuals(ind1, ind2, pset, table_list, *, forbid_full_subtree=True):
    shape1 = parse_gp_str(str(ind1))
    shape2 = parse_gp_str(str(ind2))
    order1 = leaves_inorder(shape1)
    order2 = leaves_inorder(shape2)
    span1, _ = pick_random_subtree_span(shape1, forbid_full=forbid_full_subtree)
    span2, _ = pick_random_subtree_span(shape2, forbid_full=forbid_full_subtree)
    child1_order, _ = ox_crossover_with_span(order1, order2, span1)
    child2_order, _ = ox_crossover_with_span(order2, order1, span2)
    c1_tuple = fill_leaves_with_order(shape1, iter(child1_order))
    c2_tuple = fill_leaves_with_order(shape2, iter(child2_order))
    expr1 = gp.PrimitiveTree.from_string(tuple_to_expr_str(c1_tuple), pset)
    expr2 = gp.PrimitiveTree.from_string(tuple_to_expr_str(c2_tuple), pset)
    return creator.Individual(expr1), creator.Individual(expr2)

# =========================
#  新的代价构件：选择率乘积 / 锚点惩罚 / 不连通惩罚
# =========================
def _tabs(node):
    if isinstance(node, str):
        return [node]
    return _tabs(node[1]) + _tabs(node[2])

def build_allowed_edges(selectivity_map, table_list):
    tables = set(table_list)
    return {pair for pair in selectivity_map.keys() if all(t in tables for t in pair)}

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
    L = set(_tabs(left_subtree)); R = set(_tabs(right_subtree))
    penalty = 1.0
    # 领域启发：IMDB 模式中，movie_info 需要 movie_id 锚，否则容易 Seq Scan
    if ('movie_info' in L and not (('title' in R) or ('movie_companies' in R))) or \
       ('movie_info' in R and not (('title' in L) or ('movie_companies' in L))):
        penalty *= 20.0
    return penalty

# 与 sum_join_outputs 一致的 estimate_cost（若你在别处需要）
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

# === 计算（final_rows, total_generated_rows）— 用于 fitness
def sum_join_outputs(tree, selectivity_map, table_rows, allowed_edges=None, disconnect_penalty=1e6):
    """
    返回：
      final_rows            — 子树最终输出行数
      total_generated_rows  — 所有 JOIN 节点 rows_out 之和（作为 fitness）
    引入：
      - 跨 cut 无边 => 巨额惩罚（不连通）
      - movie_info 未被 title/mc 锚定 => 访问惩罚（避免提前扫大表）
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
#  SQL 解析 / 基数前移近似
# =========================
def load_all_sql_queries(folder_path):
    sql_queries = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".sql"):
            filepath = os.path.join(folder_path, filename)
            try:
                query_info = parse_sql(filepath)  # 需要至少 "FROM"
                sql_queries[filename] = query_info
            except Exception as e:
                print(f"[ERROR] Failed to parse {filename}: {e}")
    return sql_queries

def _collect_condition_text(query_info):
    """
    尽量从 parse_sql 的结构里拼出一个 WHERE/ON 的文本，适配多种返回结构。
    """
    parts = []
    for key in ("WHERE", "where", "FILTERS", "filters", "CONDITIONS"):
        v = query_info.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(map(str, v))
    # 某些解析器会把 ON 条件分别存在 JOIN 列表里
    for key in ("JOIN", "joins"):
        joins = query_info.get(key)
        if isinstance(joins, list):
            for j in joins:
                on = j.get("on") if isinstance(j, dict) else None
                if on:
                    parts.append(str(on))
    return " ".join(parts)

def apply_base_filters(table_rows, query_info):
    """
    把明显的单表过滤折算到基表行数（粗略近似，有助于 fitness 偏向正确顺序）。
    """
    rows = dict(table_rows)
    text = _collect_condition_text(query_info).lower()

    def has(*subs):
        return any(s.lower() in text for s in subs)

    # kind_type: kind='movie' => 近似 1 行
    if 'kind_type' in rows and has("kind_type.kind = 'movie'", "kt.kind = 'movie'", "kt.kind in ('movie')"):
        rows['kind_type'] = 1

    # info_type: 'release dates' => 近似 1 行
    if 'info_type' in rows and has("info_type.info = 'release dates'", "it1.info = 'release dates'"):
        rows['info_type'] = 1

    # company_name: country_code='[us]' => 假设 30%
    if 'company_name' in rows and has("country_code = '[us]'"):
        rows['company_name'] = max(1, int(rows['company_name'] * 0.3))

    # title: production_year > 2000 => 假设 40%
    if 'title' in rows and has("production_year > 2000"):
        rows['title'] = max(1, int(rows['title'] * 0.4))

    # movie_info: 仅当含强词时保守缩些（仍很大）
    if 'movie_info' in rows:
        shrink = 1.0
        if has("note like '%internet%'"):
            shrink *= 0.7
        if has("info like 'usa:% 199%'", "info like 'usa:% 200%'"):
            shrink *= 0.5
        rows['movie_info'] = max(1, int(rows['movie_info'] * shrink))

    return rows

# =========================
#  优化主流程
# =========================
def optimize_query(query_info, selectivity_map, table_rows, filename="(unknown)"):
    table_list = list(dict.fromkeys(query_info["FROM"]))  # 去重
    if len(table_list) < 2:
        print(f"Query {filename} has less than 2 tables. Skipping.")
        return

    # 基数前移近似（可关掉：用 table_rows 即可）
    base_rows = apply_base_filters(table_rows, query_info)

    # 预构建 join 图允许的边集合
    allowed_edges = build_allowed_edges(selectivity_map, table_list)

    pset = create_pset(table_list)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("clone", deepcopy)
    toolbox.register("expr", generate_expr_unique_terminals, pset=pset, table_list=table_list)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_cost(individual):
        try:
            tree = parse_gp_str(str(individual))
            used_tables = get_all_tables(tree)
            # 叶子检查：恰好一次覆盖
            if set(used_tables) != set(table_list) or len(set(used_tables)) != len(used_tables):
                return (float("inf"),)
            _, total_sum = sum_join_outputs(
                tree, selectivity_map, base_rows,
                allowed_edges=allowed_edges,
                disconnect_penalty=1e6
            )
            return (total_sum,)
        except Exception:
            return (float("inf"),)

    toolbox.register("evaluate", eval_cost)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register(
        "mate",
        crossover_matching_or_ox,
        pset=pset,
        table_list=table_list,
        forbid_full=True,
        prefer_largest=True,
        min_leaf_count=3,
        forbid_full_subtree=True,
    )
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)  # 保留
    toolbox.register("mutate", mutate_individual, pset=pset, table_list=table_list)

    print(f"\n🧪 Optimizing Query: {filename}  (Tables: {table_list})")
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    # 初评
    fits0 = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits0):
        ind.fitness.values = fit
    hof.update(pop)

    log_data = []
    NGEN = 100
    elite_num = 5
    mu = len(pop)
    lmbda = mu

    for gen in range(1, NGEN + 1):
        parents = toolbox.select(pop, k=mu)
        offspring = algorithms.varOr(parents, toolbox, lambda_=lmbda, cxpb=0.4, mutpb=0.5)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        hof.update(pop)
        pop = offspring

        if len(hof) > 0:
            inj = min(elite_num, len(hof), len(pop))
            pop[:inj] = [deepcopy(h) for h in hof[:inj]]

        hof.update(pop)
        best_ind = hof[0]
        best_cost = best_ind.fitness.values[0] if best_ind.fitness.valid else toolbox.evaluate(best_ind)[0]
        best_tree_tuple = parse_gp_str(str(best_ind))
        best_tables = get_all_tables(best_tree_tuple)
        log_data.append((gen, best_cost, tuple_to_expr_str(best_tree_tuple), ",".join(best_tables)))

    # 写日志
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"{filename}_log.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best_Cost", "Best_Join_Tree_Str", "Used_Tables"])
        writer.writerows(log_data)

    # 汇总
    summary_file = "summary_best_results.csv"
    if not os.path.exists(summary_file):
        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["SQL_File", "Best_Cost", "Best_Join_Tree_Str", "Table_Order"])

    best = hof[0]
    best_tree_tuple = parse_gp_str(str(best))
    best_cost = toolbox.evaluate(best)[0] if not best.fitness.valid else best.fitness.values[0]
    best_tables = get_all_tables(best_tree_tuple)
    tree_str = tuple_to_expr_str(best_tree_tuple)
    with open(summary_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([filename, best_cost, tree_str, ",".join(best_tables)])

# =========================
#  主执行
# =========================
if __name__ == "__main__":
    random.seed(42)
    selectivity_map, table_rows = load_selectivity_with_rows("Join-Selectivities.xlsx")
    all_queries = load_all_sql_queries("TestQueries")
    for filename, query_info in all_queries.items():
        try:
            optimize_query(query_info, selectivity_map, table_rows, filename=filename)
        except Exception as e:
            print(f"[ERROR] Optimizing {filename} failed: {e}")
