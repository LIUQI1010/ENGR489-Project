import os
import random
from deap import gp, creator, base
from graphviz import Digraph

# ===== GP 基础设置 =====
def join_tables(a, b):
    return f"join_tables({a}, {b})"

def create_pset(table_list):
    pset = gp.PrimitiveSet("MAIN", 0)
    pset.addPrimitive(join_tables, 2, name="join_tables")
    for table in table_list:
        pset.addTerminal(table, name=table)
    return pset

def generate_expr_unique_terminals_expr_str(table_list):
    shuffled = random.sample(table_list, len(table_list))
    def build_expr(names):
        if len(names) == 1:
            return names[0]
        split = random.randint(1, len(names) - 1)
        left = build_expr(names[:split])
        right = build_expr(names[split:])
        return f"join_tables({left}, {right})"
    return build_expr(shuffled)

def generate_tree_unique(pset, table_list):
    expr_str = generate_expr_unique_terminals_expr_str(table_list)
    return gp.PrimitiveTree.from_string(expr_str, pset)

# ===== 子树变异 =====
def mutate_subtree_unique(individual, pset, max_tries=10):
    ind = gp.PrimitiveTree(individual)
    internal_idxs = [i for i, node in enumerate(ind) if getattr(node, "arity", 0) > 0]
    if not internal_idxs:
        return type(ind)(ind), None
    idx = random.choice(internal_idxs)  # mutation 选中的节点
    sl = ind.searchSubtree(idx)
    sub_terms = [node.name for node in ind[sl] if getattr(node, "arity", 0) == 0]
    sub_terms = list(dict.fromkeys(sub_terms))
    if len(sub_terms) <= 1:
        return type(ind)(ind), idx
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
    return type(ind)(ind), idx  # 返回新个体和 mutation 节点索引


# ===== 安全解析 join_tables 表达式成 tuple =====
def parse_gp_str(expr_str):
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

# ===== 绘图函数 =====
def draw_two_join_trees(before_str, after_str, highlight_idx=None, filename="join_tree_compare"):
    """竖向（Top→Bottom）黑白线框；Before/After 并排；同索引节点浅蓝高亮。"""

    def parse_gp_str(expr_str):
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

    def add_nodes_edges(dot, expr, counter, highlight_idx, prefix):
        """前序遍历建图；返回该子树根节点ID。节点ID用 prefix 前缀避免与另一棵树重名。"""
        if isinstance(expr, str):
            node_id = f"{prefix}{counter[0]}"; counter[0] += 1
            dot.node(node_id, expr, shape="ellipse")  # 叶子：黑白线框
            return node_id
        else:
            label, left, right = expr  # ('JOIN', L, R)
            node_id = f"{prefix}{counter[0]}"; cur_idx = counter[0]; counter[0] += 1
            if highlight_idx is not None and cur_idx == highlight_idx:
                dot.node(node_id, label, shape="box", style="filled", fillcolor="lightblue")
            else:
                dot.node(node_id, label, shape="box")  # JOIN：黑白线框
            left_id  = add_nodes_edges(dot, left,  counter, highlight_idx, prefix)
            right_id = add_nodes_edges(dot, right, counter, highlight_idx, prefix)
            dot.edge(node_id, left_id)
            dot.edge(node_id, right_id)
            return node_id

    before_tree = parse_gp_str(before_str)
    after_tree  = parse_gp_str(after_str)

    dot = Digraph(comment="Join Tree Comparison", format="png")
    # 竖向布局 + 合理间距；newrank 让集群更独立，compound 便于边连接
    dot.attr(rankdir="TB", nodesep="0.35", ranksep="0.5", newrank="true", compound="true")

    # 左边：Before
    with dot.subgraph(name="cluster_before") as c:
        c.attr(label="Before Mutation", labelloc="t")
        root_before = add_nodes_edges(c, before_tree, counter=[0], highlight_idx=highlight_idx, prefix="b")

    # 右边：After
    with dot.subgraph(name="cluster_after") as c:
        c.attr(label="After Mutation", labelloc="t")
        root_after = add_nodes_edges(c, after_tree, counter=[0], highlight_idx=highlight_idx, prefix="a")

    # 加一条不可见的边来拉开两个集群的距离（不影响竖向结构）
    dot.edge(root_before, root_after, style="invis", constraint="false", weight="1")

    dot.render(filename, cleanup=True)
    print(f"✅ 保存图像: {filename}.png")

# ===== 主程序 =====
if __name__ == "__main__":
    random.seed(56)
    table_list = ["T1", "T2", "T3", "T4", "T5", "T6"]
    pset = create_pset(table_list)

    try:
        creator.FitnessMin
    except AttributeError:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    tree = generate_tree_unique(pset, table_list)
    ind = creator.Individual(tree)
    print("Before mutation:", ind)

    mutated, selected_idx = mutate_subtree_unique(ind, pset)
    print("After mutation: ", mutated)

    draw_two_join_trees(str(ind), str(mutated), highlight_idx=selected_idx)
