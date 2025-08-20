# demo_swap_matching_subtrees_blue_min2.py
# pip install deap graphviz

import random
from itertools import zip_longest
from graphviz import Digraph
from deap import gp, creator, base

# ========= 基础：原语集合 / 随机树生成 =========
def join_tables(a, b):
    return f"join_tables({a}, {b})"

def create_pset(table_list):
    pset = gp.PrimitiveSet("MAIN", 0)
    pset.addPrimitive(join_tables, 2, name="join_tables")
    for t in table_list:
        pset.addTerminal(t, name=t)
    return pset

def generate_expr_unique_terminals_expr_str(table_list):
    shuffled = random.sample(table_list, len(table_list))
    def build(names):
        if len(names) == 1:
            return names[0]
        k = random.randint(1, len(names) - 1)
        left = build(names[:k]); right = build(names[k:])
        return f"join_tables({left}, {right})"
    return build(shuffled)

def generate_tree_unique(pset, table_list):
    return gp.PrimitiveTree.from_string(generate_expr_unique_terminals_expr_str(table_list), pset)

# ========= 安全解析 / 重建 / 叶序 =========
def parse_gp_str(expr_str: str):
    expr_str = expr_str.strip()
    if not expr_str.startswith("join_tables"):
        return expr_str
    inside = expr_str[len("join_tables("):-1]
    depth = 0; split_index = None
    for i, ch in enumerate(inside):
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif ch == "," and depth == 0:
            split_index = i; break
    left_str = inside[:split_index].strip()
    right_str = inside[split_index+1:].strip()
    return ("JOIN", parse_gp_str(left_str), parse_gp_str(right_str))

def tuple_to_expr_str(t):
    if isinstance(t, str): return t
    _, l, r = t
    return f"join_tables({tuple_to_expr_str(l)}, {tuple_to_expr_str(r)})"

def leaves_inorder(t):
    if isinstance(t, str): return [t]
    return leaves_inorder(t[1]) + leaves_inorder(t[2])

def get_leaf_order_from_ind(ind):
    return leaves_inorder(parse_gp_str(str(ind)))

# ========= 文本树渲染（ASCII / Unicode） =========
def _style_tokens(use_unicode=True):
    return ("├── ", "└── ", "│   ", "    ") if use_unicode else ("+-- ", "`-- ", "|   ", "    ")

def _format_leaf(name, highlight_set=None, ansi=False):
    highlight_set = highlight_set or set()
    if name not in highlight_set:
        return name
    if ansi:
        return f"\x1b[44;97m {name} \x1b[0m"
    return f"[{name}]"

def ascii_tree_lines_from_tuple(tree_tuple, highlight_leaves=None, use_unicode=True, ansi=False):
    branch_mid, branch_last, pipe, space = _style_tokens(use_unicode)
    highlight_leaves = highlight_leaves or set()
    lines = []
    def rec(node, prefix="", is_last=True):
        if isinstance(node, str):
            lines.append(prefix + (branch_last if is_last else branch_mid) + _format_leaf(node, highlight_leaves, ansi))
            return
        label = "JOIN"
        lines.append(prefix + (branch_last if is_last else branch_mid) + label)
        left, right = node[1], node[2]
        child_prefix = prefix + (space if is_last else pipe)
        rec(left,  child_prefix, False)
        rec(right, child_prefix, True)
    if isinstance(tree_tuple, str):
        lines.append(_format_leaf(tree_tuple, highlight_leaves, ansi))
    else:
        lines.append("JOIN")
        left, right = tree_tuple[1], tree_tuple[2]
        rec(left, "", False)
        rec(right, "", True)
    return lines

def _with_title_lines(title, lines):
    return [title] + lines

def _side_by_side(lines_left, lines_right, gap=6, pad_left=None):
    pad_left = pad_left if pad_left is not None else (max((len(s) for s in lines_left), default=0))
    out = []
    for a, b in zip_longest(lines_left, lines_right, fillvalue=""):
        out.append(a.ljust(pad_left) + " " * gap + b)
    return out

def render_text_grid_2x2(tl_lines, tr_lines, bl_lines, br_lines,
                         title_tl="Parent 1", title_tr="Parent 2",
                         title_bl="Child 1",  title_br="Child 2",
                         col_gap=6, row_gap=1):
    tl = _with_title_lines(title_tl, tl_lines)
    tr = _with_title_lines(title_tr, tr_lines)
    bl = _with_title_lines(title_bl, bl_lines)
    br = _with_title_lines(title_br, br_lines)
    top = _side_by_side(tl, tr, gap=col_gap)
    bottom = _side_by_side(bl, br, gap=col_gap, pad_left=len(top[0].split(" " * col_gap)[0]))
    return "\n".join(top + [""] * row_gap + bottom)

# ========= 2×2 图片绘图（子树标蓝：内部+叶子） =========
def draw_four_trees_grid_strict(p1_str, p2_str, c1_str, c2_str,
                                label_tl="Parent 1", label_tr="Parent 2",
                                label_bl="Child 1",  label_br="Child 2",
                                highlight_leaves_bl=None, highlight_leaves_br=None,
                                filename="grid_2x2"):
    """
    highlight_leaves_bl / br: 要高亮的“子树的叶集合”。凡叶集合是它的子集的节点（包含内部节点和叶子）都会被标蓝。
    """
    highlight_leaves_bl = set(highlight_leaves_bl or [])
    highlight_leaves_br = set(highlight_leaves_br or [])

    def add_nodes_edges(dot, expr, counter, prefix, subtree_leafset_for_highlight):
        if isinstance(expr, str):
            node_id = f"{prefix}{counter[0]}"; counter[0] += 1
            leafset = {expr}
            if leafset.issubset(subtree_leafset_for_highlight) and leafset:
                dot.node(node_id, expr, shape="ellipse", style="filled", fillcolor="lightblue")
            else:
                dot.node(node_id, expr, shape="ellipse")
            return node_id, leafset

        label, left, right = expr
        node_id = f"{prefix}{counter[0]}"; counter[0] += 1
        l_id, l_set = add_nodes_edges(dot, left,  counter, prefix, subtree_leafset_for_highlight)
        r_id, r_set = add_nodes_edges(dot, right, counter, prefix, subtree_leafset_for_highlight)
        leafset = l_set | r_set

        if leafset.issubset(subtree_leafset_for_highlight) and leafset:
            dot.node(node_id, label, shape="box", style="filled", fillcolor="lightblue")
        else:
            dot.node(node_id, label, shape="box")

        dot.edge(node_id, l_id); dot.edge(node_id, r_id)
        return node_id, leafset

    p1 = parse_gp_str(p1_str); p2 = parse_gp_str(p2_str)
    c1 = parse_gp_str(c1_str); c2 = parse_gp_str(c2_str)

    dot = Digraph(comment="Join Crossover 2x2", format="png")
    dot.attr(rankdir="TB", nodesep="0.35", ranksep="0.6")

    # 标题
    dot.node("label_tl", label_tl, shape="plaintext")
    dot.node("label_tr", label_tr, shape="plaintext")
    dot.node("label_bl", label_bl, shape="plaintext")
    dot.node("label_br", label_br, shape="plaintext")

    # 四棵树
    root_tl, _ = add_nodes_edges(dot, p1, counter=[0], prefix="tl", subtree_leafset_for_highlight=set())
    root_tr, _ = add_nodes_edges(dot, p2, counter=[0], prefix="tr", subtree_leafset_for_highlight=set())
    root_bl, _ = add_nodes_edges(dot, c1, counter=[0], prefix="bl", subtree_leafset_for_highlight=highlight_leaves_bl)
    root_br, _ = add_nodes_edges(dot, c2, counter=[0], prefix="br", subtree_leafset_for_highlight=highlight_leaves_br)

    # 行对齐
    dot.body.append(f"{{rank=same; label_tl label_tr}}")
    dot.body.append(f"{{rank=same; {root_tl} {root_tr}}}")
    dot.body.append(f"{{rank=same; label_bl label_br}}")
    dot.body.append(f"{{rank=same; {root_bl} {root_br}}}")

    # 标题压在树上
    for lab, root in [("label_tl", root_tl), ("label_tr", root_tr),
                      ("label_bl", root_bl), ("label_br", root_br)]:
        dot.edge(lab, root, style="invis", weight="50", minlen="1")

    # 列对齐
    dot.edge(root_tl, root_bl, style="invis", weight="200", minlen="6")
    dot.edge(root_tr, root_br, style="invis", weight="200", minlen="6")

    dot.render(filename, cleanup=True)
    print(f"✅ 保存图像: {filename}.png")

# ========= 子树枚举 / 替换 & 交换实现 =========
def enumerate_subtrees_with_paths(t):
    """
    返回 [(path, subtree, leafset)]，path 为 ('L','R',...) 的元组。根 path = ()。
    叶子节点也作为子树返回（leafset 为单元素）。
    """
    out = []
    def rec(node, path):
        if isinstance(node, str):
            out.append((path, node, frozenset([node])))
            return frozenset([node])
        _, l, r = node
        Lset = rec(l, path + ('L',))
        Rset = rec(r, path + ('R',))
        myset = Lset | Rset
        out.append((path, node, myset))
        return myset
    rec(t, ())
    return out

def replace_subtree_at_path(t, path, new_subtree):
    """在 tuple 树 t 的给定 path 处替换为 new_subtree，返回新树。"""
    if len(path) == 0:
        return new_subtree
    _, l, r = t
    if path[0] == 'L':
        return ("JOIN", replace_subtree_at_path(l, path[1:], new_subtree), r)
    else:
        return ("JOIN", l, replace_subtree_at_path(r, path[1:], new_subtree))

def crossover_swap_matching_subtrees_with_info(ind1, ind2, pset,
                                               forbid_full=True,
                                               prefer_largest=True,
                                               min_leaf_count=2):
    """
    若存在“叶集合相同（结构可不同）”的子树，则交换之。
    限制：
      - forbid_full=True：不交换整棵树（叶集合=全体）
      - min_leaf_count=2：子树至少包含 2 个叶子（避免只换单叶）
      - prefer_largest=True：优先交换叶子数最大的匹配子树；否则随机选择
    返回：(child1, child2, swapped_leafset)
    """
    t1 = parse_gp_str(str(ind1))
    t2 = parse_gp_str(str(ind2))
    leaves_all = frozenset(leaves_inorder(t1))
    assert leaves_all == frozenset(leaves_inorder(t2))

    subs1 = enumerate_subtrees_with_paths(t1)
    subs2 = enumerate_subtrees_with_paths(t2)

    # 映射 leafset -> 列表(可能同一叶集合在树中出现多个位置/结构)
    m1 = {}
    for path, subtree, s in subs1:
        m1.setdefault(s, []).append((path, subtree))
    m2 = {}
    for path, subtree, s in subs2:
        m2.setdefault(s, []).append((path, subtree))

    # 筛选候选：两边都存在；可选不为整棵树；叶子数 >= min_leaf_count
    candidates = set(m1.keys()) & set(m2.keys())
    if forbid_full:
        candidates = {s for s in candidates if s != leaves_all}
    if min_leaf_count is not None and min_leaf_count > 1:
        candidates = {s for s in candidates if len(s) >= min_leaf_count}

    if not candidates:
        # 没有匹配就返回原样（也可改成回退 OX）
        return creator.Individual(gp.PrimitiveTree(ind1)), creator.Individual(gp.PrimitiveTree(ind2)), set()

    # 选择要交换的叶集合
    if prefer_largest:
        key = max(candidates, key=lambda s: len(s))
    else:
        key = random.choice(list(candidates))

    # 两侧各挑一个位置
    path1, sub1 = random.choice(m1[key])
    path2, sub2 = random.choice(m2[key])

    # 交换
    new_t1 = replace_subtree_at_path(t1, path1, sub2)
    new_t2 = replace_subtree_at_path(t2, path2, sub1)

    # 转回个体
    expr1 = gp.PrimitiveTree.from_string(tuple_to_expr_str(new_t1), pset)
    expr2 = gp.PrimitiveTree.from_string(tuple_to_expr_str(new_t2), pset)
    child1 = creator.Individual(expr1)
    child2 = creator.Individual(expr2)

    return child1, child2, set(key)

# ========= DEMO：交换“叶集合相同”的子树（含 PNG + 文本） =========
def demo_swap(pset, table_list, seed=None, use_unicode=True):
    if seed is not None: random.seed(seed)

    # 定义 DEAP 类型（防重复定义）
    try: creator.FitnessMin
    except AttributeError: creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try: creator.Individual
    except AttributeError: creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # 随机父代
    p1 = creator.Individual(generate_tree_unique(pset, table_list))
    p2 = creator.Individual(generate_tree_unique(pset, table_list))

    # 进行“匹配子树交换”（子树至少2个叶子）
    c1, c2, swapped = crossover_swap_matching_subtrees_with_info(
        p1, p2, pset, forbid_full=True, prefer_largest=True, min_leaf_count=2
    )
    print("[SWAP] swapped leafset:", sorted(swapped))

    # 图片：在子代上高亮整棵交换子树（内部+叶子）
    draw_four_trees_grid_strict(str(p1), str(p2), str(c1), str(c2),
        label_tl="Parent 1", label_tr="Parent 2",
        label_bl="Child 1 (swap matching subtree ≥2 leaves)", label_br="Child 2 (swap matching subtree ≥2 leaves)",
        highlight_leaves_bl=swapped, highlight_leaves_br=swapped,
        filename="grid_2x2_swap")

    # 文本（叶子用方括号标记）
    tl = ascii_tree_lines_from_tuple(parse_gp_str(str(p1)), use_unicode=use_unicode)
    tr = ascii_tree_lines_from_tuple(parse_gp_str(str(p2)), use_unicode=use_unicode)
    bl = ascii_tree_lines_from_tuple(parse_gp_str(str(c1)), highlight_leaves=swapped, use_unicode=use_unicode)
    br = ascii_tree_lines_from_tuple(parse_gp_str(str(c2)), highlight_leaves=swapped, use_unicode=use_unicode)
    txt = render_text_grid_2x2(
        tl, tr, bl, br,
        title_tl="Parent 1", title_tr="Parent 2",
        title_bl="Child 1 (swap matching subtree ≥2 leaves)", title_br="Child 2 (swap matching subtree ≥2 leaves)",
        col_gap=6, row_gap=1
    )
    with open("grid_2x2_swap.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n=== Text 2x2 (SWAP) ===\n" + txt)

    return p1, p2, c1, c2, swapped

# ========= 入口 =========
def main(seed=None, use_unicode=True):
    if seed is not None: random.seed(seed)
    table_list = ["T1","T2","T3","T4","T5", "T6", "T7", "T8"]  # 表多一些更容易出现“非整树”的大子树
    pset = create_pset(table_list)
    demo_swap(pset, table_list, seed=seed, use_unicode=True)

if __name__ == "__main__":
    # Windows 若控制台对 Unicode 线条不友好，可设 use_unicode=False
    main(seed=110, use_unicode=True)
