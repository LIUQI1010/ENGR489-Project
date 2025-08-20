# gp_crossover_demos_text_and_img.py
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

def fill_leaves_with_order(shape_tuple, order_iter):
    if isinstance(shape_tuple, str):
        return next(order_iter)
    _, l, r = shape_tuple
    return ("JOIN",
            fill_leaves_with_order(l, order_iter),
            fill_leaves_with_order(r, order_iter))

# ========= 文字版树（ASCII / Unicode） =========
def _style_tokens(use_unicode=True):
    if use_unicode:
        return ("├── ", "└── ", "│   ", "    ")  # branch_mid, branch_last, pipe, space
    else:
        return ("+-- ", "`-- ", "|   ", "    ")

def _format_leaf(name, highlight_set=None, ansi=False):
    highlight_set = highlight_set or set()
    if name not in highlight_set:
        return name
    # 高亮：默认用方括号包裹；若 ansi=True，则蓝底白字
    if ansi:
        return f"\x1b[44;97m {name} \x1b[0m"
    return f"[{name}]"

def ascii_tree_lines_from_tuple(tree_tuple, highlight_leaves=None, use_unicode=True, ansi=False):
    """返回字符串列表，可 '\n'.join 成文本树。根在第一行。"""
    branch_mid, branch_last, pipe, space = _style_tokens(use_unicode)
    highlight_leaves = highlight_leaves or set()
    lines = []

    def rec(node, prefix="", is_last=True):
        if isinstance(node, str):
            lines.append(prefix + (branch_last if is_last else branch_mid) + _format_leaf(node, highlight_leaves, ansi))
            return
        # 内部节点
        label = "JOIN"
        lines.append(prefix + (branch_last if is_last else branch_mid) + label)
        left, right = node[1], node[2]
        # 左孩子（非最后），右孩子（最后）
        child_prefix = prefix + (space if is_last else pipe)
        rec(left,  child_prefix, False)
        rec(right, child_prefix, True)

    # 先打印根（不带分支符号），再递归两个孩子
    if isinstance(tree_tuple, str):
        lines.append(_format_leaf(tree_tuple, highlight_leaves, ansi))
    else:
        lines.append("JOIN")
        left, right = tree_tuple[1], tree_tuple[2]
        rec(left, "", False)
        rec(right, "", True)
    return lines

def ascii_tree_str_from_expr(expr_str, highlight_leaves=None, use_unicode=True, ansi=False):
    t = parse_gp_str(expr_str)
    return "\n".join(ascii_tree_lines_from_tuple(t, highlight_leaves, use_unicode, ansi))

def _with_title_lines(title, lines):
    return [title] + lines

def _side_by_side(lines_left, lines_right, gap=4, pad_left=None):
    pad_left = pad_left if pad_left is not None else (max((len(s) for s in lines_left), default=0))
    out = []
    for a, b in zip_longest(lines_left, lines_right, fillvalue=""):
        out.append(a.ljust(pad_left) + " " * gap + b)
    return out

def render_text_grid_2x2(tl_lines, tr_lines, bl_lines, br_lines,
                         title_tl="Parent 1", title_tr="Parent 2",
                         title_bl="Child 1",  title_br="Child 2",
                         col_gap=6, row_gap=1):
    """把四棵树的行列表排成 2×2 文本网格，返回单个字符串。"""
    tl = _with_title_lines(title_tl, tl_lines)
    tr = _with_title_lines(title_tr, tr_lines)
    bl = _with_title_lines(title_bl, bl_lines)
    br = _with_title_lines(title_br, br_lines)
    top = _side_by_side(tl, tr, gap=col_gap)
    bottom = _side_by_side(bl, br, gap=col_gap, pad_left=len(top[0].split(" " * col_gap)[0]))
    return "\n".join(top + [""] * row_gap + bottom)

# ========= 严格 2×2 图片绘图（无 cluster，Windows 稳定） =========
def draw_four_trees_grid_strict(p1_str, p2_str, c1_str, c2_str,
                                label_tl="Parent 1", label_tr="Parent 2",
                                label_bl="Child 1",  label_br="Child 2",
                                highlight_leaves_bl=None, highlight_leaves_br=None,
                                filename="grid_2x2"):
    """严格 2×2：两行 rank=same + 两列不可见竖边对齐；不使用 cluster。"""
    highlight_leaves_bl = highlight_leaves_bl or set()
    highlight_leaves_br = highlight_leaves_br or set()

    def add_nodes_edges(dot, expr, counter, prefix, leaf_highlight=None):
        if isinstance(expr, str):
            node_id = f"{prefix}{counter[0]}"; counter[0] += 1
            if leaf_highlight and expr in leaf_highlight:
                dot.node(node_id, expr, shape="ellipse", style="filled", fillcolor="lightblue")
            else:
                dot.node(node_id, expr, shape="ellipse")
            return node_id
        label, left, right = expr
        node_id = f"{prefix}{counter[0]}"; counter[0] += 1
        dot.node(node_id, label, shape="box")
        l_id = add_nodes_edges(dot, left, counter, prefix, leaf_highlight)
        r_id = add_nodes_edges(dot, right, counter, prefix, leaf_highlight)
        dot.edge(node_id, l_id); dot.edge(node_id, r_id)
        return node_id

    p1 = parse_gp_str(p1_str); p2 = parse_gp_str(p2_str)
    c1 = parse_gp_str(c1_str); c2 = parse_gp_str(c2_str)

    dot = Digraph(comment="Join Crossover 2x2", format="png")
    dot.attr(rankdir="TB", nodesep="0.35", ranksep="0.6")

    # 标题节点（纯文本）
    dot.node("label_tl", label_tl, shape="plaintext")
    dot.node("label_tr", label_tr, shape="plaintext")
    dot.node("label_bl", label_bl, shape="plaintext")
    dot.node("label_br", label_br, shape="plaintext")

    # 四棵树
    root_tl = add_nodes_edges(dot, p1, counter=[0], prefix="tl")
    root_tr = add_nodes_edges(dot, p2, counter=[0], prefix="tr")
    root_bl = add_nodes_edges(dot, c1, counter=[0], prefix="bl", leaf_highlight=highlight_leaves_bl)
    root_br = add_nodes_edges(dot, c2, counter=[0], prefix="br", leaf_highlight=highlight_leaves_br)

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

# ========= OX（顺序交叉，保留父形状） =========
def ox_crossover(order1, order2, return_span=False):
    size = len(order1)
    a, b = sorted(random.sample(range(size), 2))
    hole = set(order1[a:b])
    res = [None]*size; res[a:b] = order1[a:b]
    idx = b
    for e in order2:
        if e not in hole:
            if idx >= size: idx = 0
            res[idx] = e; idx += 1
    return (res, (a, b)) if return_span else res

# ======== 新增：以“子树叶子整段”作为 OX 保留段（禁止整棵树） ========
def internal_subtrees(expr_tuple):
    """收集所有内部节点对应的子树（包含根）。"""
    lst = []
    def rec(node):
        if isinstance(node, str):
            return
        lst.append(node)
        rec(node[1]); rec(node[2])
    rec(expr_tuple)
    return lst

def pick_random_subtree_span(expr_tuple, *, forbid_full=True):
    """
    随机挑选一个内部子树，并返回：
    - 它在全树叶序中的连续段 [a, b)
    - 该子树的叶集合（用于高亮/解释）
    约束：
    - forbid_full=True 时，不允许选到“覆盖全部叶子”的根子树。
      若仅有根可选（退化情况），则退化为随机保留一个叶子的段。
    """
    internals = internal_subtrees(expr_tuple)
    full_leaves = leaves_inorder(expr_tuple)
    total = len(full_leaves)

    # 过滤掉“整棵树”的候选（当 forbid_full=True）
    if forbid_full:
        candidates = []
        for sub in internals:
            sub_cnt = len(leaves_inorder(sub))
            if sub_cnt < total:  # 严格小于，避免整棵树
                candidates.append(sub)
    else:
        candidates = internals[:]

    if not candidates:
        # 没有合格的内部子树（例如只有根或是单节点树）
        # 退化为保留一个叶子
        a = random.randrange(total)
        return (a, a+1), {full_leaves[a]}

    sub = random.choice(candidates)
    sub_leaves = leaves_inorder(sub)
    a = full_leaves.index(sub_leaves[0])
    b = a + len(sub_leaves)
    return (a, b), set(sub_leaves)

def ox_crossover_with_span(order1, order2, span):
    """
    OX 的定制版：使用给定的 [a,b) 作为保留段。
    返回 (child_order, span)。
    """
    size = len(order1)
    a, b = span
    hole = set(order1[a:b])
    res = [None] * size
    res[a:b] = order1[a:b]
    idx = b
    for e in order2:
        if e not in hole:
            if idx >= size: idx = 0
            res[idx] = e
            idx += 1
    return res, span

def crossover_ox_keep_shape_by_subtree_with_info(ind1, ind2, pset, *, forbid_full=True):
    """
    OX（顺序交叉），但“保留段”由父代里随机选的子树决定（叶子整段保留）。
    - 子代1：保留 P1 的某子树叶序整段，剩余由 P2 顺序填补；树形保持 P1
    - 子代2：对称处理；树形保持 P2
    额外约束：forbid_full=True 时不选整棵树
    返回：(child1, child2, o1, o2, c1_order, c2_order, keep1, keep2)
    """
    # 解析为 tuple，得到叶序
    t1 = parse_gp_str(str(ind1))
    t2 = parse_gp_str(str(ind2))
    o1 = leaves_inorder(t1)
    o2 = leaves_inorder(t2)

    # 用“子树”确定 OX 的保留段（带禁整树约束）
    span1, keep1 = pick_random_subtree_span(t1, forbid_full=forbid_full)
    span2, keep2 = pick_random_subtree_span(t2, forbid_full=forbid_full)

    # 基于给定 span 做 OX
    c1_order, _ = ox_crossover_with_span(o1, o2, span1)
    c2_order, _ = ox_crossover_with_span(o2, o1, span2)

    # 保持父代树形填回
    c1_tuple = fill_leaves_with_order(t1, iter(c1_order))
    c2_tuple = fill_leaves_with_order(t2, iter(c2_order))

    expr1 = gp.PrimitiveTree.from_string(tuple_to_expr_str(c1_tuple), pset)
    expr2 = gp.PrimitiveTree.from_string(tuple_to_expr_str(c2_tuple), pset)
    child1 = creator.Individual(expr1); child2 = creator.Individual(expr2)

    return child1, child2, o1, o2, c1_order, c2_order, keep1, keep2

# ======== 原随机段 OX（可留作对比） ========
def crossover_ox_keep_shape_with_info(ind1, ind2, pset):
    o1 = get_leaf_order_from_ind(ind1)
    o2 = get_leaf_order_from_ind(ind2)
    c1_order, span1 = ox_crossover(o1, o2, return_span=True)
    c2_order, span2 = ox_crossover(o2, o1, return_span=True)
    s1 = parse_gp_str(str(ind1)); s2 = parse_gp_str(str(ind2))
    c1_tuple = fill_leaves_with_order(s1, iter(c1_order))
    c2_tuple = fill_leaves_with_order(s2, iter(c2_order))
    expr1 = gp.PrimitiveTree.from_string(tuple_to_expr_str(c1_tuple), pset)
    expr2 = gp.PrimitiveTree.from_string(tuple_to_expr_str(c2_tuple), pset)
    child1 = creator.Individual(expr1); child2 = creator.Individual(expr2)
    keep1 = set(o1[span1[0]:span1[1]]); keep2 = set(o2[span2[0]:span2[1]])
    return child1, child2, o1, o2, c1_order, c2_order, keep1, keep2

# ========= 子树交叉（交换子树 + 严格修复） =========
def repair_leaves_strict(expr_tuple, all_tables):
    """
    逐叶修复：每张表最多用一次；遇到重复/未知叶时，动态分配“下一个未用表”。
    """
    all_set = set(all_tables)
    used = set()

    def next_unused():
        for t in all_tables:
            if t not in used:
                return t
        return None  # 理论上不会触发

    def rec(node):
        if isinstance(node, str):
            if node in all_set and node not in used:
                used.add(node)
                return node
            repl = next_unused()
            if repl is None:
                return all_tables[-1]  # 兜底，防崩
            used.add(repl)
            return repl
        return ("JOIN", rec(node[1]), rec(node[2]))

    fixed = rec(expr_tuple)

    # 事后校验：如叶集合不等于 all_tables 且数量匹配，则按 all_tables 顺序重填
    leaves = leaves_inorder(fixed)
    if len(leaves) == len(all_tables) and set(leaves) != set(all_tables):
        fixed = fill_leaves_with_order(expr_tuple, iter(all_tables))
    return fixed

def crossover_subtree_safe_with_info(ind1, ind2, pset, table_list):
    """随机内部节点互换子树 + 严格修复；返回子代与高亮集合（子代里来自对方子树的叶）。"""
    ch1 = gp.PrimitiveTree(ind1); ch2 = gp.PrimitiveTree(ind2)
    internal1 = [i for i, n in enumerate(ch1) if getattr(n, "arity", 0) > 0]
    internal2 = [i for i, n in enumerate(ch2) if getattr(n, "arity", 0) > 0]
    if not internal1 or not internal2:
        return creator.Individual(ch1), creator.Individual(ch2), set(), set()

    i1 = random.choice(internal1); i2 = random.choice(internal2)
    s1 = ch1.searchSubtree(i1); s2 = ch2.searchSubtree(i2)

    # 交换前记录两侧子树的叶集合（用于高亮）
    swap1_leaves = set(leaves_inorder(parse_gp_str(str(gp.PrimitiveTree(ch1[s1])))))
    swap2_leaves = set(leaves_inorder(parse_gp_str(str(gp.PrimitiveTree(ch2[s2])))))

    # 交换
    tmp = gp.PrimitiveTree(ch1[s1])
    ch1[s1] = gp.PrimitiveTree(ch2[s2])
    ch2[s2] = tmp

    # 修复
    t1_fixed = repair_leaves_strict(parse_gp_str(str(ch1)), table_list)
    t2_fixed = repair_leaves_strict(parse_gp_str(str(ch2)), table_list)

    # 转回个体
    expr1 = gp.PrimitiveTree.from_string(tuple_to_expr_str(t1_fixed), pset)
    expr2 = gp.PrimitiveTree.from_string(tuple_to_expr_str(t2_fixed), pset)
    child1 = creator.Individual(expr1); child2 = creator.Individual(expr2)

    # 高亮集合：来自对方子树 ∩ 子代实际叶
    child1_leaves = set(get_leaf_order_from_ind(child1))
    child2_leaves = set(get_leaf_order_from_ind(child2))
    hi1 = swap2_leaves & child1_leaves
    hi2 = swap1_leaves & child2_leaves
    return child1, child2, hi1, hi2

# ========= DEMO：OX 图片 + 文字（子树整段保留，且不允许整树） =========
def demo_ox(pset, table_list, seed=None, use_unicode=True):
    if seed is not None: random.seed(seed)
    p1 = creator.Individual(generate_tree_unique(pset, table_list))
    p2 = creator.Individual(generate_tree_unique(pset, table_list))
    # 禁止整棵树被选中作为保留子树
    c1, c2, o1, o2, co1, co2, keep1, keep2 = crossover_ox_keep_shape_by_subtree_with_info(
        p1, p2, pset, forbid_full=True
    )
    print("[OX-Subtree] P1:", o1)
    print("[OX-Subtree] P2:", o2)
    print("[OX-Subtree] C1:", co1, " keep(subtree):", sorted(keep1))
    print("[OX-Subtree] C2:", co2, " keep(subtree):", sorted(keep2))

    # 图片
    draw_four_trees_grid_strict(str(p1), str(p2), str(c1), str(c2),
        label_tl="Parent 1", label_tr="Parent 2",
        label_bl="Child 1 (OX keep P1 subtree)", label_br="Child 2 (OX keep P2 subtree)",
        highlight_leaves_bl=keep1, highlight_leaves_br=keep2,
        filename="grid_2x2_ox")

    # 文字
    tl = ascii_tree_lines_from_tuple(parse_gp_str(str(p1)), use_unicode=use_unicode)
    tr = ascii_tree_lines_from_tuple(parse_gp_str(str(p2)), use_unicode=use_unicode)
    bl = ascii_tree_lines_from_tuple(parse_gp_str(str(c1)), highlight_leaves=keep1, use_unicode=use_unicode)
    br = ascii_tree_lines_from_tuple(parse_gp_str(str(c2)), highlight_leaves=keep2, use_unicode=use_unicode)
    txt = render_text_grid_2x2(
        tl, tr, bl, br,
        title_tl="Parent 1", title_tr="Parent 2",
        title_bl="Child 1 (OX keep P1 subtree)", title_br="Child 2 (OX keep P2 subtree)",
        col_gap=6, row_gap=1
    )
    with open("grid_2x2_ox.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n=== Text 2x2 (OX-Subtree) ===\n" + txt)

    return p1, p2, c1, c2

# ========= DEMO：子树交叉 图片 + 文字 =========
def demo_subtree(pset, table_list, seed=None, use_unicode=True):
    if seed is not None: random.seed(seed)
    p1 = creator.Individual(generate_tree_unique(pset, table_list))
    p2 = creator.Individual(generate_tree_unique(pset, table_list))
    c1, c2, hi1, hi2 = crossover_subtree_safe_with_info(p1, p2, pset, table_list)
    print("[SUBTREE] C1 highlight (from P2):", sorted(hi1))
    print("[SUBTREE] C2 highlight (from P1):", sorted(hi2))

    # 图片
    draw_four_trees_grid_strict(str(p1), str(p2), str(c1), str(c2),
        label_tl="Parent 1", label_tr="Parent 2",
        label_bl="Child 1 (Subtree+Repair)", label_br="Child 2 (Subtree+Repair)",
        highlight_leaves_bl=hi1, highlight_leaves_br=hi2,
        filename="grid_2x2_subtree")

    # 文字
    tl = ascii_tree_lines_from_tuple(parse_gp_str(str(p1)), use_unicode=use_unicode)
    tr = ascii_tree_lines_from_tuple(parse_gp_str(str(p2)), use_unicode=use_unicode)
    bl = ascii_tree_lines_from_tuple(parse_gp_str(str(c1)), highlight_leaves=hi1, use_unicode=use_unicode)
    br = ascii_tree_lines_from_tuple(parse_gp_str(str(c2)), highlight_leaves=hi2, use_unicode=use_unicode)
    txt = render_text_grid_2x2(
        tl, tr, bl, br,
        title_tl="Parent 1", title_tr="Parent 2",
        title_bl="Child 1 (Subtree+Repair)", title_br="Child 2 (Subtree+Repair)",
        col_gap=6, row_gap=1
    )
    with open("grid_2x2_subtree.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n=== Text 2x2 (Subtree) ===\n" + txt)

    return p1, p2, c1, c2

# ========= 入口 =========
def main(seed=None, use_unicode=True):
    if seed is not None: random.seed(seed)
    table_list = ["T1","T2","T3","T4","T5","T6"]
    pset = create_pset(table_list)

    # 定义 DEAP 类型（防重复定义）
    try: creator.FitnessMin
    except AttributeError: creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try: creator.Individual
    except AttributeError: creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    demo_ox(pset, table_list, seed=seed, use_unicode=use_unicode)         # 生成 png + txt（OX 子树整段保留，禁整树）
    demo_subtree(pset, table_list, seed=seed, use_unicode=use_unicode)    # 生成 png + txt（子树交换+修复）

if __name__ == "__main__":
    # Windows 若控制台对 Unicode 线条不友好，可设 use_unicode=False
    main(seed=4, use_unicode=True)
