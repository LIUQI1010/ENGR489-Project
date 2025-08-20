# demo_projected_graft_blue.py
# pip install deap graphviz
# 系统需安装 graphviz 可执行程序 (dot)

import random
import argparse
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

def leafset(t):
    if isinstance(t, str): return {t}
    return leafset(t[1]) | leafset(t[2])

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

# ========= Graphviz 2×2 绘图（子树标蓝：内部+叶子） =========
def draw_four_trees_grid_strict(p1_str, p2_str, c1_str, c2_str,
                                label_tl="Parent 1", label_tr="Parent 2",
                                label_bl="Child 1 (Projected Graft)",  label_br="Child 2 (Projected Graft)",
                                highlight_leaves_bl=None, highlight_leaves_br=None,
                                filename="grid_2x2_projected"):
    """
    highlight_leaves_bl / br: 要高亮的“子树的叶集合”。凡叶集合是它的子集的节点（包含内部节点和叶子）都会被标蓝。
    """
    highlight_leaves_bl = set(highlight_leaves_bl or [])
    highlight_leaves_br = set(highlight_leaves_br or [])

    def add_nodes_edges(dot, expr, counter, prefix, subtree_leafset_for_highlight):
        if isinstance(expr, str):
            node_id = f"{prefix}{counter[0]}"; counter[0] += 1
            leafs = {expr}
            if leafs.issubset(subtree_leafset_for_highlight) and leafs:
                dot.node(node_id, expr, shape="ellipse", style="filled", fillcolor="lightblue")
            else:
                dot.node(node_id, expr, shape="ellipse")
            return node_id, leafs

        label, left, right = expr
        node_id = f"{prefix}{counter[0]}"; counter[0] += 1
        l_id, l_set = add_nodes_edges(dot, left,  counter, prefix, subtree_leafset_for_highlight)
        r_id, r_set = add_nodes_edges(dot, right, counter, prefix, subtree_leafset_for_highlight)
        leafs = l_set | r_set

        if leafs.issubset(subtree_leafset_for_highlight) and leafs:
            dot.node(node_id, label, shape="box", style="filled", fillcolor="lightblue")
        else:
            dot.node(node_id, label, shape="box")

        dot.edge(node_id, l_id); dot.edge(node_id, r_id)
        return node_id, leafs

    p1 = parse_gp_str(p1_str); p2 = parse_gp_str(p2_str)
    c1 = parse_gp_str(c1_str); c2 = parse_gp_str(c2_str)

    dot = Digraph(comment="Join Projected Graft 2x2", format="png")
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

# ========= 子树枚举 / 替换 =========
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

# ========= 核心：Projected Graft（投影嫁接） =========
def prune_and_project(expr, target_set):
    """
    把 tuple 树 expr 裁剪到 target_set（目标叶集合）上：
    - 若子树与 target_set 无交集 => 返回 None
    - 若两边都有交集 => 保留该 JOIN 节点
    - 若只有一边有交集 => 压缩掉另一边，返回那一边（可能形成链）
    """
    if isinstance(expr, str):
        return expr if expr in target_set else None
    _, L, R = expr
    Lp = prune_and_project(L, target_set)
    Rp = prune_and_project(R, target_set)
    if Lp is None and Rp is None:
        return None
    if Lp is None:
        return Rp
    if Rp is None:
        return Lp
    return ("JOIN", Lp, Rp)

def choose_graft_subtree(tree_tuple, *, forbid_full=True, min_leaf_count=2, prefer_largest=True):
    """从给定树里选择一个用于嫁接的内部子树（返回 path, subtree, leafset）。"""
    subs = enumerate_subtrees_with_paths(tree_tuple)
    allS = frozenset(leafset(tree_tuple))
    cands = []
    for path, sub, S in subs:
        if isinstance(sub, str):   # 跳过单叶；我们希望至少是内部节点
            continue
        if forbid_full and S == allS:  # 禁整棵树
            continue
        if min_leaf_count and len(S) < min_leaf_count:
            continue
        cands.append((path, sub, S))
    if not cands:
        return None
    if prefer_largest:
        # 优先叶子数最大的
        max_size = max(len(S) for _,_,S in cands)
        pool = [(p,sub,S) for (p,sub,S) in cands if len(S)==max_size]
        return random.choice(pool)
    return random.choice(cands)

def crossover_projected_graft(ind1, ind2, pset,
                              forbid_full=True, min_leaf_count=2, prefer_largest=True):
    """
    子代1：完整保留父1外部结构；在父1中选一棵内部子树，其叶集合为 S1，
          用 父2 在 S1 上的“投影拓扑” 替换父1该段。
    子代2：对称操作（在父2上选子树 S2，用父1在 S2 上的投影替换）。
    """
    t1 = parse_gp_str(str(ind1))
    t2 = parse_gp_str(str(ind2))
    assert set(leaves_inorder(t1)) == set(leaves_inorder(t2)), "两父代必须覆盖同一批表"

    pick1 = choose_graft_subtree(t1, forbid_full=forbid_full, min_leaf_count=min_leaf_count, prefer_largest=prefer_largest)
    pick2 = choose_graft_subtree(t2, forbid_full=forbid_full, min_leaf_count=min_leaf_count, prefer_largest=prefer_largest)

    # 若任一侧无合适候选，则原样返回（也可回退到其它交叉）
    if pick1 is None or pick2 is None:
        return (creator.Individual(gp.PrimitiveTree(ind1)),
                creator.Individual(gp.PrimitiveTree(ind2)),
                set(), set())

    path1, sub1, S1 = pick1
    path2, sub2, S2 = pick2

    proj_from_2_on_S1 = prune_and_project(t2, S1)
    proj_from_1_on_S2 = prune_and_project(t1, S2)

    # 兜底：确保投影至少包含2片叶子
    def count_leaves(x):
        return 1 if isinstance(x, str) else count_leaves(x[1]) + count_leaves(x[2])
    if proj_from_2_on_S1 is None or count_leaves(proj_from_2_on_S1) < 2:
        child1_expr = gp.PrimitiveTree(ind1)
        S1 = set()
    else:
        new_t1 = replace_subtree_at_path(t1, path1, proj_from_2_on_S1)
        child1_expr = gp.PrimitiveTree.from_string(tuple_to_expr_str(new_t1), pset)

    if proj_from_1_on_S2 is None or count_leaves(proj_from_1_on_S2) < 2:
        child2_expr = gp.PrimitiveTree(ind2)
        S2 = set()
    else:
        new_t2 = replace_subtree_at_path(t2, path2, proj_from_1_on_S2)
        child2_expr = gp.PrimitiveTree.from_string(tuple_to_expr_str(new_t2), pset)

    child1 = creator.Individual(child1_expr)
    child2 = creator.Individual(child2_expr)
    return child1, child2, set(S1), set(S2)

# ========= DEMO：Projected Graft（含 PNG + 文本） =========
def demo_projected_graft(pset, table_list, seed=None, use_unicode=True):
    if seed is not None: random.seed(seed)

    # 定义 DEAP 类型（防重复定义）
    try: creator.FitnessMin
    except AttributeError: creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try: creator.Individual
    except AttributeError: creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # 随机父代
    p1 = creator.Individual(generate_tree_unique(pset, table_list))
    p2 = creator.Individual(generate_tree_unique(pset, table_list))

    # 进行“投影嫁接”（子树至少2个叶子；禁止整树）
    c1, c2, S1, S2 = crossover_projected_graft(
        p1, p2, pset, forbid_full=True, min_leaf_count=2, prefer_largest=True
    )
    print("[GRAFT] Child1 grafted leafset from P2 on subset:", sorted(S1))
    print("[GRAFT] Child2 grafted leafset from P1 on subset:", sorted(S2))

    # 图片：在子代上高亮整棵“被替换/投影嫁接”的子树（内部+叶子）
    draw_four_trees_grid_strict(str(p1), str(p2), str(c1), str(c2),
        label_tl="Parent 1", label_tr="Parent 2",
        label_bl="Child 1 (Projected Graft)", label_br="Child 2 (Projected Graft)",
        highlight_leaves_bl=S1, highlight_leaves_br=S2,
        filename="grid_2x2_projected")

    # 文本（叶子用方括号标记）
    tl = ascii_tree_lines_from_tuple(parse_gp_str(str(p1)), use_unicode=use_unicode)
    tr = ascii_tree_lines_from_tuple(parse_gp_str(str(p2)), use_unicode=use_unicode)
    bl = ascii_tree_lines_from_tuple(parse_gp_str(str(c1)), highlight_leaves=S1, use_unicode=use_unicode)
    br = ascii_tree_lines_from_tuple(parse_gp_str(str(c2)), highlight_leaves=S2, use_unicode=use_unicode)
    txt = render_text_grid_2x2(
        tl, tr, bl, br,
        title_tl="Parent 1", title_tr="Parent 2",
        title_bl="Child 1 (Projected Graft)", title_br="Child 2 (Projected Graft)",
        col_gap=6, row_gap=1
    )
    with open("grid_2x2_projected.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n=== Text 2x2 (PROJECTED GRAFT) ===\n" + txt)

    return p1, p2, c1, c2, S1, S2

# ========= 入口 =========
def main(n_tables=8, seed=None, use_unicode=True):
    if seed is not None: random.seed(seed)
    # 生成表名：T1..Tn
    table_list = [f"T{i}" for i in range(1, n_tables + 1)]
    pset = create_pset(table_list)
    demo_projected_graft(pset, table_list, seed=seed, use_unicode=use_unicode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projected Graft crossover demo for join trees.")
    parser.add_argument("-n", "--n_tables", type=int, default=6, help="表的数量（>=2，建议>=4）")
    parser.add_argument("--seed", type=int, default=10, help="随机种子")
    parser.add_argument("--ascii", action="store_true", help="ASCII 样式（控制台不支持 Unicode 时使用）")
    args = parser.parse_args()
    main(n_tables=args.n_tables, seed=args.seed, use_unicode=not args.ascii)
