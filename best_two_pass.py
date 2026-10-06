#!/usr/bin/env python3
"""
最优配置：Pass 1 (K=20, γ=5) + Pass 2 (传递性恢复)
"""

import json
import numpy as np
from collections import defaultdict

print("Loading data...")
with open('llm_em_all_4dims_3.7max.json') as f:
    full_data = json.load(f)
with open('llm_em_all_4dims_3.7max_bad_cases_pure_text_full.json') as f:
    bad_data = json.load(f)

import sys
sys.path.insert(0, '.')
from llm_em import load_ground_truth_clusters
nodes, gt_clusters, node_to_gt_id = load_ground_truth_clusters()

merged_data = dict(full_data)
for key, val in bad_data.items():
    merged_data[key] = val

pair_info = {}
for key, val in merged_data.items():
    if ':' not in key:
        continue
    u, v = key.split(':')
    reasoning = val.get('Reasoning', [])
    if not reasoning:
        continue
    r = reasoning[0]
    if 'entity_match' not in r or r.get('reasoning') == 'Parsing failed':
        continue
    canon = (min(u, v), max(u, v))
    pair_info[canon] = {
        'score': r['entity_match'] + r['attribute_match'] + r['state_match'] + r['context_match'],
        'entity': r['entity_match'],
    }
print(f"  Valid pairs: {len(pair_info)}")


def get_components(adj):
    visited = set()
    comps = []
    for node in adj:
        if node not in visited:
            comp = set()
            stack = [node]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                for nb in adj.get(n, set()):
                    if nb not in visited:
                        stack.append(nb)
            comps.append(comp)
    return comps


def get_comp_count(adj):
    return len(get_components(adj))


def eval_with_gt(current_edges, node_to_gt_id, pair_info):
    tp = fp = tn = fn = 0
    seen = set()
    for (u, v), info in pair_info.items():
        if u not in node_to_gt_id or v not in node_to_gt_id:
            continue
        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        seen.add(pair)
        pred = 1 if (u, v) in current_edges else 0
        gt = 1 if node_to_gt_id[u] == node_to_gt_id[v] else 0
        if pred == 1 and gt == 1: tp += 1
        elif pred == 1 and gt == 0: fp += 1
        elif pred == 0 and gt == 0: tn += 1
        elif pred == 0 and gt == 1: fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'prec': prec, 'rec': rec, 'f1': f1}


# ============================================================
# Pass 1: K=20, γ=5
# ============================================================
print("\n" + "="*60)
print("PASS 1: Iterative Pruning (K=20, γ=5)")
print("="*60)

init_edges = {(u,v): info for (u,v), info in pair_info.items() if info['score'] >= 2}
current_edges = dict(init_edges)
delta_history = []
prev_comps = None
prune_per_round = 100
K, gamma = 20, 5

for round_num in range(500):
    adj = defaultdict(set)
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
    if not current_edges:
        break
    n_comps = get_comp_count(adj)
    delta_c = n_comps - prev_comps if prev_comps is not None else 0
    prev_comps = n_comps
    delta_history.append(delta_c)
    
    # 计算支撑度
    max_cn = 1
    cn_cache = {}
    for (u, v) in current_edges:
        cn = len(adj[u] & adj[v])
        cn_cache[(u, v)] = cn
        max_cn = max(max_cn, cn)
    supports = {}
    for (u, v) in current_edges:
        cn = cn_cache[(u, v)]
        union = adj[u] | adj[v]
        jac = cn / len(union) if union else 0
        supports[(u, v)] = (cn / max_cn) + jac
    
    # 停止判断
    if len(delta_history) >= K and round_num > 10:
        if sum(delta_history[-K:]) <= gamma:
            print(f"  Stopped at round {round_num}, edges={len(current_edges)}, comps={n_comps}")
            break
    
    sorted_edges = sorted(supports.items(), key=lambda x: x[1])
    to_prune = min(prune_per_round, len(sorted_edges))
    for i in range(to_prune):
        del current_edges[sorted_edges[i][0]]

# Pass 1 评估
adj = defaultdict(set)
for u, v in current_edges:
    adj[u].add(v)
    adj[v].add(u)
components = get_components(adj)
m1 = eval_with_gt(current_edges, node_to_gt_id, pair_info)

print(f"\n  Pass 1 result: F1={m1['f1']:.4f} P={m1['prec']:.4f} R={m1['rec']:.4f} "
      f"FP={m1['FP']} FN={m1['FN']}")

# FN 分布
node_to_comp = {}
for ci, comp in enumerate(components):
    for n in comp:
        node_to_comp[n] = ci

fn_intra = fn_cross = fn_nopair = 0
for (u, v), info in pair_info.items():
    if u not in node_to_gt_id or v not in node_to_gt_id:
        continue
    pred = 1 if (u, v) in current_edges else 0
    gt = 1 if node_to_gt_id[u] == node_to_gt_id[v] else 0
    if pred == 0 and gt == 1:
        c_u, c_v = node_to_comp.get(u), node_to_comp.get(v)
        if c_u is not None and c_v is not None:
            if c_u == c_v: fn_intra += 1
            else: fn_cross += 1
        else: fn_nopair += 1

print(f"  FN breakdown: intra={fn_intra} ({fn_intra/m1['FN']*100:.1f}%), "
      f"cross={fn_cross} ({fn_cross/m1['FN']*100:.1f}%), "
      f"no_pair={fn_nopair} ({fn_nopair/m1['FN']*100:.1f}%)")


# ============================================================
# Pass 2: 传递性恢复（多种策略）
# ============================================================
print("\n" + "="*60)
print("PASS 2: Transitivity-Based FN Recovery")
print("="*60)

# 计算候选边及其 strength
candidates = []
for ci, comp in enumerate(components):
    if len(comp) < 2:
        continue
    comp_list = list(comp)
    for i in range(len(comp_list)):
        for j in range(i+1, len(comp_list)):
            u, v = comp_list[i], comp_list[j]
            canon = (min(u, v), max(u, v))
            if canon not in current_edges:
                nu, nv = adj[u], adj[v]
                cn = len(nu & nv)
                min_deg = min(len(nu), len(nv))
                strength = cn / min_deg if min_deg > 0 else 0
                score = pair_info.get(canon, {}).get('score', 0)
                entity = pair_info.get(canon, {}).get('entity', 0)
                candidates.append((canon, strength, score, entity))

print(f"  Candidate edges: {len(candidates)}")
candidates.sort(key=lambda x: -x[1])

# Strategy C: strength 阈值
print(f"\n  {'Strategy':<35} {'Added':>6} {'TP':>5} {'FP':>5} "
      f"{'F1':>7} {'Prec':>7} {'Rec':>7} {'FP':>6} {'FN':>6}")
print(f"  {'-'*90}")

# Pass 1 only
print(f"  {'Pass 1 only':<35} {'—':>6} {'—':>5} {'—':>5} "
      f"{m1['f1']:>7.4f} {m1['prec']:>7.4f} {m1['rec']:>7.4f} "
      f"{m1['FP']:>6d} {m1['FN']:>6d}")

best_f1 = m1['f1']
best_label = "Pass 1 only"
best_edges = None

for s_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    edges_p2 = dict(current_edges)
    added = rec_tp = add_fp = 0
    for canon, strength, score, entity in candidates:
        if strength >= s_thresh:
            edges_p2[canon] = {'score': score, 'entity': entity}
            added += 1
            u, v = canon
            if u in node_to_gt_id and v in node_to_gt_id:
                if node_to_gt_id[u] == node_to_gt_id[v]:
                    rec_tp += 1
                else:
                    add_fp += 1
    m2 = eval_with_gt(edges_p2, node_to_gt_id, pair_info)
    marker = ""
    if m2['f1'] > best_f1:
        best_f1 = m2['f1']
        best_label = f"strength≥{s_thresh}"
        best_edges = edges_p2
        marker = " ← NEW BEST"
    print(f"  {'strength≥'+str(s_thresh):<35} {added:>6d} {rec_tp:>5d} {add_fp:>5d} "
          f"{m2['f1']:>7.4f} {m2['prec']:>7.4f} {m2['rec']:>7.4f} "
          f"{m2['FP']:>6d} {m2['FN']:>6d}{marker}")

# Strategy D: strength + score 联合
print(f"\n  Hybrid strategies:")
for s_thresh in [0.3, 0.4, 0.5]:
    for score_thresh in [1, 2]:
        edges_d = dict(current_edges)
        added = rec_tp = add_fp = 0
        for canon, strength, score, entity in candidates:
            if strength >= s_thresh and score >= score_thresh:
                edges_d[canon] = {'score': score, 'entity': entity}
                added += 1
                u, v = canon
                if u in node_to_gt_id and v in node_to_gt_id:
                    if node_to_gt_id[u] == node_to_gt_id[v]:
                        rec_tp += 1
                    else:
                        add_fp += 1
        md = eval_with_gt(edges_d, node_to_gt_id, pair_info)
        marker = ""
        if md['f1'] > best_f1:
            best_f1 = md['f1']
            best_label = f"s≥{s_thresh}+score≥{score_thresh}"
            best_edges = edges_d
            marker = " ← NEW BEST"
        print(f"  {'s≥'+str(s_thresh)+' + score≥'+str(score_thresh):<35} {added:>6d} "
              f"{rec_tp:>5d} {add_fp:>5d} {md['f1']:>7.4f} {md['prec']:>7.4f} "
              f"{md['rec']:>7.4f} {md['FP']:>6d} {md['FN']:>6d}{marker}")

# Strategy E: entity_match 约束
print(f"\n  Entity-match constrained strategies:")
for s_thresh in [0.3, 0.4, 0.5]:
    edges_e = dict(current_edges)
    added = rec_tp = add_fp = 0
    for canon, strength, score, entity in candidates:
        if strength >= s_thresh and entity == 1:
            edges_e[canon] = {'score': score, 'entity': entity}
            added += 1
            u, v = canon
            if u in node_to_gt_id and v in node_to_gt_id:
                if node_to_gt_id[u] == node_to_gt_id[v]:
                    rec_tp += 1
                else:
                    add_fp += 1
    me = eval_with_gt(edges_e, node_to_gt_id, pair_info)
    marker = ""
    if me['f1'] > best_f1:
        best_f1 = me['f1']
        best_label = f"s≥{s_thresh}+entity=1"
        best_edges = edges_e
        marker = " ← NEW BEST"
    print(f"  {'s≥'+str(s_thresh)+' + entity=1':<35} {added:>6d} "
          f"{rec_tp:>5d} {add_fp:>5d} {me['f1']:>7.4f} {me['prec']:>7.4f} "
          f"{me['rec']:>7.4f} {me['FP']:>6d} {me['FN']:>6d}{marker}")


# ============================================================
# 最终对比
# ============================================================
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

baseline3 = eval_with_gt(
    {(u,v): info for (u,v), info in pair_info.items() if info['score'] >= 3},
    node_to_gt_id, pair_info)
baseline2 = eval_with_gt(
    {(u,v): info for (u,v), info in pair_info.items() if info['score'] >= 2},
    node_to_gt_id, pair_info)

# 找最佳 Pass 2 结果
best_m = eval_with_gt(best_edges, node_to_gt_id, pair_info) if best_edges else m1

print(f"\n  {'Strategy':<45} {'F1':>7} {'Prec':>7} {'Rec':>7} {'FP':>6} {'FN':>6}")
print(f"  {'-'*80}")
print(f"  {'Baseline (sum>=3)':<45} {baseline3['f1']:>7.4f} {baseline3['prec']:>7.4f} "
      f"{baseline3['rec']:>7.4f} {baseline3['FP']:>6d} {baseline3['FN']:>6d}")
print(f"  {'Baseline (sum>=2)':<45} {baseline2['f1']:>7.4f} {baseline2['prec']:>7.4f} "
      f"{baseline2['rec']:>7.4f} {baseline2['FP']:>6d} {baseline2['FN']:>6d}")
print(f"  {'Pass 1 only (K=20,γ=5)':<45} {m1['f1']:>7.4f} {m1['prec']:>7.4f} "
      f"{m1['rec']:>7.4f} {m1['FP']:>6d} {m1['FN']:>6d}")
print(f"  {'Pass 1+2: '+best_label:<45} {best_m['f1']:>7.4f} {best_m['prec']:>7.4f} "
      f"{best_m['rec']:>7.4f} {best_m['FP']:>6d} {best_m['FN']:>6d}")

# 改进量
print(f"\n  vs Baseline (sum>=3):")
fp_d = baseline3['FP'] - best_m['FP']
fn_d = baseline3['FN'] - best_m['FN']
f1_d = best_m['f1'] - baseline3['f1']
err_b = baseline3['FP'] + baseline3['FN']
err_n = best_m['FP'] + best_m['FN']
print(f"    F1:  {baseline3['f1']:.4f} → {best_m['f1']:.4f} ({f1_d:+.4f}, {f1_d/baseline3['f1']*100:+.1f}%)")
print(f"    FP:  {baseline3['FP']} → {best_m['FP']} ({fp_d:+d}, {fp_d/baseline3['FP']*100:+.1f}%)")
print(f"    FN:  {baseline3['FN']} → {best_m['FN']} ({fn_d:+d}, {fn_d/baseline3['FN']*100:+.1f}%)")
print(f"    Err: {err_b} → {err_n} ({(err_b-err_n)/err_b*100:+.1f}%)")

print(f"\n  vs Pass 1 only:")
fp_d2 = m1['FP'] - best_m['FP']
fn_d2 = m1['FN'] - best_m['FN']
f1_d2 = best_m['f1'] - m1['f1']
print(f"    F1:  {m1['f1']:.4f} → {best_m['f1']:.4f} ({f1_d2:+.4f}, {f1_d2/m1['f1']*100:+.1f}%)")
print(f"    FP:  {m1['FP']} → {best_m['FP']} ({fp_d2:+d}, {fp_d2/m1['FP']*100:+.1f}%)")
print(f"    FN:  {m1['FN']} → {best_m['FN']} ({fn_d2:+d}, {fn_d2/m1['FN']*100:+.1f}%)")

print("\nDone!")
