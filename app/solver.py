"""双反射界面联合追踪求解器。

每列必须同时选出严格分离的上/下两个候选点；相邻两列同时校验
两条界面的坡差与厚度变化，任何连续三列的界面二阶差也参与约束
与全局裁决。采用联合状态动态规划：一列的状态是一个
(上候选, 下候选) 有序对，转移时一次性检查两条界面的全部约束，
因此不会出现"先追一条界面、再为另一条补点"的伪轨迹。

启用 phase_continuity 后，候选回波还带有相位（positive/negative）：
普通边界两侧两条界面须分别保持相位，声明的极性反转边界两侧须分别
反转相位。相位只作为可行性硬约束，不参与裁决。

最优性按词典序：
  1. 置信度总和最大
  2. 两条界面最大二阶差最小
  3. 两条界面总行程最小
  4. 逐列候选编号（上界面编号、下界面编号）字典序最小（稳定决胜）
"""

from __future__ import annotations

# 各字段规模上限：调用方提交 8~24 列，每列 3~8 个候选点。
MIN_COLS = 8
MAX_COLS = 24
MIN_CANDIDATES = 3
MAX_CANDIDATES = 8

# 一次请求至多声明两个互异的极性反转边界。
MAX_POLARITY_FLIPS = 2

PHASE_SIGN = {"positive": 1, "negative": -1}
PHASE_NAME = {1: "positive", -1: "negative"}


class ValidationError(ValueError):
    """请求数据不满足接口契约。"""


def _as_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} 必须是整数")
    return value


def _parse_phase_continuity(raw, n_cols):
    """校验 phase_continuity 字段，返回反转边界集合（省略时为 None）。

    边界 i 表示第 i 列与第 i+1 列之间，合法取值 0 <= i <= n_cols-2。
    """
    if raw is None:  # 字段缺省或显式 null：沿用无相位语义
        return None
    if not isinstance(raw, dict):
        raise ValidationError("phase_continuity 必须是对象")

    flips_raw = raw.get("polarity_flips", [])
    if not isinstance(flips_raw, list):
        raise ValidationError("phase_continuity.polarity_flips 必须是数组")
    if len(flips_raw) > MAX_POLARITY_FLIPS:
        raise ValidationError(
            f"phase_continuity.polarity_flips 至多声明 {MAX_POLARITY_FLIPS} 个边界"
        )

    max_boundary = n_cols - 2
    flips = []
    for k, item in enumerate(flips_raw):
        name = f"phase_continuity.polarity_flips[{k}]"
        boundary = _as_int(item, name)
        if boundary < 0 or boundary > max_boundary:
            raise ValidationError(
                f"{name} 越界：边界编号须在 0~{max_boundary} 之间"
            )
        if boundary in flips:
            raise ValidationError(
                f"phase_continuity.polarity_flips 存在重复边界: {boundary}"
            )
        flips.append(boundary)
    return frozenset(flips)


def _parse_column(raw, index, phase_enabled=False):
    """解析并校验单列数据，返回按深度排序的候选列表。"""
    if not isinstance(raw, dict):
        raise ValidationError(f"columns[{index}] 必须是对象")
    cands_raw = raw.get("candidates")
    if not isinstance(cands_raw, list):
        raise ValidationError(f"columns[{index}].candidates 必须是数组")
    if not (MIN_CANDIDATES <= len(cands_raw) <= MAX_CANDIDATES):
        raise ValidationError(
            f"columns[{index}] 候选点数量必须在 {MIN_CANDIDATES}~{MAX_CANDIDATES} 之间"
        )

    seen_ids = set()
    cands = []
    for j, item in enumerate(cands_raw):
        if not isinstance(item, dict):
            raise ValidationError(f"columns[{index}].candidates[{j}] 必须是对象")
        cid = item.get("id")
        if not isinstance(cid, str) or not cid:
            raise ValidationError(
                f"columns[{index}].candidates[{j}].id 必须是非空字符串"
            )
        if cid in seen_ids:
            raise ValidationError(f"columns[{index}] 内候选编号重复: {cid}")
        seen_ids.add(cid)

        depth = _as_int(
            item.get("depth"), f"columns[{index}].candidates[{j}].depth"
        )
        conf = _as_int(
            item.get("confidence"),
            f"columns[{index}].candidates[{j}].confidence",
        )
        if conf <= 0:
            raise ValidationError(
                f"columns[{index}].candidates[{j}].confidence 必须是正整数"
            )
        cand = {"id": cid, "depth": depth, "confidence": conf}
        if phase_enabled:
            raw_phase = item.get("phase")
            if raw_phase not in PHASE_SIGN:
                raise ValidationError(
                    f"columns[{index}].candidates[{j}].phase 缺失或非法："
                    "启用 phase_continuity 时必须为 positive 或 negative"
                )
            cand["phase"] = PHASE_SIGN[raw_phase]
        cands.append(cand)

    # 深度相同的候选不影响正确性（它们之间无法配成严格分离对），
    # 排序仅用于输出稳定与编号决胜。
    cands.sort(key=lambda c: (c["depth"], c["id"]))
    return cands


def _parse_limits(payload):
    if not isinstance(payload, dict):
        raise ValidationError("limits 必须是对象")

    def get(name):
        if name not in payload:
            raise ValidationError(f"limits.{name} 缺失")
        return _as_int(payload[name], f"limits.{name}")

    raw_slope = get("max_slope")
    raw_tchange = get("max_thickness_change")
    for name, value in (
        ("max_slope", raw_slope),
        ("max_thickness_change", raw_tchange),
    ):
        if value < 0:
            raise ValidationError(f"limits.{name} 不能为负")

    # 二阶差上限可选：缺省表示不加二阶差硬约束（但仍作为裁决目标）。
    raw_second = payload.get("max_second_diff")
    if raw_second is not None:
        raw_second = _as_int(raw_second, "limits.max_second_diff")
        if raw_second < 0:
            raise ValidationError("limits.max_second_diff 不能为负")

    limits = {
        "max_thickness": get("max_thickness"),
        "min_thickness": get("min_thickness"),
        "max_slope": raw_slope,
        "max_thickness_change": raw_tchange,
        "max_second_diff": raw_second,
    }
    if limits["min_thickness"] < 1:
        raise ValidationError("limits.min_thickness 必须 >= 1")
    if limits["max_thickness"] < limits["min_thickness"]:
        raise ValidationError(
            "limits.max_thickness 不能小于 limits.min_thickness"
        )
    if limits["max_slope"] < 0:
        raise ValidationError("limits.max_slope 不能为负")
    return limits


def parse_request(payload):
    """校验请求体，返回 (columns, limits, flips)。

    flips 为 None 表示未启用相位连续性；否则为极性反转边界集合。
    """
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    cols_raw = payload.get("columns")
    if not isinstance(cols_raw, list):
        raise ValidationError("columns 必须是数组")
    if not (MIN_COLS <= len(cols_raw) <= MAX_COLS):
        raise ValidationError(f"columns 数量必须在 {MIN_COLS}~{MAX_COLS} 之间")

    flips = _parse_phase_continuity(payload.get("phase_continuity"), len(cols_raw))
    columns = [
        _parse_column(col, i, phase_enabled=flips is not None)
        for i, col in enumerate(cols_raw)
    ]
    limits = _parse_limits(payload.get("limits"))
    return columns, limits, flips


def _build_states(column, limits):
    """构造单列的全部合法联合状态：(上候选下标, 下候选下标)。"""
    states = []
    n = len(column)
    for ui in range(n):
        upper = column[ui]
        for li in range(n):
            if li == ui:
                continue
            lower = column[li]
            thickness = lower["depth"] - upper["depth"]
            if limits["min_thickness"] <= thickness <= limits["max_thickness"]:
                states.append((ui, li))
    # 决胜稳定用：编号字典序（上编号, 下编号）。
    states.sort(key=lambda s: (column[s[0]]["id"], column[s[1]]["id"]))
    return states


def _better(candidate, current):
    """固定二阶差上限下的词典序比较：(-累计置信度, 总行程) 越小越优，
    全平则按逐列编号状态序号路径决胜。"""
    if current is None:
        return True
    (ck, cpath), (bk, bpath) = candidate, current
    if ck != bk:
        return ck < bk
    return cpath < bpath


def _run_dp(columns, col_states, limits, cap, flips=None, collect_costs=False):
    """在"每条连续三列的二阶差 <= cap"硬约束下做分层状态 DP。

    固定 cap 后只剩置信度（最大化）与行程（最小化）两个可加目标，
    每个 (上一列状态, 当前列状态) 只保留唯一词典序最优前缀。

    flips 非 None 时额外施加相位连续性：边界 ci-1 位于 flips 中则
    上下界面跨边界都必须反转相位，否则都必须保持相位。

    collect_costs=True 时顺带收集全部局部可行三元组产生的二阶差值
    （用于二分全局最优 cap 的候选集合）。

    返回:
      feasible              是否存在全程可行路径
      best                  最优记录 ((neg_conf, travel), path) 或 None
      costs                 collect_costs 时为二阶差值集合，否则为空
    """
    n_cols = len(columns)
    costs = set()
    hard_cap = float("inf") if cap is None else cap
    # 第 0 层：rank -> ((neg_conf, travel), path)，每状态天然唯一。
    prev = {}
    for rank, (ui, li) in enumerate(col_states[0]):
        conf = columns[0][ui]["confidence"] + columns[0][li]["confidence"]
        prev[rank] = ((-conf, 0), [rank])

    for ci in range(1, n_cols):
        cur_col = columns[ci]
        prev_col = columns[ci - 1]
        cur_states = col_states[ci]
        prev_states = col_states[ci - 1]
        nxt = {}

        if ci == 1:
            def predecessors_of(pr):
                entry = prev.get(pr)
                return ((None, entry),) if entry is not None else ()
        else:
            by_prev_rank = {}
            for (ppr, qpr), entry in prev.items():
                by_prev_rank.setdefault(qpr, []).append((ppr, entry))

            def predecessors_of(pr):
                return by_prev_rank.get(pr, ())

        # 本跨段（第 ci-1 列与第 ci 列之间）要求的相位关系。
        phase_relation = None
        if flips is not None:
            phase_relation = -1 if (ci - 1) in flips else 1

        for rnk, (ui, li) in enumerate(cur_states):
            up, lo = cur_col[ui], cur_col[li]
            thickness = lo["depth"] - up["depth"]
            conf_pair = up["confidence"] + lo["confidence"]

            for pr, (pui, pli) in enumerate(prev_states):
                pup, plo = prev_col[pui], prev_col[pli]
                uslope = abs(up["depth"] - pup["depth"])
                lslope = abs(lo["depth"] - plo["depth"])
                if uslope > limits["max_slope"] or lslope > limits["max_slope"]:
                    continue
                pthick = plo["depth"] - pup["depth"]
                if abs(thickness - pthick) > limits["max_thickness_change"]:
                    continue
                if phase_relation is not None and (
                    up["phase"] != phase_relation * pup["phase"]
                    or lo["phase"] != phase_relation * plo["phase"]
                ):
                    continue

                records = predecessors_of(pr)
                if ci >= 2 and collect_costs:
                    for ppui0, ppli0 in col_states[ci - 2]:
                        cu = abs(up["depth"] - 2 * pup["depth"]
                                 + columns[ci - 2][ppui0]["depth"])
                        cl = abs(lo["depth"] - 2 * plo["depth"]
                                 + columns[ci - 2][ppli0]["depth"])
                        c = max(cu, cl)
                        if c <= hard_cap:
                            costs.add(c)

                for ppr, (old_key, path) in records:
                    if ci >= 2:
                        ppui, ppli = col_states[ci - 2][ppr]
                        sec_u = abs(
                            up["depth"] - 2 * pup["depth"]
                            + columns[ci - 2][ppui]["depth"]
                        )
                        sec_l = abs(
                            lo["depth"] - 2 * plo["depth"]
                            + columns[ci - 2][ppli]["depth"]
                        )
                        if max(sec_u, sec_l) > hard_cap:
                            continue

                    neg_conf, travel = old_key
                    cand = (
                        (neg_conf - conf_pair, travel + uslope + lslope),
                        path + [rnk],
                    )
                    state_key = (pr, rnk)
                    if _better(cand, nxt.get(state_key)):
                        nxt[state_key] = cand
        prev = nxt
        if not prev:
            return False, None, costs

    best = None
    for entry in prev.values():
        if _better(entry, best):
            best = entry
    return best is not None, best, costs


def solve(columns, limits, flips=None):
    """联合追踪求解。

    策略（严格按裁决词典序）：
      1. 在用户给定上限 L 下跑一次 DP 取得全局最大置信度 C*，同时收集
         可达的局部二阶差值；上限 L 下不可行即明确无解（不输出轨迹）；
      2. 对候选二阶差值二分：求最小的 K，使"在硬约束 K 下仍能达到
         置信度 C*"（可达置信度对 K 单调）；
      3. 在 K 下做最终（置信度, 行程, 编号路径）词典序 DP。

    flips 非 None 时相位连续性在全部 DP 中作为硬约束施加。
    """
    col_states = [_build_states(col, limits) for col in columns]
    if any(not states for states in col_states):
        return {"feasible": False}

    cap_limit = limits.get("max_second_diff")
    feasible, best_l, costs = _run_dp(
        columns, col_states, limits, cap_limit, flips, collect_costs=True
    )
    if not feasible:
        return {"feasible": False}
    max_confidence = -best_l[0][0]

    # 0 始终是候选（二阶差可能恰好为 0；列数 >= 8 必有连续三列）。
    candidates = sorted(costs | {0})

    def reaches_max_confidence(cap):
        ok, best, _ = _run_dp(columns, col_states, limits, cap, flips)
        return ok and -best[0][0] == max_confidence

    lo_i, hi_i = 0, len(candidates) - 1
    while lo_i < hi_i:
        mid = (lo_i + hi_i) // 2
        if reaches_max_confidence(candidates[mid]):
            hi_i = mid
        else:
            lo_i = mid + 1
    optimal_cap = candidates[lo_i]

    _, best, _ = _run_dp(columns, col_states, limits, optimal_cap, flips)
    (neg_conf, total_travel), ranks = best
    path = [col_states[ci][rank] for ci, rank in enumerate(ranks)]
    return _build_result(
        columns, path, -neg_conf, optimal_cap, total_travel, flips
    )


def _build_result(columns, path, total_conf, max_second_diff, total_travel,
                  flips=None):
    upper, lower, thicknesses = [], [], []
    u_slopes, l_slopes = [], []
    u_seconds, l_seconds = [], []

    for ci, (ui, li) in enumerate(path):
        up, lo = columns[ci][ui], columns[ci][li]
        upper.append({"column": ci, "id": up["id"], "depth": up["depth"]})
        lower.append({"column": ci, "id": lo["id"], "depth": lo["depth"]})
        thicknesses.append(
            {"column": ci, "thickness": lo["depth"] - up["depth"]}
        )
        if ci >= 1:
            pu = columns[ci - 1][path[ci - 1][0]]["depth"]
            pl = columns[ci - 1][path[ci - 1][1]]["depth"]
            u_slopes.append(
                {"from": ci - 1, "to": ci, "slope": up["depth"] - pu}
            )
            l_slopes.append(
                {"from": ci - 1, "to": ci, "slope": lo["depth"] - pl}
            )
        if ci >= 2:
            ppu = columns[ci - 2][path[ci - 2][0]]["depth"]
            ppl = columns[ci - 2][path[ci - 2][1]]["depth"]
            pu = columns[ci - 1][path[ci - 1][0]]["depth"]
            pl = columns[ci - 1][path[ci - 1][1]]["depth"]
            u_seconds.append(
                {"columns": [ci - 2, ci - 1, ci], "second_diff": up["depth"] - 2 * pu + ppu}
            )
            l_seconds.append(
                {"columns": [ci - 2, ci - 1, ci], "second_diff": lo["depth"] - 2 * pl + ppl}
            )

    result = {
        "feasible": True,
        "upper_horizon": upper,
        "lower_horizon": lower,
        "thicknesses": thicknesses,
        "slopes": {"upper": u_slopes, "lower": l_slopes},
        "second_diffs": {"upper": u_seconds, "lower": l_seconds},
    }
    if flips is not None:
        # 逐列相位直接取自入选候选；实际反转边界由所选轨迹推导，
        # 可行时必然恰为声明边界集合。
        phases = {"upper": [], "lower": []}
        actual_flips = []
        for ci, (ui, li) in enumerate(path):
            phases["upper"].append(
                {"column": ci, "phase": PHASE_NAME[columns[ci][ui]["phase"]]}
            )
            phases["lower"].append(
                {"column": ci, "phase": PHASE_NAME[columns[ci][li]["phase"]]}
            )
            if ci >= 1:
                pui, pli = path[ci - 1]
                if columns[ci][ui]["phase"] != columns[ci - 1][pui]["phase"]:
                    actual_flips.append(ci - 1)
        result["phases"] = phases
        result["polarity_flips"] = actual_flips
    result["verdict"] = {
        "total_confidence": total_conf,
        "max_second_diff": max_second_diff,
        "total_travel": total_travel,
    }
    return result


def trace(payload):
    """供 HTTP 层调用的入口：校验 -> 求解。"""
    columns, limits, flips = parse_request(payload)
    result = solve(columns, limits, flips)
    if not result["feasible"]:
        result["status"] = "no_solution"
        result["message"] = "不存在满足全部约束的上下界面联合拾取组合"
    else:
        result["status"] = "ok"
    return result
