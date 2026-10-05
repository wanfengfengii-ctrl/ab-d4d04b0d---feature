"""双反射界面联合追踪求解器。

每列必须同时选出严格分离的上/下两个候选点；相邻两列同时校验
两条界面的坡差与厚度变化，任何连续三列的界面二阶差也参与约束
与全局裁决。采用联合状态动态规划：一列的状态是一个
(上候选, 下候选) 有序对，转移时一次性检查两条界面的全部约束，
因此不会出现"先追一条界面、再为另一条补点"的伪轨迹。

启用 phase_continuity 后，每个候选带 positive/negative 相位：
普通相邻边界两侧两条界面须分别保持相位，声明的极性反转边界
（polarity_flips，整数 i 表示第 i 与 i+1 列之间）两侧须分别
反转相位。该约束同样在联合状态转移时一次性校验。

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

# 相位连续性开启时，每条剖面最多声明两个互异反转边界。
MAX_POLARITY_FLIPS = 2
PHASES = ("positive", "negative")


class ValidationError(ValueError):
    """请求数据不满足接口契约。"""


def _as_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} 必须是整数")
    return value


def _parse_column(raw, index, with_phase=False):
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
        phase = None
        if with_phase:
            # 启用相位连续性时，每个候选必须显式给出合法相位。
            phase = item.get("phase")
            if phase not in PHASES:
                raise ValidationError(
                    f"columns[{index}].candidates[{j}].phase "
                    f"必须是 {' 或 '.join(PHASES)}"
                )
        cands.append(
            {"id": cid, "depth": depth, "confidence": conf, "phase": phase}
        )

    # 深度相同的候选不影响正确性（它们之间无法配成严格分离对），
    # 排序仅用于输出稳定与编号决胜。
    cands.sort(key=lambda c: (c["depth"], c["id"]))
    return cands


def _parse_phase_continuity(payload, n_cols):
    """校验 phase_continuity 配置。

    返回 None（未启用，键不存在）或 {"flips": frozenset(边界列号)}。
    边界 i 表示第 i 与第 i+1 列之间，故取值域为 0..n_cols-2；
    至多声明两个互异边界，越界或重复均拒绝。
    """
    if "phase_continuity" not in payload:
        return None
    raw = payload["phase_continuity"]
    if not isinstance(raw, dict):
        raise ValidationError("phase_continuity 必须是对象")
    flips_raw = raw.get("polarity_flips", [])
    if not isinstance(flips_raw, list):
        raise ValidationError("phase_continuity.polarity_flips 必须是数组")
    if len(flips_raw) > MAX_POLARITY_FLIPS:
        raise ValidationError(
            f"polarity_flips 至多声明 {MAX_POLARITY_FLIPS} 个反转边界"
        )
    flips = set()
    for k, value in enumerate(flips_raw):
        i = _as_int(value, f"phase_continuity.polarity_flips[{k}]")
        if not (0 <= i <= n_cols - 2):
            raise ValidationError(
                f"phase_continuity.polarity_flips[{k}]={i} 越界："
                f"边界列号必须在 0~{n_cols - 2} 之间"
            )
        if i in flips:
            raise ValidationError(f"polarity_flips 存在重复边界: {i}")
        flips.add(i)
    # 未知字段不影响语义，按宽容策略放行。
    return {"flips": frozenset(flips)}


def _parse_limits(payload):
    if not isinstance(payload, dict):
        raise ValidationError("limits 必须是对象")

    def get(name):
        if name not in payload:
            raise ValidationError(f"limits.{name} 缺失")
        return _as_int(payload[name], f"limits.{name}")

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
    """校验请求体，返回 (columns, limits, phase_cfg)。

    phase_cfg 为 None 表示未启用 phase_continuity（旧请求口径）。
    """
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    cols_raw = payload.get("columns")
    if not isinstance(cols_raw, list):
        raise ValidationError("columns 必须是数组")
    if not (MIN_COLS <= len(cols_raw) <= MAX_COLS):
        raise ValidationError(f"columns 数量必须在 {MIN_COLS}~{MAX_COLS} 之间")

    phase_cfg = _parse_phase_continuity(payload, len(cols_raw))
    with_phase = phase_cfg is not None
    columns = [_parse_column(col, i, with_phase) for i, col in enumerate(cols_raw)]
    limits = _parse_limits(payload.get("limits"))
    return columns, limits, phase_cfg


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


def _run_dp(columns, col_states, limits, cap, phase_cfg=None,
            collect_costs=False):
    """在"每条连续三列的二阶差 <= cap"硬约束下做分层状态 DP。

    固定 cap 后只剩置信度（最大化）与行程（最小化）两个可加目标，
    每个 (上一列状态, 当前列状态) 只保留唯一词典序最优前缀。

    phase_cfg 非 None 时，相邻列转移额外校验两条界面的相位：
    普通边界两侧各自保持相位，声明的反转边界两侧各自反转相位。

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

                if phase_cfg is not None:
                    # 边界 (ci-1, ci)：声明为反转边界时两条界面各自
                    # 换相，否则各自保相。上/下两条界面独立校验。
                    must_flip = (ci - 1) in phase_cfg["flips"]
                    up_same = up["phase"] == pup["phase"]
                    lo_same = lo["phase"] == plo["phase"]
                    if must_flip:
                        if up_same or lo_same:
                            continue
                    elif not up_same or not lo_same:
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


def solve(columns, limits, phase_cfg=None):
    """联合追踪求解。

    策略（严格按裁决词典序）：
      1. 在用户给定上限 L 下跑一次 DP 取得全局最大置信度 C*，同时收集
         可达的局部二阶差值；上限 L 下不可行即明确无解（不输出轨迹）；
      2. 对候选二阶差值二分：求最小的 K，使"在硬约束 K 下仍能达到
         置信度 C*"（可达置信度对 K 单调）；
      3. 在 K 下做最终（置信度, 行程, 编号路径）词典序 DP。

    phase_cfg 非 None 时，相位连续性与几何约束在同一次联合 DP 中
    一起裁决；相位规则导致的不可行与几何不可行同口径返回无解。
    """
    col_states = [_build_states(col, limits) for col in columns]
    if any(not states for states in col_states):
        return {"feasible": False}

    cap_limit = limits.get("max_second_diff")
    feasible, best_l, costs = _run_dp(
        columns, col_states, limits, cap_limit, phase_cfg,
        collect_costs=True
    )
    if not feasible:
        return {"feasible": False}
    max_confidence = -best_l[0][0]

    # 0 始终是候选（二阶差可能恰好为 0；列数 >= 8 必有连续三列）。
    candidates = sorted(costs | {0})

    def reaches_max_confidence(cap):
        ok, best, _ = _run_dp(columns, col_states, limits, cap, phase_cfg)
        return ok and -best[0][0] == max_confidence

    lo_i, hi_i = 0, len(candidates) - 1
    while lo_i < hi_i:
        mid = (lo_i + hi_i) // 2
        if reaches_max_confidence(candidates[mid]):
            hi_i = mid
        else:
            lo_i = mid + 1
    optimal_cap = candidates[lo_i]

    _, best, _ = _run_dp(columns, col_states, limits, optimal_cap, phase_cfg)
    (neg_conf, total_travel), ranks = best
    path = [col_states[ci][rank] for ci, rank in enumerate(ranks)]
    return _build_result(
        columns, path, -neg_conf, optimal_cap, total_travel, phase_cfg
    )


def _build_result(columns, path, total_conf, max_second_diff, total_travel,
                  phase_cfg=None):
    upper, lower, thicknesses = [], [], []
    u_slopes, l_slopes = [], []
    u_seconds, l_seconds = [], []
    u_phases, l_phases = [], []

    for ci, (ui, li) in enumerate(path):
        up, lo = columns[ci][ui], columns[ci][li]
        upper.append({"column": ci, "id": up["id"], "depth": up["depth"]})
        lower.append({"column": ci, "id": lo["id"], "depth": lo["depth"]})
        thicknesses.append(
            {"column": ci, "thickness": lo["depth"] - up["depth"]}
        )
        if phase_cfg is not None:
            u_phases.append({"column": ci, "phase": up["phase"]})
            l_phases.append({"column": ci, "phase": lo["phase"]})
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
        "verdict": {
            "total_confidence": total_conf,
            "max_second_diff": max_second_diff,
            "total_travel": total_travel,
        },
    }
    if phase_cfg is not None:
        # 实际反转边界：选中轨迹上相位发生切换的列间边界。
        # 由转移约束，它恰好等于声明且被轨迹跨越的 polarity_flips。
        actual_flips = [
            ci - 1
            for ci in range(1, len(path))
            if columns[ci][path[ci][0]]["phase"]
            != columns[ci - 1][path[ci - 1][0]]["phase"]
        ]
        result["phases"] = {"upper": u_phases, "lower": l_phases}
        result["polarity_flips"] = actual_flips
    return result


def trace(payload):
    """供 HTTP 层调用的入口：校验 -> 求解。"""
    columns, limits, phase_cfg = parse_request(payload)
    result = solve(columns, limits, phase_cfg)
    if not result["feasible"]:
        result["status"] = "no_solution"
        result["message"] = "不存在满足全部约束的上下界面联合拾取组合"
    else:
        result["status"] = "ok"
    return result
