"""求解器单元测试：构造场景 + 随机实例暴力枚举对照。"""

import os
import random
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solver import (  # noqa: E402
    MAX_CANDIDATES,
    MAX_COLS,
    MIN_CANDIDATES,
    MIN_COLS,
    ValidationError,
    _build_states,
    parse_request,
    solve,
    trace,
)

DEFAULT_LIMITS = {
    "min_thickness": 6,
    "max_thickness": 10,
    "max_slope": 1,
    "max_thickness_change": 1,
    "max_second_diff": 1,
}


def col(cands):
    """cands: [(id, depth, confidence), ...]"""
    return {"candidates": [
        {"id": cid, "depth": d, "confidence": c} for cid, d, c in cands
    ]}


def colp(cands):
    """cands: [(id, depth, confidence, phase), ...]"""
    return {"candidates": [
        {"id": cid, "depth": d, "confidence": c, "phase": p}
        for cid, d, c, p in cands
    ]}


def phase_body(columns, flips=None, limits=None):
    body = payload(columns, limits)
    body["phase_continuity"] = {"polarity_flips": flips or []}
    return body


def payload(columns, limits=None):
    return {"columns": columns, "limits": limits or DEFAULT_LIMITS}


def smooth_columns(n=8):
    """平滑地层 + 每列高置信干扰点的典型剖面。

    平滑轨迹：上界面 10+i，下界面 18+i，置信度 5；
    干扰点深度 0 / 30，置信度 9——逐列选最强会得到交叉突跳。
    """
    return [
        col([("U", 10 + i, 5), ("L", 18 + i, 5), ("X", 0, 9), ("Y", 30, 9)])
        for i in range(n)
    ]


def chosen_pairs(columns, result):
    """从结果反解每列 (上候选下标, 下候选下标)（columns 已按深度排序）。"""
    pairs = []
    for i, c in enumerate(columns):
        index = {cand["id"]: k for k, cand in enumerate(c)}
        pairs.append((
            index[result["upper_horizon"][i]["id"]],
            index[result["lower_horizon"][i]["id"]],
        ))
    return pairs


def validate_result(columns, limits, result, phase_cfg=None):
    """对成功结果做完整约束复核，返回裁决三元组。"""
    n_cols = len(columns)
    assert result["feasible"] is True
    upper, lower = result["upper_horizon"], result["lower_horizon"]
    assert len(upper) == len(lower) == n_cols

    du = [p["depth"] for p in upper]
    dl = [p["depth"] for p in lower]
    pairs = chosen_pairs(columns, result)
    total_conf = 0

    # 逐列严格分离与厚度限值。
    for i in range(n_cols):
        assert upper[i]["column"] == lower[i]["column"] == i
        ui, li = pairs[i]
        assert ui != li
        total_conf += columns[i][ui]["confidence"] + columns[i][li]["confidence"]
        t = dl[i] - du[i]
        assert t == result["thicknesses"][i]["thickness"]
        assert limits["min_thickness"] <= t <= limits["max_thickness"]

    # 相邻坡差与厚度变化。
    for i in range(1, n_cols):
        su, sl = du[i] - du[i - 1], dl[i] - dl[i - 1]
        assert result["slopes"]["upper"][i - 1]["slope"] == su
        assert result["slopes"]["lower"][i - 1]["slope"] == sl
        assert abs(su) <= limits["max_slope"]
        assert abs(sl) <= limits["max_slope"]
        t1 = dl[i] - du[i]
        t0 = dl[i - 1] - du[i - 1]
        assert abs(t1 - t0) <= limits["max_thickness_change"]

    # 连续三列二阶差与裁决值。
    max_sec = 0
    for i in range(2, n_cols):
        su = du[i] - 2 * du[i - 1] + du[i - 2]
        sl = dl[i] - 2 * dl[i - 1] + dl[i - 2]
        assert result["second_diffs"]["upper"][i - 2]["second_diff"] == su
        assert result["second_diffs"]["lower"][i - 2]["second_diff"] == sl
        cap = limits.get("max_second_diff")
        if cap is not None:
            assert abs(su) <= cap
            assert abs(sl) <= cap
        max_sec = max(max_sec, abs(su), abs(sl))

    travel = sum(
        abs(du[i] - du[i - 1]) + abs(dl[i] - dl[i - 1])
        for i in range(1, n_cols)
    )
    verdict = result["verdict"]
    assert verdict["total_confidence"] == total_conf
    assert verdict["max_second_diff"] == max_sec
    assert verdict["total_travel"] == travel

    if phase_cfg is not None:
        flips = set(phase_cfg["flips"])
        up_phase = [columns[i][pairs[i][0]]["phase"] for i in range(n_cols)]
        lo_phase = [columns[i][pairs[i][1]]["phase"] for i in range(n_cols)]
        assert [p["phase"] for p in result["phases"]["upper"]] == up_phase
        assert [p["phase"] for p in result["phases"]["lower"]] == lo_phase
        assert [p["column"] for p in result["phases"]["upper"]] == \
            list(range(n_cols))
        actual = set(result["polarity_flips"])
        # 声明反转边界必反转，普通边界必保相；实际反转边界集合须一致。
        for i in range(1, n_cols):
            boundary = i - 1
            up_changed = up_phase[i] != up_phase[i - 1]
            lo_changed = lo_phase[i] != lo_phase[i - 1]
            if boundary in flips:
                assert up_changed and lo_changed
            else:
                assert not up_changed and not lo_changed
            assert (boundary in actual) == up_changed
        assert actual <= flips  # 未声明的边界绝不能反转
    else:
        assert "phases" not in result and "polarity_flips" not in result

    return total_conf, max_sec, travel


class ScenarioTests(unittest.TestCase):
    def test_smooth_joint_trace_beats_greedy_strongest(self):
        # 逐列最强回波是深度 0/30 的干扰点（成对置信度 18），
        # 但任何跨列延续都违反坡差；联合追踪必须给出平滑双界面。
        columns, limits, _ = parse_request(payload(smooth_columns()))
        result = trace(payload(smooth_columns()))
        self.assertEqual(result["status"], "ok")
        validate_result(columns, limits, result)
        self.assertEqual([p["id"] for p in result["upper_horizon"]], ["U"] * 8)
        self.assertEqual([p["id"] for p in result["lower_horizon"]], ["L"] * 8)
        self.assertEqual([p["depth"] for p in result["upper_horizon"]],
                         [10 + i for i in range(8)])
        self.assertEqual([p["depth"] for p in result["lower_horizon"]],
                         [18 + i for i in range(8)])
        self.assertEqual(result["verdict"]["max_second_diff"], 0)
        self.assertEqual(result["verdict"]["total_travel"], 14)
        self.assertEqual(result["verdict"]["total_confidence"], 80)

    def test_second_diff_renders_no_solution(self):
        # 深度序列 0,0,3,3,6,6,...：坡差/厚度变化全部放行、二阶差
        # 上限为 2 时无可行组合；放宽到 3 后有解且裁决值恰为 3。
        zigzag = [
            col([("U", 3 * (i // 2), 5), ("L", 3 * (i // 2) + 10, 5),
                 ("Z", 100, 9)])
            for i in range(8)
        ]
        limits = {
            "min_thickness": 8, "max_thickness": 12,
            "max_slope": 3, "max_thickness_change": 1, "max_second_diff": 2,
        }
        result = trace(payload(zigzag, limits))
        self.assertFalse(result["feasible"])
        self.assertEqual(result["status"], "no_solution")
        self.assertNotIn("upper_horizon", result)  # 不伪造局部轨迹

        limits["max_second_diff"] = 3
        columns, plimits, _ = parse_request(payload(zigzag, limits))
        ok = trace(payload(zigzag, limits))
        self.assertTrue(ok["feasible"])
        validate_result(columns, plimits, ok)
        self.assertEqual(ok["verdict"]["max_second_diff"], 3)

    def test_thickness_range_renders_no_solution(self):
        limits = dict(DEFAULT_LIMITS, min_thickness=9, max_thickness=9)
        # 平滑对厚度恰为 8，干扰点无法成合法对 -> 无解。
        result = trace(payload(smooth_columns(), limits))
        self.assertFalse(result["feasible"])
        self.assertEqual(result["status"], "no_solution")

    def test_optional_second_diff_still_adjudicated(self):
        # 省略 max_second_diff：无二阶差硬约束，但裁决仍在最大置信度
        # 前提下最小化最大二阶差。
        columns, _, _ = parse_request(payload(smooth_columns()))
        limits = {k: DEFAULT_LIMITS[k] for k in
                  ("min_thickness", "max_thickness",
                   "max_slope", "max_thickness_change")}
        result = solve(columns, limits)
        validate_result(columns, limits, result)
        self.assertEqual(result["verdict"]["max_second_diff"], 0)
        self.assertEqual(result["verdict"]["total_confidence"], 80)

    def test_stable_tie_break_by_candidate_id(self):
        # 每个上/下位置各有两个等深、等置信、不同编号候选，
        # 目标值完全相同，逐列编号字典序最小者稳定胜出。
        columns, limits, _ = parse_request(payload([
            col([("b", 5, 3), ("a", 5, 3), ("d", 12, 3), ("c", 12, 3)])
            for _ in range(8)
        ]))
        result = solve(columns, limits)
        validate_result(columns, limits, result)
        self.assertEqual([p["id"] for p in result["upper_horizon"]], ["a"] * 8)
        self.assertEqual([p["id"] for p in result["lower_horizon"]], ["c"] * 8)

    def test_confidence_priority_over_curvature(self):
        # 高置信折线 H/LH（二阶差 2）与低置信直线 S/LS（二阶差 0）
        # 都可行：第一目标置信度优先，必须选折线。
        upper_zig = [10, 10, 12, 12, 14, 14, 16, 16]
        zigzag = []
        for i in range(8):
            uz = upper_zig[i]
            zigzag.append(col([
                ("H", uz, 9),    # 高置信折线上界面
                ("S", 9, 1),     # 低置信直线上界面
                ("LH", uz + 8, 5),   # 跟随 H 的下界面
                ("LS", 17, 5),       # 跟随 S 的下界面
            ]))
        limits = {
            "min_thickness": 6, "max_thickness": 10,
            "max_slope": 2, "max_thickness_change": 2, "max_second_diff": 2,
        }
        columns, plimits, _ = parse_request(payload(zigzag, limits))
        result = solve(columns, plimits)
        validate_result(columns, plimits, result)
        self.assertEqual([p["id"] for p in result["upper_horizon"]], ["H"] * 8)
        self.assertEqual(result["verdict"]["max_second_diff"], 2)
        self.assertEqual(result["verdict"]["total_confidence"], 14 * 8)

    def test_column_count_bounds(self):
        with self.assertRaises(ValidationError):
            parse_request(payload(smooth_columns(MIN_COLS - 1)))
        with self.assertRaises(ValidationError):
            parse_request(payload(smooth_columns(MAX_COLS + 1)))

    def test_candidate_count_bounds(self):
        too_few = smooth_columns()
        too_few[0] = col([("a", 0, 1), ("b", 5, 1)])
        with self.assertRaises(ValidationError):
            parse_request(payload(too_few))

        too_many = smooth_columns()
        too_many[0] = col([(c, i, 1) for i, c in enumerate("abcdefghi")])
        with self.assertRaises(ValidationError):
            parse_request(payload(too_many))

        just_right = smooth_columns()
        just_right[0] = col([("a", 0, 1), ("b", 5, 1), ("c", 9, 1)])
        parse_request(payload(just_right))  # 3 个合法

    def test_validation_errors(self):
        base = smooth_columns()
        with self.assertRaises(ValidationError):
            parse_request([])
        with self.assertRaises(ValidationError):
            parse_request({"columns": base})  # 缺 limits
        with self.assertRaises(ValidationError):
            parse_request(payload(base, {**DEFAULT_LIMITS, "min_thickness": 0}))
        with self.assertRaises(ValidationError):
            parse_request(payload(base, {**DEFAULT_LIMITS, "max_slope": -1}))
        with self.assertRaises(ValidationError):
            parse_request(payload(base, {
                **DEFAULT_LIMITS, "min_thickness": 9, "max_thickness": 8}))

        dup = payload(smooth_columns())
        dup["columns"][0]["candidates"][2]["id"] = "U"
        with self.assertRaises(ValidationError):
            parse_request(dup)

        zero_conf = payload(smooth_columns())
        zero_conf["columns"][1]["candidates"][0]["confidence"] = 0
        with self.assertRaises(ValidationError):
            parse_request(zero_conf)

        bool_depth = payload(smooth_columns())
        bool_depth["columns"][1]["candidates"][0]["depth"] = True
        with self.assertRaises(ValidationError):
            parse_request(bool_depth)

        float_depth = payload(smooth_columns())
        float_depth["columns"][1]["candidates"][0]["depth"] = 1.5
        with self.assertRaises(ValidationError):
            parse_request(float_depth)


def brute_force(columns, limits, phase_cfg=None):
    """穷举所有联合状态序列，返回与求解器同口径的最优键。

    key = (-总置信度, 最大二阶差, 总行程, 状态序号元组)；None 表示无解。
    phase_cfg 非 None 时，普通边界保相、声明边界反相才允许延续。
    """
    col_states = [_build_states(c, limits) for c in columns]
    if any(not s for s in col_states):
        return None
    n = len(columns)
    flips = phase_cfg["flips"] if phase_cfg is not None else None
    best = None

    def rec(ci, ranks, pdu, pdl, ppdu, ppdl, pup, plo,
            neg_conf, max_sec, travel):
        nonlocal best
        if ci == n:
            key = (neg_conf, max_sec, travel, tuple(ranks))
            if best is None or key < best:
                best = key
            return
        for rank, (ui, li) in enumerate(col_states[ci]):
            up, lo = columns[ci][ui], columns[ci][li]
            thick = lo["depth"] - up["depth"]
            if pdu is not None:
                us = abs(up["depth"] - pdu)
                ls = abs(lo["depth"] - pdl)
                if us > limits["max_slope"] or ls > limits["max_slope"]:
                    continue
                if abs(thick - (pdl - pdu)) > limits["max_thickness_change"]:
                    continue
                if flips is not None:
                    must_flip = (ci - 1) in flips
                    up_same = up["phase"] == pup["phase"]
                    lo_same = lo["phase"] == plo["phase"]
                    if must_flip:
                        if up_same or lo_same:
                            continue
                    elif not up_same or not lo_same:
                        continue
                nmax, ntrav = max_sec, travel + us + ls
                if ppdu is not None:
                    cap2 = limits.get("max_second_diff")
                    cap2 = float("inf") if cap2 is None else cap2
                    su = abs(up["depth"] - 2 * pdu + ppdu)
                    sl = abs(lo["depth"] - 2 * pdl + ppdl)
                    if su > cap2 or sl > cap2:
                        continue
                    nmax = max(nmax, su, sl)
            else:
                nmax, ntrav = 0, 0
            rec(ci + 1, ranks + [rank], up["depth"], lo["depth"],
                pdu, pdl, up, lo,
                neg_conf - up["confidence"] - lo["confidence"], nmax, ntrav)

    rec(0, [], None, None, None, None, None, None, 0, 0, 0)
    return best


def result_rank_path(columns, limits, result):
    """结果 -> 每列在 _build_states 中的状态序号（与决胜序号口径一致）。"""
    ranks = []
    for i, c in enumerate(columns):
        index = {cand["id"]: k for k, cand in enumerate(c)}
        pair = (
            index[result["upper_horizon"][i]["id"]],
            index[result["lower_horizon"][i]["id"]],
        )
        ranks.append(_build_states(c, limits).index(pair))
    return tuple(ranks)


def random_body(rng):
    n = rng.randint(MIN_COLS, 10)
    ids = ["a", "b", "c"]
    with_phase = rng.random() < 0.5
    # 先决定反转边界，再让候选 a/b 严格遵循由此产生的相位模式，
    # 从而既有大量相位可行实例，又保留候选 c 的随机相位制造不可行。
    n_flips = rng.randint(0, 2) if with_phase else 0
    flip_set = set(rng.sample(range(n - 1), k=n_flips)) if n_flips else set()
    pattern = []
    sign = 1
    for i in range(n):
        if i > 0 and (i - 1) in flip_set:
            sign *= -1
        pattern.append("positive" if sign > 0 else "negative")

    columns = []
    for i in range(n):
        depths = rng.sample(range(0, 9), k=3)  # 同列深度互异
        if with_phase:
            phases = [pattern[i], pattern[i],
                      rng.choice(("positive", "negative"))]
            columns.append(colp([
                (ids[j], depths[j], rng.randint(1, 5), phases[j])
                for j in range(3)
            ]))
        else:
            columns.append(col([
                (ids[j], depths[j], rng.randint(1, 5)) for j in range(3)
            ]))
    limits = {
        "min_thickness": 1,
        "max_thickness": 9,
        "max_slope": rng.randint(3, 7),
        "max_thickness_change": rng.randint(3, 7),
        "max_second_diff": rng.randint(2, 8),
    }
    # 约一半实例省略二阶差硬上限（裁决仍最小化最大二阶差）。
    if rng.random() < 0.5:
        del limits["max_second_diff"]
    body = payload(columns, limits)
    if with_phase:
        body["phase_continuity"] = {
            "polarity_flips": sorted(flip_set)
        }
    return body


class PhaseScenarioTests(unittest.TestCase):
    def test_high_confidence_phase_jump_is_pseudo_reflection(self):
        # 高置信伪反射 P/Q 只出现在后 4 列且整体为负相：未声明反转时，
        # 相位连续的低置信 G/g 必须胜出（几何上二者完全等价）。
        n = 8
        columns = []
        for i in range(n):
            cands = [
                ("G", 10 + i, 5, "positive"),
                ("g", 18 + i, 5, "positive"),
                ("X", 0, 9, "positive"),
                ("Y", 30, 9, "positive"),
            ]
            if i >= 4:
                cands += [("P", 10 + i, 9, "negative"),
                          ("Q", 18 + i, 9, "negative")]
            columns.append(colp(cands))
        body = phase_body(columns, flips=[], limits=DEFAULT_LIMITS)
        cols, lim, cfg = parse_request(body)
        result = trace(body)
        validate_result(cols, lim, result, cfg)
        self.assertEqual([p["id"] for p in result["upper_horizon"]], ["G"] * n)
        self.assertEqual([p["id"] for p in result["lower_horizon"]], ["g"] * n)
        self.assertEqual(result["verdict"]["total_confidence"], 80)
        self.assertEqual(result["polarity_flips"], [])
        self.assertEqual(
            [p["phase"] for p in result["phases"]["upper"]], ["positive"] * n
        )

    def test_declared_flip_selects_high_confidence_track(self):
        # 同样的 P/Q 负相轨迹，在声明边界 3 反转后合法并以高置信胜出：
        # 前 4 列走 G/g，后 4 列走 P/Q，实际反转边界恰为 [3]。
        n = 8
        columns = []
        for i in range(n):
            cands = [
                ("G", 10 + i, 5, "positive"),
                ("g", 18 + i, 5, "positive"),
                ("X", 0, 9, "positive"),
                ("Y", 30, 9, "positive"),
            ]
            if i >= 4:
                cands += [("P", 10 + i, 9, "negative"),
                          ("Q", 18 + i, 9, "negative")]
            columns.append(colp(cands))
        body = phase_body(columns, flips=[3], limits=DEFAULT_LIMITS)
        cols, lim, cfg = parse_request(body)
        result = trace(body)
        validate_result(cols, lim, result, cfg)
        self.assertEqual(
            [p["id"] for p in result["upper_horizon"]], ["G"] * 4 + ["P"] * 4
        )
        self.assertEqual(
            [p["id"] for p in result["lower_horizon"]], ["g"] * 4 + ["Q"] * 4
        )
        self.assertEqual(result["verdict"]["total_confidence"], 40 + 72)
        self.assertEqual(result["polarity_flips"], [3])
        self.assertEqual(
            [p["phase"] for p in result["phases"]["upper"]],
            ["positive"] * 4 + ["negative"] * 4,
        )

    def test_phase_rules_render_no_solution_without_traces(self):
        # 平滑双界面 U/L 是唯一几何可行状态，但其相位在边界 3 反转；
        # 不声明反转 -> 相位规则使联合轨迹不可行，返回原无解结构。
        columns = []
        for i in range(8):
            ph = "negative" if i >= 4 else "positive"
            columns.append(colp([
                ("U", 10 + i, 5, ph), ("L", 18 + i, 5, ph),
                ("X", 0, 9, "positive"), ("Y", 30, 9, "positive"),
            ]))
        body = phase_body(columns, flips=[], limits=DEFAULT_LIMITS)
        result = trace(body)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["status"], "no_solution")
        self.assertNotIn("upper_horizon", result)
        self.assertNotIn("lower_horizon", result)
        self.assertNotIn("phases", result)
        self.assertNotIn("polarity_flips", result)

        # 声明该边界后恢复可行，且实际反转边界回传。
        body["phase_continuity"]["polarity_flips"] = [3]
        cols, lim, cfg = parse_request(body)
        ok = trace(body)
        validate_result(cols, lim, ok, cfg)
        self.assertEqual(ok["polarity_flips"], [3])

    def test_two_declared_flips(self):
        # 两条反转边界：相位模式 + - +，两条界面同步换相两次。
        n = 8
        columns = []
        for i in range(n):
            sign = 1
            for f in (2, 5):
                if i > f:
                    sign *= -1
            ph = "positive" if sign > 0 else "negative"
            columns.append(colp([
                ("U", 10 + i, 5, ph), ("L", 18 + i, 5, ph),
                ("X", 0, 9, "positive"), ("Y", 30, 9, "positive"),
            ]))
        body = phase_body(columns, flips=[2, 5], limits=DEFAULT_LIMITS)
        cols, lim, cfg = parse_request(body)
        result = trace(body)
        validate_result(cols, lim, result, cfg)
        self.assertEqual([p["id"] for p in result["upper_horizon"]], ["U"] * n)
        self.assertEqual(result["polarity_flips"], [2, 5])
        self.assertEqual(
            [p["phase"] for p in result["phases"]["upper"]],
            ["positive"] * 3 + ["negative"] * 3 + ["positive"] * 2,
        )

    def test_disabled_request_ignores_phase_fields(self):
        # 不带 phase_continuity：候选即便带 phase 也被忽略，
        # 响应不含相位字段，结果与旧口径一致。
        columns = []
        for i in range(8):
            columns.append(colp([
                ("U", 10 + i, 5, "positive"), ("L", 18 + i, 5, "negative"),
                ("X", 0, 9, "positive"), ("Y", 30, 9, "negative"),
            ]))
        plain = payload(columns)
        cols, lim, cfg = parse_request(plain)
        self.assertIsNone(cfg)
        result = trace(plain)
        validate_result(cols, lim, result)
        self.assertNotIn("phases", result)
        self.assertNotIn("polarity_flips", result)
        self.assertEqual(result["verdict"]["total_confidence"], 80)

    def test_enabled_without_flips_matches_plain_on_consistent_phase(self):
        columns = [
            colp([("U", 10 + i, 5, "positive"), ("L", 18 + i, 5, "positive"),
                  ("X", 0, 9, "positive"), ("Y", 30, 9, "positive")])
            for i in range(8)
        ]
        plain = trace(payload(columns))
        body = phase_body(columns, flips=[])
        enabled = trace(body)
        for key in ("upper_horizon", "lower_horizon", "thicknesses",
                    "slopes", "second_diffs", "verdict"):
            self.assertEqual(plain[key], enabled[key])

    def test_phase_validation_errors(self):
        import copy

        def enabled(cols, flips=None):
            # 各子用例会就地改写字段，这里深拷贝避免互相污染。
            return phase_body(copy.deepcopy(cols),
                              flips if flips is not None else [])

        base = [
            colp([("U", 10 + i, 5, "positive"), ("L", 18 + i, 5, "positive"),
                  ("X", 0, 9, "positive"), ("Y", 30, 9, "positive")])
            for i in range(8)
        ]

        # 启用后候选缺失 phase。
        missing = enabled(base)
        del missing["columns"][0]["candidates"][0]["phase"]
        with self.assertRaises(ValidationError):
            parse_request(missing)

        # 非法相位取值。
        bad_phase = enabled(base)
        bad_phase["columns"][1]["candidates"][0]["phase"] = "POSITIVE"
        with self.assertRaises(ValidationError):
            parse_request(bad_phase)
        bad_phase["columns"][1]["candidates"][0]["phase"] = None
        with self.assertRaises(ValidationError):
            parse_request(bad_phase)

        # phase_continuity 不是对象。
        null_cfg = payload(base)
        null_cfg["phase_continuity"] = None
        with self.assertRaises(ValidationError):
            parse_request(null_cfg)

        # polarity_flips 不是数组。
        bad_list = enabled(base, flips=[1])
        bad_list["phase_continuity"]["polarity_flips"] = 3
        with self.assertRaises(ValidationError):
            parse_request(bad_list)

        # 越界边界（8 列合法边界 0~6）。
        with self.assertRaises(ValidationError):
            parse_request(enabled(base, flips=[7]))
        with self.assertRaises(ValidationError):
            parse_request(enabled(base, flips=[-1]))

        # 非整数与布尔值。
        with self.assertRaises(ValidationError):
            parse_request(enabled(base, flips=[2.0]))
        with self.assertRaises(ValidationError):
            parse_request(enabled(base, flips=[True]))

        # 重复边界。
        with self.assertRaises(ValidationError):
            parse_request(enabled(base, flips=[3, 3]))

        # 超过两个。
        with self.assertRaises(ValidationError):
            parse_request(enabled(base, flips=[1, 2, 3]))

        # 边界 0、6 是合法值，应通过校验。
        parse_request(enabled(base, flips=[0, 6]))


class FuzzTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(20261001)
        feasible_hits = 0
        phase_hits = 0
        for _ in range(400):
            body = random_body(rng)
            columns, limits, phase_cfg = parse_request(body)
            expected = brute_force(columns, limits, phase_cfg)
            actual = solve(columns, limits, phase_cfg)
            if expected is None:
                self.assertFalse(actual["feasible"])
                continue
            self.assertTrue(actual["feasible"])
            validate_result(columns, limits, actual, phase_cfg)
            key = (
                -actual["verdict"]["total_confidence"],
                actual["verdict"]["max_second_diff"],
                actual["verdict"]["total_travel"],
                result_rank_path(columns, limits, actual),
            )
            self.assertEqual(key, expected)
            feasible_hits += 1
            if phase_cfg is not None:
                phase_hits += 1
        self.assertGreater(feasible_hits, 60)
        # 必须有足量启用相位的可行实例，相位输出才真正被对照过。
        self.assertGreater(phase_hits, 30)


class PerformanceTests(unittest.TestCase):
    def test_max_size_runs_fast(self):
        rng = random.Random(42)
        ids = [f"p{j}" for j in range(MAX_CANDIDATES)]
        columns, limits, _ = parse_request(payload([
            col([(ids[j], rng.randint(0, 40), rng.randint(1, 9))
                 for j in range(MAX_CANDIDATES)])
            for _ in range(MAX_COLS)
        ], {
            "min_thickness": 2, "max_thickness": 30,
            "max_slope": 12, "max_thickness_change": 10, "max_second_diff": 20,
        }))
        t0 = time.time()
        result = solve(columns, limits)
        elapsed = time.time() - t0
        self.assertTrue(result["feasible"])
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
