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


def pcol(cands):
    """cands: [(id, depth, confidence, phase), ...]"""
    return {"candidates": [
        {"id": cid, "depth": d, "confidence": c, "phase": p}
        for cid, d, c, p in cands
    ]}


def payload(columns, limits=None):
    return {"columns": columns, "limits": limits or DEFAULT_LIMITS}


def ppayload(columns, flips, limits=None):
    body = payload(columns, limits)
    body["phase_continuity"] = {"polarity_flips": list(flips)}
    return body


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


def validate_result(columns, limits, result, flips=None):
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

    if flips is not None:
        # 逐列相位与入选候选一致；实际反转边界恰为声明边界。
        phase_sign = {"positive": 1, "negative": -1}
        up_phases = [columns[i][pairs[i][0]]["phase"] for i in range(n_cols)]
        lo_phases = [columns[i][pairs[i][1]]["phase"] for i in range(n_cols)]
        assert [p["phase"] for p in result["phases"]["upper"]] == \
            [{1: "positive", -1: "negative"}[s] for s in up_phases]
        assert [p["phase"] for p in result["phases"]["lower"]] == \
            [{1: "positive", -1: "negative"}[s] for s in lo_phases]
        actual_u = [
            i for i in range(1, n_cols) if up_phases[i] != up_phases[i - 1]
        ]
        actual_l = [
            i for i in range(1, n_cols) if lo_phases[i] != lo_phases[i - 1]
        ]
        expected_flips = sorted(flips)
        assert [b + 1 for b in expected_flips] == actual_u == actual_l
        assert result["polarity_flips"] == expected_flips
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

    def test_phase_continuity_blocks_confident_jump(self):
        # 几何可追的高置信伪反射在某列相位相反：不启用相位时它被选中；
        # 启用相位连续性（无反转边界）后必须改选相位一致的真反射。
        n = 8

        def body(phased):
            cols = []
            for i in range(n):
                up_phase = "negative" if (phased and i == 4) else "positive"
                if phased:
                    cands = [
                        ("H", 10 + i, 9, up_phase),     # 高置信上界面
                        ("S", 9, 2, "positive"),        # 低置信直线上界面
                        ("LH", 18 + i, 5, "positive"),  # 跟随 H 的下界面
                        ("LS", 17, 5, "positive"),      # 跟随 S 的下界面
                    ]
                    cols.append(pcol(cands))
                else:
                    cols.append(col([
                        ("H", 10 + i, 9),
                        ("S", 9, 2),
                        ("LH", 18 + i, 5),
                        ("LS", 17, 5),
                    ]))
            return cols

        limits = {
            "min_thickness": 6, "max_thickness": 10,
            "max_slope": 2, "max_thickness_change": 1, "max_second_diff": 1,
        }
        plain = trace(payload(body(False), limits))
        self.assertEqual([p["id"] for p in plain["upper_horizon"]], ["H"] * n)
        self.assertNotIn("phases", plain)

        phased_body = ppayload(body(True), frozenset(), limits)
        columns, plimits, flips = parse_request(phased_body)
        phased = trace(phased_body)
        validate_result(columns, plimits, phased, flips)
        # 第 4 列 H 为 negative，无法延续 positive，被迫在该处切到 S。
        self.assertEqual(phased["upper_horizon"][4]["id"], "S")
        self.assertEqual(phased["phases"]["upper"][4]["phase"], "positive")
        self.assertEqual(phased["polarity_flips"], [])
        self.assertEqual(
            [p["phase"] for p in phased["phases"]["upper"]], ["positive"] * n
        )

    def test_phase_flip_required_and_reported(self):
        # 高置信轨迹在边界 3 处上下界面同时反转：声明 [3] 时选中；
        # 声明普通连续（空列表）时几何可行但相位不可行 -> 无解。
        n = 8

        def make():
            cols = []
            for i in range(n):
                pu = "negative" if i >= 4 else "positive"
                pl = "negative" if i >= 4 else "positive"
                cols.append(pcol([
                    ("U", 10 + i, 9, pu),
                    ("L", 18 + i, 5, pl),
                    ("X", 0, 1, "positive"),
                    ("Y", 30, 1, "positive"),
                ]))
            return cols

        limits = dict(DEFAULT_LIMITS)
        flipped = trace(ppayload(make(), [3], limits))
        self.assertTrue(flipped["feasible"])
        self.assertEqual([p["id"] for p in flipped["upper_horizon"]], ["U"] * n)
        self.assertEqual([p["id"] for p in flipped["lower_horizon"]], ["L"] * n)
        self.assertEqual(flipped["polarity_flips"], [3])
        self.assertEqual(
            [p["phase"] for p in flipped["phases"]["upper"]],
            ["positive"] * 4 + ["negative"] * 4,
        )
        self.assertEqual(
            [p["phase"] for p in flipped["phases"]["lower"]],
            ["positive"] * 4 + ["negative"] * 4,
        )

        steady = trace(ppayload(make(), [], limits))
        self.assertFalse(steady["feasible"])
        self.assertEqual(steady["status"], "no_solution")
        self.assertNotIn("upper_horizon", steady)
        self.assertNotIn("phases", steady)
        self.assertNotIn("polarity_flips", steady)

        # 反转声明在错误边界（2）同样不可行：相位必须恰在声明边界反转。
        wrong = trace(ppayload(make(), [2], limits))
        self.assertFalse(wrong["feasible"])

    def test_phase_omitted_keeps_legacy_behavior(self):
        # 即便候选携带 phase 字段，未启用 phase_continuity 时一律忽略，
        # 结果与不含 phase 字段的请求完全一致。
        n = 8
        plain_cols = smooth_columns()
        phased_cols = [
            {"candidates": [
                {**c, "phase": "negative"} for c in col0["candidates"]
            ]}
            for col0 in plain_cols
        ]
        a = trace(payload(plain_cols))
        b = trace(payload(phased_cols))
        self.assertEqual(a, b)

    def test_phase_validation_errors(self):
        good = ppayload([
            pcol([("U", 10 + i, 5, "positive"),
                  ("L", 18 + i, 5, "negative"),
                  ("X", 0, 9, "positive")])
            for i in range(8)
        ], [2, 5])
        parse_request(good)  # 合法：两个互异边界

        # 启用后每个候选必须给 phase。
        missing = ppayload(smooth_columns(), [])
        with self.assertRaises(ValidationError):
            parse_request(missing)

        one_missing = ppayload([
            pcol([("U", 10 + i, 5, "positive"),
                  ("L", 18 + i, 5, "positive"),
                  ("X", 0, 9, "positive")])
            for i in range(8)
        ], [])
        one_missing["columns"][0]["candidates"][0].pop("phase")
        with self.assertRaises(ValidationError):
            parse_request(one_missing)

        bad_phase = ppayload(smooth_columns(), [])
        bad_phase["columns"][0]["candidates"][0]["phase"] = "Positive"
        with self.assertRaises(ValidationError):
            parse_request(bad_phase)

        # 越界边界（负、>= n-1）与非整数。
        for bad_flip in (-1, 7, 8):
            with self.assertRaises(ValidationError):
                parse_request(ppayload(smooth_columns(), [bad_flip]))
        non_int = ppayload(smooth_columns(), [])
        non_int["phase_continuity"]["polarity_flips"] = [2.0]
        with self.assertRaises(ValidationError):
            parse_request(non_int)
        bool_flip = ppayload(smooth_columns(), [])
        bool_flip["phase_continuity"]["polarity_flips"] = [True]
        with self.assertRaises(ValidationError):
            parse_request(bool_flip)

        # 重复边界。
        with self.assertRaises(ValidationError):
            parse_request(ppayload(smooth_columns(), [3, 3]))

        # 超过两个边界。
        with self.assertRaises(ValidationError):
            parse_request(ppayload(smooth_columns(), [1, 3, 5]))

        # phase_continuity 结构非法。
        bad_struct = payload(smooth_columns())
        bad_struct["phase_continuity"] = []
        with self.assertRaises(ValidationError):
            parse_request(bad_struct)
        bad_flips = payload(smooth_columns())
        bad_flips["phase_continuity"] = {"polarity_flips": {1}}
        with self.assertRaises(ValidationError):
            parse_request(bad_flips)

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


def brute_force(columns, limits, flips=None):
    """穷举所有联合状态序列，返回与求解器同口径的最优键。

    key = (-总置信度, 最大二阶差, 总行程, 状态序号元组)；None 表示无解。
    flips 非 None 时同步施加相位连续性硬约束。
    """
    col_states = [_build_states(c, limits) for c in columns]
    if any(not s for s in col_states):
        return None
    n = len(columns)
    best = None

    def rec(ci, ranks, pdu, pdl, ppdu, ppdl, pup, plo, neg_conf, max_sec,
            travel):
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
                    relation = -1 if (ci - 1) in flips else 1
                    if (up["phase"] != relation * pup["phase"]
                            or lo["phase"] != relation * plo["phase"]):
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
    columns = []
    for _ in range(n):
        depths = rng.sample(range(0, 9), k=3)  # 同列深度互异
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
    return payload(columns, limits)


def random_phase_body(rng):
    """带相位的随机实例：相位围绕"上/下界面目标相位"生成以保证一定
    可行率，目标相位在声明边界处整体反转；部分候选随机打散制造伪反射。
    """
    n = rng.randint(MIN_COLS, 10)
    ids = ["a", "b", "c"]
    n_flips = rng.randint(0, 2)
    flips = frozenset(rng.sample(range(n - 1), k=n_flips))

    def target(i, base):
        # 穿过奇数个声明边界后相位反转。
        flipped = sum(1 for b in flips if i > b) % 2
        sign = base if flipped == 0 else -base
        return "positive" if sign == 1 else "negative"

    base_u = rng.choice([1, -1])
    base_l = -base_u  # 上下界面初始相位相反也应可行
    columns = []
    for i in range(n):
        depths = sorted(rng.sample(range(0, 9), k=3))
        tu, tl = target(i, base_u), target(i, base_l)
        cands = []
        for j, d in enumerate(depths):
            if rng.random() < 0.75:
                phase = tu if j == 0 else (tl if j == 2 else rng.choice([tu, tl]))
            else:
                phase = rng.choice(["positive", "negative"])
            cands.append((ids[j], d, rng.randint(1, 5), phase))
        columns.append(pcol(cands))
    limits = {
        "min_thickness": 1,
        "max_thickness": 9,
        "max_slope": rng.randint(3, 7),
        "max_thickness_change": rng.randint(3, 7),
        "max_second_diff": rng.randint(2, 8),
    }
    if rng.random() < 0.5:
        del limits["max_second_diff"]
    return ppayload(columns, flips, limits)


class FuzzTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(20261001)
        feasible_hits = 0
        for _ in range(300):
            body = random_body(rng)
            columns, limits, flips = parse_request(body)
            expected = brute_force(columns, limits, flips)
            actual = solve(columns, limits, flips)
            if expected is None:
                self.assertFalse(actual["feasible"])
                continue
            self.assertTrue(actual["feasible"])
            validate_result(columns, limits, actual, flips)
            key = (
                -actual["verdict"]["total_confidence"],
                actual["verdict"]["max_second_diff"],
                actual["verdict"]["total_travel"],
                result_rank_path(columns, limits, actual),
            )
            self.assertEqual(key, expected)
            feasible_hits += 1
        self.assertGreater(feasible_hits, 50)

    def test_matches_brute_force_with_phases(self):
        rng = random.Random(20261005)
        feasible_hits = 0
        for _ in range(300):
            body = random_phase_body(rng)
            columns, limits, flips = parse_request(body)
            expected = brute_force(columns, limits, flips)
            actual = solve(columns, limits, flips)
            if expected is None:
                self.assertFalse(actual["feasible"])
                # 无解结构不得泄露任何局部轨迹或相位信息。
                self.assertNotIn("upper_horizon", actual)
                self.assertNotIn("phases", actual)
                continue
            self.assertTrue(actual["feasible"])
            validate_result(columns, limits, actual, flips)
            key = (
                -actual["verdict"]["total_confidence"],
                actual["verdict"]["max_second_diff"],
                actual["verdict"]["total_travel"],
                result_rank_path(columns, limits, actual),
            )
            self.assertEqual(key, expected)
            feasible_hits += 1
        self.assertGreater(feasible_hits, 20)


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

    def test_max_size_with_phases_runs_fast(self):
        rng = random.Random(7)
        ids = [f"p{j}" for j in range(MAX_CANDIDATES)]
        flips = frozenset({3, 17})
        body = ppayload([
            pcol([(ids[j], rng.randint(0, 40), rng.randint(1, 9),
                   rng.choice(["positive", "negative"]))
                  for j in range(MAX_CANDIDATES)])
            for _ in range(MAX_COLS)
        ], flips, {
            "min_thickness": 2, "max_thickness": 30,
            "max_slope": 12, "max_thickness_change": 10, "max_second_diff": 20,
        })
        columns, limits, flips = parse_request(body)
        t0 = time.time()
        result = solve(columns, limits, flips)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 5.0)
        if result["feasible"]:
            validate_result(columns, limits, result, flips)


if __name__ == "__main__":
    unittest.main(verbosity=2)
