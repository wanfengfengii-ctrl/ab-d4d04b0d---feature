"""针对运行中服务的联合拾取 API 冒烟测试（仅用标准库）。

环境变量:
  BASE_URL  服务地址（默认 http://app:8000）

全部断言通过时退出码 0，否则非零。
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://app:8000")
TIMEOUT = 5


def request(method, path, body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        BASE_URL + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def col(cands):
    return {"candidates": [
        {"id": cid, "depth": d, "confidence": c} for cid, d, c in cands
    ]}


def colp(cands):
    return {"candidates": [
        {"id": cid, "depth": d, "confidence": c, "phase": p}
        for cid, d, c, p in cands
    ]}


LIMITS = {
    "min_thickness": 6,
    "max_thickness": 10,
    "max_slope": 1,
    "max_thickness_change": 1,
    "max_second_diff": 1,
}

checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def main():
    # 1. 健康检查。
    status, payload = request("GET", "/health")
    check("GET /health 返回 200", status == 200, str(payload))
    check("健康状态为 ok", payload.get("status") == "ok")

    # 2. 可行联合拾取：高置信干扰点 vs 平滑双界面。
    columns = [
        col([("U", 10 + i, 5), ("L", 18 + i, 5), ("X", 0, 9), ("Y", 30, 9)])
        for i in range(8)
    ]
    status, payload = request("POST", "/api/horizons/trace",
                              {"columns": columns, "limits": LIMITS})
    check("可行用例返回 200", status == 200, str(payload))
    check("feasible=true", payload.get("feasible") is True)
    check("status=ok", payload.get("status") == "ok")

    upper = payload.get("upper_horizon", [])
    lower = payload.get("lower_horizon", [])
    check("两条界面各 8 点", len(upper) == 8 and len(lower) == 8)
    check("联合追踪拒绝逐列最强干扰点",
          [p["id"] for p in upper] == ["U"] * 8
          and [p["id"] for p in lower] == ["L"] * 8,
          f"upper={[p['id'] for p in upper]}")
    check("上界面逐点严格位于下界面之上",
          all(upper[i]["depth"] < lower[i]["depth"] for i in range(8)))
    check("逐列厚度均为 8 且在限值内",
          [t["thickness"] for t in payload.get("thicknesses", [])] == [8] * 8)
    check("相邻坡差全部给出且受限",
          len(payload.get("slopes", {}).get("upper", [])) == 7
          and all(abs(s["slope"]) <= LIMITS["max_slope"]
                  for s in payload["slopes"]["upper"]
                  + payload["slopes"]["lower"]))
    check("二阶差全部给出且最大为 0",
          len(payload.get("second_diffs", {}).get("upper", [])) == 6
          and payload["verdict"]["max_second_diff"] == 0)
    verdict = payload.get("verdict", {})
    check("裁决值完整（置信度80/二阶差0/行程14）",
          verdict == {"total_confidence": 80,
                      "max_second_diff": 0, "total_travel": 14},
          str(verdict))

    # 3. 第二组联合拾取：倾斜剖面，厚度恒定。
    columns2 = [
        col([("u", 2 * i, 4), ("l", 2 * i + 7, 6), ("z", 50, 9)])
        for i in range(8)
    ]
    limits2 = {
        "min_thickness": 5, "max_thickness": 9,
        "max_slope": 2, "max_thickness_change": 0, "max_second_diff": 1,
    }
    status, payload2 = request("POST", "/api/horizons/trace",
                               {"columns": columns2, "limits": limits2})
    check("倾斜剖面返回 200 且可行",
          status == 200 and payload2.get("feasible") is True, str(payload2))
    check("倾斜剖面厚度恒定为 7",
          [t["thickness"] for t in payload2.get("thicknesses", [])] == [7] * 8)

    # 4. 无可行组合：厚度被钉死为 9（实际只有 8）。
    limits_no = dict(LIMITS, min_thickness=9, max_thickness=9)
    status, payload = request("POST", "/api/horizons/trace",
                              {"columns": columns, "limits": limits_no})
    check("无解用例返回 200", status == 200)
    check("明确 feasible=false", payload.get("feasible") is False)
    check("无解状态 no_solution", payload.get("status") == "no_solution")
    check("不伪造任何局部轨迹",
          "upper_horizon" not in payload and "lower_horizon" not in payload,
          str(payload))

    # 4b. 相位连续性：高置信负相候选仅出现在后 4 列。
    phase_columns = []
    for i in range(8):
        cands = [
            ("G", 10 + i, 5, "positive"),
            ("g", 18 + i, 5, "positive"),
            ("X", 0, 9, "positive"),
            ("Y", 30, 9, "positive"),
        ]
        if i >= 4:
            cands += [("P", 10 + i, 9, "negative"),
                      ("Q", 18 + i, 9, "negative")]
        phase_columns.append(colp(cands))

    # 不声明反转：跳相高置信候选被拒绝，相位连续的 G/g 胜出。
    status, p_no = request(
        "POST", "/api/horizons/trace",
        {"columns": phase_columns, "limits": LIMITS,
         "phase_continuity": {"polarity_flips": []}})
    check("相位保相用例返回 200", status == 200, str(p_no))
    check("保相请求 feasible=true", p_no.get("feasible") is True)
    check("跳相高置信伪反射被否决",
          [p["id"] for p in p_no.get("upper_horizon", [])] == ["G"] * 8
          and [p["id"] for p in p_no.get("lower_horizon", [])] == ["g"] * 8,
          str(p_no.get("upper_horizon")))
    check("保相用例逐列相位全 positive",
          [p["phase"] for p in p_no.get("phases", {}).get("upper", [])]
          == ["positive"] * 8
          and [p["phase"] for p in p_no.get("phases", {}).get("lower", [])]
          == ["positive"] * 8)
    check("保相用例实际反转边界为空", p_no.get("polarity_flips") == [])
    check("未启用请求不返回相位字段",
          payload.get("phases") is None and payload.get("polarity_flips") is None)

    # 声明边界 3 反转：前 G/g 后 P/Q 的高置信轨迹合法并胜出。
    status, p_flip = request(
        "POST", "/api/horizons/trace",
        {"columns": phase_columns, "limits": LIMITS,
         "phase_continuity": {"polarity_flips": [3]}})
    check("反转声明用例返回 200", status == 200, str(p_flip))
    check("反转声明 feasible=true", p_flip.get("feasible") is True)
    check("声明反转后高置信轨迹胜出",
          [p["id"] for p in p_flip.get("upper_horizon", [])]
          == ["G"] * 4 + ["P"] * 4
          and [p["id"] for p in p_flip.get("lower_horizon", [])]
          == ["g"] * 4 + ["Q"] * 4,
          str(p_flip.get("upper_horizon")))
    check("逐列相位回传为 ++++----",
          [p["phase"] for p in p_flip.get("phases", {}).get("upper", [])]
          == ["positive"] * 4 + ["negative"] * 4
          and [p["phase"] for p in p_flip.get("phases", {}).get("lower", [])]
          == ["positive"] * 4 + ["negative"] * 4)
    check("实际反转边界回传 [3]", p_flip.get("polarity_flips") == [3])

    # 相位规则导致不可行：唯一几何可行轨迹在边界 3 跳相但未声明反转。
    phase_dead = []
    for i in range(8):
        ph = "negative" if i >= 4 else "positive"
        phase_dead.append(colp([
            ("U", 10 + i, 5, ph), ("L", 18 + i, 5, ph),
            ("X", 0, 9, "positive"), ("Y", 30, 9, "positive"),
        ]))
    status, p_dead = request(
        "POST", "/api/horizons/trace",
        {"columns": phase_dead, "limits": LIMITS,
         "phase_continuity": {"polarity_flips": []}})
    check("相位不可行返回 200", status == 200, str(p_dead))
    check("相位不可行 feasible=false", p_dead.get("feasible") is False)
    check("相位不可行 no_solution", p_dead.get("status") == "no_solution")
    check("相位不可行不泄露局部轨迹/相位",
          "upper_horizon" not in p_dead and "phases" not in p_dead
          and "polarity_flips" not in p_dead, str(p_dead))

    # 非法边界与缺失相位按字段拒绝（422）。
    bad_flip = {"columns": phase_columns, "limits": LIMITS,
                "phase_continuity": {"polarity_flips": [7]}}
    status, p_bad = request("POST", "/api/horizons/trace", bad_flip)
    check("越界反转边界返回 422", status == 422, str(p_bad))

    dup_flip = {"columns": phase_columns, "limits": LIMITS,
                "phase_continuity": {"polarity_flips": [3, 3]}}
    status, p_dup = request("POST", "/api/horizons/trace", dup_flip)
    check("重复反转边界返回 422", status == 422, str(p_dup))

    too_many = {"columns": phase_columns, "limits": LIMITS,
                "phase_continuity": {"polarity_flips": [1, 2, 3]}}
    status, p_many = request("POST", "/api/horizons/trace", too_many)
    check("超过两个反转边界返回 422", status == 422, str(p_many))

    missing_phase = {"columns": columns, "limits": LIMITS,
                     "phase_continuity": {"polarity_flips": []}}
    status, p_miss = request("POST", "/api/horizons/trace", missing_phase)
    check("启用但候选缺相位返回 422", status == 422, str(p_miss))

    # 5. 入参校验：列数不足。
    status, payload = request("POST", "/api/horizons/trace",
                              {"columns": columns[:5], "limits": LIMITS})
    check("列数不足返回 422", status == 422, str(payload))

    # 6. 非法 JSON。
    req = urllib.request.Request(
        BASE_URL + "/api/horizons/trace",
        data=b"{not-json", headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=TIMEOUT)
        check("非法 JSON 返回 400", False)
    except urllib.error.HTTPError as exc:
        check("非法 JSON 返回 400", exc.code == 400)

    # 7. 未知路径。
    status, _ = request("GET", "/nope")
    check("未知路径返回 404", status == 404)

    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n冒烟结果: {len(checks) - len(failed)}/{len(checks)} 通过")
    if failed:
        print("失败项:", ", ".join(failed))
        return 1
    print("全部冒烟通过")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # 网络层异常同样判定失败
        print(f"[FAIL] 冒烟脚本异常: {exc!r}")
        sys.exit(2)
