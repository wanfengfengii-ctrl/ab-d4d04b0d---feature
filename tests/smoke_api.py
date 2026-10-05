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


def pcol(cands):
    return {"candidates": [
        {"id": cid, "depth": d, "confidence": c, "phase": p}
        for cid, d, c, p in cands
    ]}


def with_phases(body, flips):
    body = dict(body)
    body["phase_continuity"] = {"polarity_flips": list(flips)}
    return body


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

    # 5. 相位连续性：声明边界 3 处上下界面同时反转。
    phase_columns = [
        pcol([("U", 10 + i, 5, "negative" if i >= 4 else "positive"),
              ("L", 18 + i, 5, "negative" if i >= 4 else "positive"),
              ("X", 0, 9, "positive"), ("Y", 30, 9, "positive")])
        for i in range(8)
    ]
    status, pp = request(
        "POST", "/api/horizons/trace",
        with_phases({"columns": phase_columns, "limits": LIMITS}, [3]))
    check("相位用例（含反转边界）返回 200 且可行",
          status == 200 and pp.get("feasible") is True, str(pp))
    check("相位用例仍追踪 U/L 双界面",
          [p["id"] for p in pp.get("upper_horizon", [])] == ["U"] * 8
          and [p["id"] for p in pp.get("lower_horizon", [])] == ["L"] * 8,
          str(pp))
    check("返回逐列相位，边界 3 处反转",
          [x["phase"] for x in pp.get("phases", {}).get("upper", [])]
          == ["positive"] * 4 + ["negative"] * 4
          and [x["phase"] for x in pp.get("phases", {}).get("lower", [])]
          == ["positive"] * 4 + ["negative"] * 4, str(pp.get("phases")))
    check("实际反转边界回显为 [3]", pp.get("polarity_flips") == [3],
          str(pp.get("polarity_flips")))
    check("相位用例裁决值与几何追踪一致（80/0/14）",
          pp.get("verdict") == {"total_confidence": 80,
                                "max_second_diff": 0, "total_travel": 14},
          str(pp.get("verdict")))

    # 5b. 同一剖面不声明反转：几何可行但相位不可行 -> 原有无解结构。
    status, pp = request(
        "POST", "/api/horizons/trace",
        with_phases({"columns": phase_columns, "limits": LIMITS}, []))
    check("相位不可行返回 200 + feasible=false",
          status == 200 and pp.get("feasible") is False, str(pp))
    check("相位无解不泄露局部轨迹与相位",
          pp.get("status") == "no_solution"
          and "upper_horizon" not in pp and "phases" not in pp
          and "polarity_flips" not in pp, str(pp))

    # 5c. 启用相位但候选缺失 phase 字段 -> 422 字段拒绝。
    missing_phase = {"columns": columns, "limits": LIMITS}
    missing_phase["phase_continuity"] = {"polarity_flips": []}
    status, pp = request("POST", "/api/horizons/trace", missing_phase)
    check("启用相位但候选缺 phase 返回 422", status == 422, str(pp))

    # 5d. 越界 / 重复反转边界 -> 422。
    bad_boundary = with_phases(
        {"columns": phase_columns, "limits": LIMITS}, [3, 3])
    status, pp = request("POST", "/api/horizons/trace", bad_boundary)
    check("重复反转边界返回 422", status == 422, str(pp))
    oob = with_phases({"columns": phase_columns, "limits": LIMITS}, [7])
    status, pp = request("POST", "/api/horizons/trace", oob)
    check("越界反转边界返回 422", status == 422, str(pp))

    # 5e. 未启用 phase_continuity 的请求不出现相位字段（第 2 节已隐含，
    # 这里显式锁定旧响应形状）。
    status, pp = request("POST", "/api/horizons/trace",
                         {"columns": columns, "limits": LIMITS})
    check("未启用相位时响应无 phases/polarity_flips",
          "phases" not in pp and "polarity_flips" not in pp, str(pp))

    # 6. 入参校验：列数不足。
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
