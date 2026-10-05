# 浅地层雷达双界面联合追踪服务

对浅地层雷达剖面的**上、下反射界面进行成对联合追踪**。每列必须同时
选出严格分离的上/下两个候选点；相邻两列同时校验两条界面的坡差与
厚度变化；任何连续三列的界面二阶差既作为可行性硬约束，也参与全局
裁决。求解以"一个联合状态 = (上候选, 下候选) 有序对"的动态规划
完成，**不会先追一条界面再给另一条补点**，因此不会产生交叉、突跳
或厚度失真的伪轨迹。

## 运行

```bash
# 常驻服务（默认 8000，可用 API_PORT 覆盖）
docker compose up app

# 一次性校验：app 健康后自动执行 代码测试 + 镜像构建检查 + API 冒烟，
# 以退出码报告（0 全部通过）
docker compose up --build verify
echo $?
```

`API_PORT` 可通过 shell 环境变量或同目录 `.env` 覆盖：

```bash
API_PORT=9000 docker compose up app
```

健康检查：`GET /healthz`（或 `/health`），返回 `{"status":"ok"}`。

## API

`POST /api/horizons/trace`

请求体：

```json
{
  "columns": [
    {"candidates": [
      {"id": "U", "depth": 10, "confidence": 5},
      {"id": "L", "depth": 18, "confidence": 5},
      {"id": "X", "depth": 0,  "confidence": 9}
    ]}
  ],
  "limits": {
    "min_thickness": 6,
    "max_thickness": 10,
    "max_slope": 1,
    "max_thickness_change": 1,
    "max_second_diff": 1
  }
}
```

约束：

- `columns` 数量 8~24；每列 `candidates` 3~8 个；
- `id` 为列内唯一的非空字符串；`depth` 为整数；`confidence` 为正整数；
- 各限值均为整数（坡差/厚度变化/二阶差按绝对值解释）；
- `min_thickness`、`max_thickness`、`max_slope`、`max_thickness_change`
  必填；`max_second_diff` 可选——缺省时不加二阶差硬约束，但最大二阶差
  仍作为第二裁决目标被最小化。

### 相位连续性（可选）

请求体可携带 `phase_continuity` 以启用相位连续性约束：

```json
{"phase_continuity": {"polarity_flips": [3, 11]}}
```

- 启用后**每个候选**必须提供 `"phase": "positive" | "negative"`；
- `polarity_flips` 可省略（等同空列表）；至多声明 **2 个互异**整数边界，
  边界 `i` 表示第 `i` 列与第 `i+1` 列之间，合法范围 `0 ~ 列数-2`，
  越界值与重复边界按非法请求拒绝（HTTP 422）；
- 上、下两条界面在**普通边界**两侧须分别**保持**相位，在**声明的
  反转边界**两侧须分别**反转**相位（两条界面各自独立判定）；
- 相位是纯可行性硬约束，不改变裁决词典序。

省略 `phase_continuity`（或显式 `null`）时，即便候选携带 `phase` 字段
也一律忽略，输入、响应、裁决与无解语义与启用前完全一致。

成功响应（HTTP 200）包含：

| 字段 | 含义 |
| --- | --- |
| `upper_horizon` / `lower_horizon` | 两条界面逐列候选（编号、深度） |
| `thicknesses` | 逐列厚度（下深 − 上深） |
| `slopes.upper/lower` | 相邻列坡差（有符号） |
| `second_diffs.upper/lower` | 连续三列二阶差（有符号） |
| `phases.upper/lower` | 仅启用相位时：两条界面逐列相位 |
| `polarity_flips` | 仅启用相位时：所选轨迹的实际反转边界（升序） |
| `verdict` | `total_confidence` / `max_second_diff` / `total_travel` |

几何约束可满足、但相位规则使联合轨迹不可行时，仍返回原有无解结构
（不含任何局部轨迹或相位信息）：

```json
{"feasible": false, "status": "no_solution",
 "message": "不存在满足全部约束的上下界面联合拾取组合"}
```

响应中**不含任何局部轨迹**。入参非法返回 HTTP 422（或非法 JSON 400）。

## 裁决目标（严格词典序）

1. 两条界面置信度总和**最大**；
2. 两条界面的最大二阶差**最小**（min-max）；
3. 两条界面总行程（相邻坡差绝对值之和）**最小**；
4. 逐列候选编号（上编号、下编号）字典序，稳定决胜。

### 算法

- 每列构造合法联合状态 `(上候选, 下候选)`（严格分离且厚度在限值内）；
- 分层 DP 状态携带 `(上一列状态, 当前列状态)`，转移时一次性校验
  两条界面的坡差、厚度变化与二阶差；启用相位连续性时同一次转移还
  校验两条界面跨边界的保持/反转关系；
- "最大二阶差"是 min-max 聚合，不能在汇合状态上按前缀值直接淘汰。
  实现上先在给定二阶差上限下求最大置信度，再对"可达的局部二阶差
  候选集合"二分最小可行上限（可行性对上限单调），最后在该上限下
  做一次（置信度, 行程, 编号路径）词典序 DP。
- 最坏规模（24 列 × 每列 8 候选）在纯标准库下亚秒级完成。

## 测试

```bash
python -m unittest discover -s tests -v   # 单元 + 随机暴力枚举对照 + 性能
BASE_URL=http://127.0.0.1:8000 python tests/smoke_api.py  # 在线 API 冒烟
```

随机模糊测试对数百个实例做全枚举交叉验证，校验可行性、全部几何
约束以及四级裁决结果的一致性。纯 Python 标准库实现，无第三方依赖。
