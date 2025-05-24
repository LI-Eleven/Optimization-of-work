from pulp import *

# 城市信息（需求，库存容量，库存成本）
cities = ['太原', '郑州', '西安', '北京', '大同', '济南']
demand = {
    '太原': -sum([220, 190, 210, 160, 250]),
    '郑州': 220, '西安': 190, '北京': 210, '大同': 160, '济南': 250
}
stock_capacity = {
    '太原': 3000, '郑州': 1100, '西安': 1000,
    '北京': 1200, '大同': 800, '济南': 1200
}
stock_cost = {
    '太原': 0.9, '郑州': 0.7, '西安': 0.8,
    '北京': 0.5, '大同': 0.6, '济南': 0.7
}

# 车辆参数（时间限制设为10小时）
vehicle_types = ['中型', '大型']
vehicle_params = {
    '中型': {'capacity': 200, 'fee': 300, 'fuel': 1.8, 'time_limit': 10},
    '大型': {'capacity': 300, 'fee': 420, 'fuel': 2.2, 'time_limit': 10}
}
max_vehicles = 3  # 最大可用车辆数
vehicles = [(v_type, num) for v_type in vehicle_types for num in range(max_vehicles)]

# 距离与驾驶时间数据（对称矩阵）
dij = {
    ('太原', '郑州'): 500, ('太原', '西安'): 600, ('太原', '北京'): 300,
    ('太原', '大同'): 150, ('太原', '济南'): 550, ('郑州', '西安'): 400,
    ('郑州', '北京'): 600, ('郑州', '大同'): 380, ('郑州', '济南'): 420,
    ('西安', '北京'): 900, ('西安', '大同'): 500, ('西安', '济南'): 800,
    ('北京', '大同'): 320, ('北京', '济南'): 400, ('大同', '济南'): 600
}
dij.update({(j, i): d for (i, j), d in dij.items()})

transit_time = {
    ('太原', '郑州'): 3.0, ('太原', '西安'): 4.0, ('太原', '北京'): 3.5,
    ('太原', '大同'): 1.5, ('太原', '济南'): 4.0, ('郑州', '西安'): 2.5,
    ('郑州', '北京'): 4.0, ('郑州', '大同'): 3.5, ('郑州', '济南'): 3.5,
    ('西安', '北京'): 5.0, ('西安', '大同'): 4.0, ('西安', '济南'): 6.0,
    ('北京', '大同'): 4.5, ('北京', '济南'): 4.5, ('大同', '济南'): 5.0
}
transit_time.update({(j, i): t for (i, j), t in transit_time.items()})

# 创建模型
prob = LpProblem("Logistics_Optimization_Closed_Path_Corrected", LpMinimize)

# ========== 决策变量 ==========
x = LpVariable.dicts("x",
                     [(i, j, v_type, num) for i in cities for j in cities
                      for v_type in vehicle_types for num in range(max_vehicles) if i != j],
                     cat=LpBinary)
z = LpVariable.dicts("z", vehicles, cat=LpBinary)
Q = LpVariable.dicts("Q",
                     [(i, j, v_type, num) for i in cities for j in cities
                      for v_type in vehicle_types for num in range(max_vehicles) if i != j],
                     lowBound=0)
I = LpVariable.dicts("I", cities, lowBound=0)

# ========== 目标函数（关键修正点） ==========
# 运输成本按趟次计算（不再乘以Q）
transport_cost = lpSum(
    dij[i, j] * vehicle_params[v_type]['fuel'] * x[i, j, v_type, num]
    for (i, j, v_type, num) in x
)
fixed_cost = lpSum(vehicle_params[v_type]['fee'] * z[v_type, num] for (v_type, num) in vehicles)
inventory_cost = lpSum(stock_cost[i] * I[i] for i in cities)
prob += transport_cost + fixed_cost + inventory_cost

# ========== 约束条件 ==========
# 1. 流量平衡约束（含太原）
for city in cities:
    if city == '太原':
        inflow = lpSum(Q[j, city, v_type, num] for j in cities for v_type, num in vehicles if j != city)
        outflow = lpSum(Q[city, j, v_type, num] for j in cities for v_type, num in vehicles if j != city)
        prob += outflow - inflow - demand[city] == I[city]
    else:
        inflow = lpSum(Q[j, city, v_type, num] for j in cities for v_type, num in vehicles if j != city)
        outflow = lpSum(Q[city, j, v_type, num] for j in cities for v_type, num in vehicles if j != city)
        prob += inflow - outflow - demand[city] == I[city]

# 2. 车辆路径连续性约束（关键改进）
for v_type, num in vehicles:
    # 必须从太原出发并返回
    prob += lpSum(x['太原', j, v_type, num] for j in cities if j != '太原') == z[v_type, num]
    prob += lpSum(x[i, '太原', v_type, num] for i in cities if i != '太原') == z[v_type, num]

    # 中间城市进出平衡
    for city in cities:
        if city != '太原':
            inflow = lpSum(x[i, city, v_type, num] for i in cities if i != city)
            outflow = lpSum(x[city, j, v_type, num] for j in cities if j != city)
            prob += inflow == outflow

# 3. 容量与时间约束
for (i, j, v_type, num) in Q:
    prob += Q[i, j, v_type, num] <= vehicle_params[v_type]['capacity'] * x[i, j, v_type, num]

for v_type, num in vehicles:
    prob += lpSum(
        transit_time.get((i, j), 0) * x[i, j, v_type, num]
        for i in cities for j in cities if i != j
    ) <= vehicle_params[v_type]['time_limit']

# 4. 库存约束
for city in cities:
    prob += I[city] >= 0
    prob += I[city] <= stock_capacity[city]

# 求解
prob.solve(PULP_CBC_CMD(msg=True, timeLimit=60))  # 增加60秒求解时间限制

# 结果输出
print(f"\n求解状态: {LpStatus[prob.status]}")
if LpStatus[prob.status] == 'Optimal':
    print(f"总成本: {value(prob.objective):.2f} 元")

    # 统计使用的车辆
    used_vehicles = [(v, n) for (v, n) in vehicles if value(z[v, n]) > 0.5]
    print("\n使用的车辆:")
    for v, n in used_vehicles:
        print(f"  {v}货车-{n + 1}号")

    # 输出运输路径
    print("\n运输路径详情:")
    path_details = {}
    for (i, j, v_type, num) in x:
        if value(x[i, j, v_type, num]) > 0.5:
            key = (v_type, num)
            if key not in path_details:
                path_details[key] = []
            path_details[key].append((i, j, value(Q[i, j, v_type, num])))

    for (v_type, num), routes in path_details.items():
        print(f"\n{v_type}货车-{num + 1}号路径:")
        total_time = sum(transit_time[i, j] for (i, j, _) in routes)
        for i, j, q in routes:
            print(f"  {i}→{j}  运量: {q:.0f}箱  时间: {transit_time[i, j]}小时")
        print(f"  总行驶时间: {total_time:.1f}小时 (限额: {vehicle_params[v_type]['time_limit']}小时)")

    # 库存输出
    print("\n各城市库存:")
    for city in cities:
        print(f"  {city}: {value(I[city]):.1f}箱 (容量上限: {stock_capacity[city]})")
else:
    print("\n未找到可行解. 检查建议:")
    print("- 是否车辆时间限制过紧（尝试放宽时间限制至12小时）")
    print("- 是否车辆数量不足（尝试增大max_vehicles至15）")
    print("- 检查需求与库存容量是否匹配（如太原库存容量是否足够存放未运输货物）")