# 每日调仓运行指南 Daily Rebalance Guide

## 🎯 快速开始

### 方式 1: 双击运行（推荐）
```
daily_run.bat
```

### 方式 2: 强制运行（市场关闭也执行）
```
daily_run.bat --force
```

---

## 📊 如何查看是否下单？

### 1️⃣ 实时查看日志（推荐）
运行 `daily_run.bat` 后，**最后一行会显示日志文件位置**：
```
[SUCCESS] Daily run completed. Execution log: logs\daily_run_20260218_1045.log
```

### 2️⃣ 查看日志文件
所有日志自动保存到 `logs/` 文件夹：
```
logs/daily_run_20260218_1045.log
logs/daily_run_20260218_1115.log
... (每次运行都会生成新文件)
```

### 3️⃣ 日志中的关键信息

#### ✅ **下单成功** 样例：
```
============================================================
ORDER EXECUTION SUMMARY
============================================================
Total Sell Orders: 2
Total Buy Orders:  3
Skipped Trades:    1

[SELL] AAPL       $ 1500.00
       ✓ SUCCESS
[BUY]  MSFT       $ 2000.00
       ✓ SUCCESS
[BUY]  TSLA       $ 1500.00
       ✓ SUCCESS

============================================================
EXECUTION RESULT:
  Orders Submitted: 5
  Orders Failed:    0
============================================================
✓ 5 orders submitted successfully
```

#### ❌ **没有下单** 样例：
```
============================================================
ORDER EXECUTION SUMMARY
============================================================
Total Sell Orders: 0
Total Buy Orders:  0
Skipped Trades:    50

Skipped Trades (Too Small):
  - AAPL: $8.52 (below $10 threshold)
  - MSFT: $5.30 (below $10 threshold)
  ...

============================================================
EXECUTION RESULT:
  Orders Submitted: 0
  Orders Failed:    0
============================================================
⚠ NO ORDERS EXECUTED IN THIS REBALANCE!
```

---

## 🔍 完整执行流程

每次运行会输出以下步骤：

### 第 1 步：加载策略配置
```
>> STEP 1: Loading Strategy Configuration
   ✓ Loaded config/live_strategy.json
   • Universe: 100 stocks
   • Active Factors: ['Momentum', 'Reversion', 'Skew', 'Drift-Reversion']
```

### 第 2 步：波动率目标计算
```
>> STEP 2: Volatility Targeting
   • Vol Targeting: ENABLED
   • Target Vol: 20.0%
   • Max Leverage: 1.0x
   • Market Vol (20d): 18.5%. Target: 20.0%. Scalar: 1.08x
```

### 第 3 步：因子计算和组合构建
```
>> STEP 3: Factor Calculation & Portfolio Construction
   ✓ Factor Calculation Complete
   • Target Weights Generated: 5 assets
   • Exposure Scalar: 1.08x
     - NVDA: 21.60%
     - TSLA: 21.60%
     - MSFT: 21.60%
     - AAPL: 21.60%
     - AMZN: 13.60%
```

### 第 4 步：执行订单
```
>> STEP 4: Order Execution
============================================================
ORDER EXECUTION SUMMARY
...
```

---

## ⚠️ 常见原因：为什么没有下单？

### 1. 市场关闭 ❌
```
Market is CLOSED. Next Open: 2026-02-18 09:30:00-05:00
Market is Closed. Use --force to run anyway. Exiting.
```
**解决**: 使用 `--force` 参数强制运行

### 2. 调整幅度太小 ❌
```
Skipped Trades (Too Small):
  - AAPL: $8.52 (below $10 threshold)
```
**原因**: 现有权重和目标权重差距小于 $10，自动过滤以减少交易费用

**解决**: 
- 增加账户余额
- 降低 live_strategy.json 中的 `leverage`
- 修改 oms.py 中的 `MIN_TRADE_SIZE` （第 55 行）

### 3. 没有生成目标权重 ❌
```
⚠ No target weights generated. Portfolio would be empty.
Check factor calculation or symbol data availability.
```
**原因**: 因子计算失败或数据不完整

**解决**: 检查 live_strategy.json 中的 universe 和 factors 配置

### 4. API 连接问题 ❌
```
Failed to get account info. Aborting rebalance.
```
**原因**: Alpaca API 密钥错误或网络问题

**解决**: 
- 检查 .env 文件中的 APCA_API_KEY_ID 和 APCA_API_SECRET_KEY
- 检查网络连接

---

## 📋 配置调整

### 文件位置
```
config/live_strategy.json
```

### 关键配置项

```json
{
    "universe": ["AAPL", "MSFT", ...],     // 交易股票列表
    "active_factors": ["Momentum", ...],    // 使用的因子
    "use_vol_target": true,                 // 是否启用波动率目标
    "vol_target": 0.20,                     // 目标波动率（20%）
    "leverage": 1.0                         // 最大杠杆倍数
}
```

---

## 🔄 定时自动运行（Windows 任务计划）

### 创建每日任务
1. 打开 `任务计划程序` (Task Scheduler)
2. 创建基本任务
3. **触发器**: 每天 09:35 AM（开市 5 分钟后）
4. **操作**: 
   - 程序: `daily_run.bat`
   - 位置: `c:\Users\Allen\Documents\ancserAlpacaLab\`

---

## 📧 获取帮助

- 检查 `logs/` 文件夹中的最新日志
- 查看错误信息和堆栈跟踪
- 确认 .env 文件配置正确

