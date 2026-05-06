# 市场环境

## 基类

::: powerzoo.envs.market.base.MarketEnv
    options:
      show_source: false
      members:
        - step
        - clear
        - settle
        - revenue

---

## CostBasedMarketEnv（基于成本的分配）

基于成本的 LMP 套利环境。发电机由线性成本 DC-OPF（`mc_c @ p`）分配；不存在报价-成本分离。

::: powerzoo.envs.market.cost_based_market.CostBasedMarketEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step
        - render
        - close
        - steps_per_day

---

## BidBasedMarketEnv（分段线性报价）

带显式分段线性报价曲线的竞争式市场环境。LMP 由基于报价的分配（而非真实成本）派生，使真实的报价-成本分离与策略性报价研究成为可能。

::: powerzoo.envs.market.bid_based_market.BidBasedMarketEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step
        - render
        - close
        - steps_per_day
