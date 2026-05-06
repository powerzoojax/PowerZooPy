# Market Environments

## Base Class

::: powerzoo.envs.market.base.MarketEnv
    options:
      show_source: false
      members:
        - step
        - clear
        - settle
        - revenue

---

## CostBasedMarketEnv (cost-based dispatch)

Cost-based LMP arbitrage environment.  Generators are dispatched by a
linear-cost DC-OPF (`mc_c @ p`); there is no bid–cost separation.

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

## BidBasedMarketEnv (piecewise-linear offers)

Competitive market environment with explicit piecewise-linear offer
curves.  LMP is derived from offer-based dispatch (not true costs),
enabling realistic bid–cost separation and strategic bidding research.

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
