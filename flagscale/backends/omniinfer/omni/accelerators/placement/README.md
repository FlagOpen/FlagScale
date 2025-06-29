# omni-placement

OmniPlacement provides a framework to optimize the token to expert mapping, the goal of the
optimization is to achieve the best possible through put with a targeted latency.

The token to expert mapping is modelled as `[T, ES]` where `ES` is the score of each expert
representing it's importance.

Each mapping is assigned a 'cost'， which is an approximation of execution latency.

TODO:
1. single token optimization
2. multi token optimizationm

Goal:
1. minimize cross machine communication --> this might be the highest priority
2. minimize corss device communication
3. minimize computation cost --> this might not be the bottleneck but it could simplify the
optimization of the corss machine/device communication

1. 先装c++.so
2. pip install -e ., 自动挂载.so