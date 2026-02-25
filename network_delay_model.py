"""网络时延建模示例（纯标准库版本）。

功能：
1. 采用 M/G/1 排队模型生成排队+服务时延。
2. 叠加传播、处理、基站间路由与抖动噪声，得到端到端时延样本。
3. 用对数正态分布拟合端到端时延并输出 SLA 统计。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, log, sqrt
import random
import statistics
from typing import Dict, List, Tuple


@dataclass
class DelayConfig:
    """网络时延模型参数（单位：毫秒）。"""

    # M/G/1 参数（G: 一般服务时间分布，这里默认使用对数正态）
    arrival_rate_pps: float = 1300.0
    mean_service_time_ms: float = 0.45
    service_time_cv: float = 0.6  # 服务时间变异系数（std/mean）

    # 固定项
    base_propagation_ms: float = 8.0
    processing_ms: float = 0.9

    # 基站间路由参数
    bs_hops_mean: float = 3.5
    bs_hops_std: float = 1.0
    per_hop_fiber_ms: float = 0.6
    per_hop_switch_ms: float = 0.25
    routing_jitter_std_ms: float = 0.35

    # 接入侧其他抖动
    jitter_std_ms: float = 1.2

    samples: int = 20_000
    random_seed: int = 42


class NetworkDelayModel:
    """网络时延仿真 + 统计建模。"""

    def __init__(self, cfg: DelayConfig):
        if cfg.arrival_rate_pps <= 0:
            raise ValueError("arrival_rate_pps 必须大于 0")
        if cfg.mean_service_time_ms <= 0:
            raise ValueError("mean_service_time_ms 必须大于 0")
        if cfg.service_time_cv <= 0:
            raise ValueError("service_time_cv 必须大于 0")
        if cfg.bs_hops_mean <= 0:
            raise ValueError("bs_hops_mean 必须大于 0")

        # M/G/1 稳定条件 rho=lambda*E[S] < 1
        self.arrival_rate = cfg.arrival_rate_pps
        self.mean_service_s = cfg.mean_service_time_ms / 1000.0
        rho = self.arrival_rate * self.mean_service_s
        if rho >= 1.0:
            raise ValueError(f"队列不稳定: rho={rho:.3f}，需满足 rho<1")

        self.cfg = cfg
        self.rho = rho
        self.rng = random.Random(cfg.random_seed)

        # 将给定 mean + cv 转换为对数正态底层正态参数
        # 若 S~LogNormal(mu, sigma^2):
        # mean = exp(mu + sigma^2/2), cv^2 = exp(sigma^2)-1
        self._svc_sigma = sqrt(log(1.0 + cfg.service_time_cv**2))
        self._svc_mu = log(self.mean_service_s) - 0.5 * self._svc_sigma**2

    def _sample_service_time_s(self) -> float:
        """采样一般服务时间分布 G（默认对数正态）。"""
        return self.rng.lognormvariate(self._svc_mu, self._svc_sigma)

    def _simulate_bs_routing_delay_ms(self) -> float:
        """模拟基站间路由时延：跳数 × 每跳时延 + 路由抖动。"""
        hops = max(1, int(round(self.rng.gauss(self.cfg.bs_hops_mean, self.cfg.bs_hops_std))))
        per_hop_ms = self.cfg.per_hop_fiber_ms + self.cfg.per_hop_switch_ms
        routing_base_ms = hops * per_hop_ms
        routing_jitter = self.rng.gauss(0.0, self.cfg.routing_jitter_std_ms)
        return max(0.01, routing_base_ms + routing_jitter)

    def simulate_delay_ms(self) -> List[float]:
        """生成端到端时延样本（毫秒）。

        使用 Lindley 递推模拟 M/G/1 队列：
        W_{n+1} = max(0, W_n + S_n - A_n)
        其中 A_n 为泊松到达下的相邻到达间隔（指数分布）。
        """
        samples: List[float] = []
        waiting_s = 0.0

        for _ in range(self.cfg.samples):
            inter_arrival_s = self.rng.expovariate(self.arrival_rate)
            service_s = self._sample_service_time_s()

            # 当前包在系统中的排队+服务时延
            queue_and_service_s = waiting_s + service_s

            # 更新下一包等待时间（Lindley recursion）
            waiting_s = max(0.0, waiting_s + service_s - inter_arrival_s)

            routing_ms = self._simulate_bs_routing_delay_ms()
            access_jitter_ms = self.rng.gauss(0.0, self.cfg.jitter_std_ms)

            total_ms = (
                queue_and_service_s * 1000.0
                + self.cfg.base_propagation_ms
                + self.cfg.processing_ms
                + routing_ms
                + access_jitter_ms
            )
            samples.append(max(0.01, total_ms))

        return samples

    @staticmethod
    def fit_lognormal(samples_ms: List[float]) -> Tuple[float, float]:
        """拟合对数正态参数(mu, sigma)，其中 log(X) ~ N(mu, sigma^2)。"""
        log_x = [log(x) for x in samples_ms]
        mu = statistics.mean(log_x)
        sigma = statistics.stdev(log_x)
        return mu, sigma

    @staticmethod
    def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
        """对数正态分布 CDF。"""
        if x <= 0:
            return 0.0
        z = (log(x) - mu) / (sigma * sqrt(2.0))
        return 0.5 * (1.0 + erf(z))

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """线性插值分位数，p 取值 [0,100]。"""
        if not values:
            raise ValueError("values 不能为空")
        if p <= 0:
            return min(values)
        if p >= 100:
            return max(values)

        sorted_vals = sorted(values)
        pos = (len(sorted_vals) - 1) * p / 100.0
        low = int(pos)
        high = min(low + 1, len(sorted_vals) - 1)
        frac = pos - low
        return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac

    @classmethod
    def summarize(cls, samples_ms: List[float]) -> Dict[str, float]:
        """输出常用 SLA 指标。"""
        return {
            "mean_ms": statistics.mean(samples_ms),
            "std_ms": statistics.stdev(samples_ms),
            "p50_ms": cls.percentile(samples_ms, 50),
            "p95_ms": cls.percentile(samples_ms, 95),
            "p99_ms": cls.percentile(samples_ms, 99),
            "max_ms": max(samples_ms),
        }


def main() -> None:
    cfg = DelayConfig()
    model = NetworkDelayModel(cfg)

    samples = model.simulate_delay_ms()
    mu, sigma = model.fit_lognormal(samples)
    stats = model.summarize(samples)

    print("=== 网络时延统计（ms）===")
    for k, v in stats.items():
        print(f"{k:>8}: {v:8.3f}")

    print("\n=== M/G/1 参数 ===")
    print(
        f"lambda={cfg.arrival_rate_pps:.1f}pps, "
        f"E[S]={cfg.mean_service_time_ms:.3f}ms, "
        f"CV={cfg.service_time_cv:.3f}, rho={model.rho:.3f}"
    )

    print("\n=== 基站间路由建模参数 ===")
    print(
        f"hops~N({cfg.bs_hops_mean:.1f},{cfg.bs_hops_std:.1f}), "
        f"per_hop={cfg.per_hop_fiber_ms + cfg.per_hop_switch_ms:.2f}ms"
    )

    threshold = 20.0
    prob_le_threshold = model.lognormal_cdf(threshold, mu, sigma)
    print("\n=== 拟合分布参数 ===")
    print(f"mu={mu:.4f}, sigma={sigma:.4f}")
    print(f"P(delay <= {threshold:.1f}ms) ≈ {prob_le_threshold:.4%}")

    sla_p99_ms = 25.0
    if stats["p99_ms"] > sla_p99_ms:
        print(f"告警: P99={stats['p99_ms']:.2f}ms 超过 SLA {sla_p99_ms:.2f}ms")
    else:
        print(f"通过: P99={stats['p99_ms']:.2f}ms 满足 SLA {sla_p99_ms:.2f}ms")


if __name__ == "__main__":
    main()
