"""Phase 1: 检测器仿真.

检验 theory_draft.md 的三个论断，全程不调用任何 LLM——自变量是聚合规则与校准方法,
不是 trigger 措辞.

信号模型 (theory_draft.md A2/A4):
    H0: 每个推理步的分数 ~ F0
    H1(N): 比例 eps_N = eps_0 * rho^N 的步携带信号 ~ F1, 其余 ~ F0

三种判定规则:
    raw_sum   预印本 Algorithm 4: 直接对原始相似度求和, 阈值靠调
    conf_sum  共形 p 值 + 求和型聚合 (Tr-GoF 证明次优的那一族)
    conf_trgof 共形 p 值 + 截断拟合优度聚合
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 信号模型


def sample_scores(rng, n, eps, mu, n_trials):
    """返回 (n_trials, n) 的分数矩阵; 每步以概率 eps 携带信号."""
    carries = rng.random((n_trials, n)) < eps
    return rng.standard_normal((n_trials, n)) + mu * carries


def hop_eps(eps0, rho, n_hops):
    return eps0 * rho**n_hops


# ---------------------------------------------------------------- 共形校准


def conformal_pvalues(scores, calib):
    """干净轨迹校准: p_t = (1 + #{j: calib_j >= s_t}) / (m + 1).

    可交换性建在"候选轨迹 vs 干净轨迹语料"上 —— 跨语料分布漂移即失效 (EXP-001/E1).
    """
    calib_sorted = np.sort(calib)
    m = calib_sorted.size
    # #{j: calib_j >= s} = m - searchsorted(calib, s, 'left')
    ge = m - np.searchsorted(calib_sorted, scores, side="left")
    return (1.0 + ge) / (m + 1.0)


def wrongkey_pvalues(rng, scores_true, n_keys, mu_shift=0.0):
    """Wrong-key 校准 (移植自 SeqWM 2605.11036 Sec 4.3).

    对同一条候选轨迹, 用 n_keys 个错误密钥各生成一个 trigger pattern tau'=T(k',t),
    重算分数, 构成该轨迹自身的经验零分布. 可交换性由构造保证:
    在 H0 下真密钥与错误密钥地位对称, 故真分数在 n_keys+1 个分数中的秩均匀.

    关键差异: 零分布随候选轨迹一起漂移, 因此对分布漂移免疫.

    scores_true: (n_trials, n) 真密钥下的逐步分数
    返回: (n_trials, n) 逐步 p 值
    """
    n_trials, n = scores_true.shape
    # 错误密钥分数与真密钥分数共享同一条轨迹的基线漂移 mu_shift
    wrong = rng.standard_normal((n_keys, n_trials, n)) + mu_shift
    ge = (wrong >= scores_true[None, :, :]).sum(axis=0)
    return (1.0 + ge) / (n_keys + 1.0)


# ---------------------------------------------------------------- 聚合规则


def stat_raw_sum(scores, pvals):
    return scores.sum(axis=1)


def stat_conf_sum(scores, pvals):
    """求和型: Sigma_t -log(p_t) (Fisher). 属 Tr-GoF 证明次优的可加族."""
    return -np.log(np.clip(pvals, 1e-12, None)).sum(axis=1)


def _phi_divergence(u, v, s=2.0):
    """Ber(u) 与 Ber(v) 间的 phi_s 散度. s=2 即卡方型 (Higher Criticism 家族)."""
    u = np.clip(u, 1e-12, 1 - 1e-12)
    v = np.clip(v, 1e-12, 1 - 1e-12)
    if abs(s - 1.0) < 1e-9:  # KL
        return u * np.log(u / v) + (1 - u) * np.log((1 - u) / (1 - v))
    if abs(s) < 1e-9:
        return v * np.log(v / u) + (1 - v) * np.log((1 - v) / (1 - u))
    num = u**s * v ** (1 - s) + (1 - u) ** s * (1 - v) ** (1 - s) - 1.0
    return num / (s * (s - 1.0))


def stat_conf_trgof(scores, pvals, s=2.0, p_plus=None):
    """Tr-GoF: sup_{r >= p_+} K_s^+(F_n(r), r), 截断到 F_n(r) > r 的一侧.

    在排序后的 p 值处取 sup 即可 (经验 CDF 是阶梯函数).
    """
    n = pvals.shape[1]
    if p_plus is None:
        p_plus = 1.0 / n  # 滤掉极端小 p 值以保稳定
    ps = np.sort(pvals, axis=1)
    fn = np.arange(1, n + 1)[None, :] / n  # F_n 在第 i 个 p 值处的取值
    div = _phi_divergence(fn, ps, s=s)
    mask = (fn > ps) & (ps >= p_plus)  # 截断: 仅取超出零分布的一侧
    div = np.where(mask, div, 0.0)
    return div.max(axis=1)


DETECTORS = {
    "raw_sum": stat_raw_sum,
    "conf_sum": stat_conf_sum,
    "conf_trgof": stat_conf_trgof,
}


# ---------------------------------------------------------------- 校准与功效


def null_threshold(rng, name, n, calib, alpha, n_null, mu_null_shift=0.0):
    """蒙特卡洛求 H0 下统计量的 (1-alpha) 分位数, 即临界值."""
    null_scores = rng.standard_normal((n_null, n)) + mu_null_shift
    pv = conformal_pvalues(null_scores, calib)
    stats = DETECTORS[name](null_scores, pv)
    return np.quantile(stats, 1 - alpha)


def empirical_fpr(rng, name, n, calib, thresh, n_trials, mu_null_shift=0.0):
    """在(可能发生分布漂移的)真实 H0 下测经验 FPR."""
    null_scores = rng.standard_normal((n_trials, n)) + mu_null_shift
    pv = conformal_pvalues(null_scores, calib)
    stats = DETECTORS[name](null_scores, pv)
    return float((stats >= thresh).mean())


def empirical_power(rng, name, n, calib, thresh, eps, mu, n_trials):
    scores = sample_scores(rng, n, eps, mu, n_trials)
    pv = conformal_pvalues(scores, calib)
    stats = DETECTORS[name](scores, pv)
    return float((stats >= thresh).mean())
