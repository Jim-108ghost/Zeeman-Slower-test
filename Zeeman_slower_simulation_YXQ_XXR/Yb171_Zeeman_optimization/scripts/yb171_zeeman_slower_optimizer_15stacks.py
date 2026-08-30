# -*- coding: utf-8 -*-
"""
Permanent-magnet Zeeman slower optimizer
----------------------------------------
按照 Yu Xianquan 博士论文 Chapter 7 的思路：
1) 给定目标磁场 B_desired(z)
2) 用经验缩放因子 a 生成每个 longitudinal stack pair 的磁铁数 n_i
3) 固定 n_i，对每一对 stack 到原子束轴的距离 d_i 做 coordinate descent
4) 外层自动扫描 a，选取总磁场误差最小的方案

几何约定
--------
x : 原子束传播方向
z : transverse magnetic field 方向
第 i 对 stack 位于 x_i。
上/下 stack 关于原子束轴对称。

n_i 表示“单侧 stack 中的磁铁块数”：
    upper stack: n_i blocks
    lower stack: n_i blocks
所以第 i 对 stack 的实际磁铁总数是 2*n_i。

两种磁场模型
------------
stack_model="paper":
    严格按照论文/论文截图中的 point-dipole stack 思路，
    把单侧 n_i 块磁铁等效成位于 z=+d_i 的一个磁偶极矩 n_i*m，
    下侧等效放在 z=-d_i。

stack_model="resolved":
    更适合你的 8 mm × 3 mm 薄圆片磁铁。
    把每一块圆片磁铁都当作独立 point dipole，
    单侧 n_i 块沿 z 方向堆叠，stack 的几何中心位于 z=±d_i。
    这个模型仍然是 magnetic-dipole approximation，但不再把整叠磁铁压缩到一个点。

注意：
point-dipole approximation 在磁铁尺寸远小于观测距离时最可靠。
最终机械设计完成后，建议再用 FEMM / COMSOL / Radia 做一次验证。
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

MU0 = 4 * np.pi * 1e-7


@dataclass
class ZeemanSlowerConfig:
    # ---------- target field ----------
    csv_path: str
    L_m: float = 0.185
    target_mode: str = "crop"
    # "crop"   : CSV 中直接取扩展拟合区间的目标场
    # "rescale": 把 CSV 的完整 z 范围线性压缩/拉伸到 [0, L]

    # 有效 Zeeman slower 仍是 [0, L]；这两个参数只扩展磁场拟合网格，
    # 用区间外趋势抑制有限磁阵列在两端产生的边界误差。
    fit_left_extension_m: float = 0.0
    fit_right_extension_m: float = 0.0
    extension_fit_weight: float = 1.0
    # 外推数据离物理端点越远，磁阵列越不可能继续追踪。使用高斯
    # 衰减只保留紧邻端点的趋势信息，防止远端大残差支配目标函数。
    extension_fit_decay_m: float = 3e-3

    # ---------- magnet array ----------
    n_stacks: int = 15

    # 整个装置（上下两侧合计）的磁铁数量硬上限。None 表示不限制。
    # 每一对 stack 实际使用 2*n_i 块磁铁。
    max_total_magnets: int | None = None

    # 若非空，则严格使用这里给出的 longitudinal stack positions，
    # 并忽略 stack_edge_extension_m。单位为 m，数量必须等于 n_stacks。
    x_stack_positions_m: tuple = ()

    # 为减小有限磁阵列在 0 和 L 两端的边界误差，允许首末 stack
    # 的中心略微伸出有效减速区。有效目标场仍只在 [0, L] 内拟合。
    stack_edge_extension_m: float = 0.0

    # N35 disk magnet: diameter 8 mm, thickness 3 mm, Br = 1.2 T
    Br_T: float = 1.2
    magnet_diameter_m: float = 8e-3
    magnet_thickness_m: float = 3e-3
    magnet_gap_m: float = 0.0

    # ---------- d_i ----------
    # Yu thesis initializes all d_i at 50 mm.
    d0_m: float = 50e-3

    # 下面两个只是目前根据参考 CAD 给出的“暂定机械范围”。
    # 有了你自己的真空管 OD / frame 尺寸后，应改成真实限制。
    d_min_m: float = 20e-3
    d_max_m: float = 100e-3

    # coordinate-descent 的多级步长。
    # 若要最严格模拟论文的固定 delta-d，可改成例如 (0.5e-3,)
    d_steps_m: tuple = (2e-3, 1e-3, 0.5e-3)

    # ---------- scan a ----------
    a_min: float = 0.5
    a_max: float = 10.0
    a_step: float = 0.05

    # ---------- numerical ----------
    n_sample: int = 601
    max_sweeps_each_step: int = 30

    # 首尾使用完全对称的权重，同时保留中段和全长误差。
    start_fit_region_m: float = 30e-3
    start_fit_weight: float = 1.0
    end_fit_region_m: float = 30e-3
    end_fit_weight: float = 1.0

    # 在加权 MSE 之外惩罚有效区内的最大单点误差，避免首尾端点
    # 因只占少量采样点而被均方误差忽略。
    max_abs_error_weight: float = 0.0

    # 外层 a 扫描只产生一组整数 n_i 的初值；随后逐 stack 尝试
    # n_i +/- step，并为每个候选重新优化 d_i，以细化整数磁铁数量。
    refine_integer_counts: bool = True
    count_refinement_steps: tuple = (2, 1)
    max_count_refinement_sweeps: int = 5

    # 离散 coordinate descent 结束后，用 Powell 对全部 d_i 做一次
    # 连续联合优化，以消除有限步长以及逐坐标移动造成的局部残差。
    continuous_distance_polish: bool = True
    continuous_polish_maxiter: int = 500

    # "paper" or "resolved"
    stack_model: str = "resolved"

    # ---------- vacuum tube / mechanical clearance ----------
    tube_outer_diameter_m: float = 16e-3
    # 磁铁内表面与真空管外壁之间的最小空气/机械间隙。
    # 用户暂未指定，因此默认 0；实际加工时建议改为真实值。
    tube_clearance_m: float = 0.0


class PermanentMagnetZeemanSlower:
    def __init__(self, cfg: ZeemanSlowerConfig):
        self.cfg = cfg
        self._validate_config()

        self.magnet_volume_m3 = (
            np.pi * (cfg.magnet_diameter_m / 2) ** 2 * cfg.magnet_thickness_m
        )
        # Uniform-magnetization estimate: m = M V = Br V / mu0
        self.single_magnet_moment_Am2 = (
            cfg.Br_T * self.magnet_volume_m3 / MU0
        )

        self.z_raw_m, self.B_raw_T = self._load_target_csv(cfg.csv_path)

        base_grid_m = np.linspace(
            -cfg.fit_left_extension_m,
            cfg.L_m + cfg.fit_right_extension_m,
            cfg.n_sample,
        )
        # 扩展区与有效区拼接后，普通 linspace 往往不会恰好包含
        # z=0 和 z=L。显式加入物理端点，确保优化和误差统计不会漏点。
        self.z_m = np.unique(
            np.concatenate((base_grid_m, np.array([0.0, cfg.L_m])))
        )
        self.B_target_T = self._make_target_on_design_grid()

        if cfg.x_stack_positions_m:
            # 用户指定的位置是硬约束，优化中不再改变 x_i。
            self.x_stack_m = np.asarray(
                cfg.x_stack_positions_m,
                dtype=float,
            ).copy()
        else:
            # 默认按论文思路均匀分布；首末位置可略微伸出有效减速区，
            # 用作边界场整形的 guard stacks。
            self.x_stack_m = np.linspace(
                -cfg.stack_edge_extension_m,
                cfg.L_m + cfg.stack_edge_extension_m,
                cfg.n_stacks,
            )
        self.B_target_at_stack_T = np.interp(
            self.x_stack_m, self.z_m, self.B_target_T
        )

        # 正/负目标场对应把整对 stack 的磁化方向翻转
        self.orientation = np.where(
            self.B_target_at_stack_T >= 0.0, 1.0, -1.0
        )

        # 论文式 n_i 初始化中的 B_single：
        # 一块上磁铁 + 一块下磁铁，在 x=x_i, z=0 产生的场
        self.B_single_at_d0_T = abs(
            self._dipole_pair_field(
                np.array([0.0]),
                x_i=0.0,
                d=cfg.d0_m,
                moment=self.single_magnet_moment_Am2,
            )[0]
        )

    def _validate_config(self):
        c = self.cfg
        if c.stack_model not in ("paper", "resolved"):
            raise ValueError("stack_model 必须是 'paper' 或 'resolved'")
        if c.target_mode not in ("crop", "rescale"):
            raise ValueError("target_mode 必须是 'crop' 或 'rescale'")
        if c.max_total_magnets is not None:
            if int(c.max_total_magnets) != c.max_total_magnets:
                raise ValueError("max_total_magnets 必须是整数或 None")
            if c.max_total_magnets < 0:
                raise ValueError("max_total_magnets 不能为负数")
        if c.d_min_m <= 0 or c.d_max_m <= c.d_min_m:
            raise ValueError("需要 0 < d_min_m < d_max_m")
        if not (c.d_min_m <= c.d0_m <= c.d_max_m):
            raise ValueError("d0_m 必须在 [d_min_m, d_max_m] 内")
        if c.a_min <= 0 or c.a_max < c.a_min or c.a_step <= 0:
            raise ValueError("a 扫描范围不合法")
        if c.stack_edge_extension_m < 0:
            raise ValueError("stack_edge_extension_m 不能为负数")
        if c.x_stack_positions_m:
            x_stack = np.asarray(c.x_stack_positions_m, dtype=float)
            if x_stack.size != c.n_stacks:
                raise ValueError(
                    "x_stack_positions_m 的数量必须等于 n_stacks"
                )
            if not np.all(np.isfinite(x_stack)):
                raise ValueError("x_stack_positions_m 中存在非有限值")
            if np.any(np.diff(x_stack) <= 0):
                raise ValueError("x_stack_positions_m 必须严格递增")
        if c.max_count_refinement_sweeps < 1:
            raise ValueError("max_count_refinement_sweeps 必须至少为 1")
        if any(int(step) <= 0 for step in c.count_refinement_steps):
            raise ValueError("count_refinement_steps 必须为正整数")
        if c.start_fit_region_m < 0:
            raise ValueError("start_fit_region_m 不能为负数")
        if c.start_fit_weight < 1:
            raise ValueError("start_fit_weight 必须不小于 1")
        if c.end_fit_region_m < 0:
            raise ValueError("end_fit_region_m 不能为负数")
        if c.end_fit_weight < 1:
            raise ValueError("end_fit_weight 必须不小于 1")
        if c.fit_left_extension_m < 0 or c.fit_right_extension_m < 0:
            raise ValueError("拟合区间的左右扩展长度不能为负数")
        if c.extension_fit_weight < 0:
            raise ValueError("extension_fit_weight 不能为负数")
        if c.extension_fit_decay_m <= 0:
            raise ValueError("extension_fit_decay_m 必须大于 0")
        if c.max_abs_error_weight < 0:
            raise ValueError("max_abs_error_weight 不能为负数")

    @staticmethod
    def _load_target_csv(csv_path):
        df = pd.read_csv(csv_path)

        if "z_m" in df.columns:
            z = df["z_m"].to_numpy(dtype=float)
        elif "z_cm" in df.columns:
            z = df["z_cm"].to_numpy(dtype=float) * 1e-2
        elif "z_mm" in df.columns:
            z = df["z_mm"].to_numpy(dtype=float) * 1e-3
        else:
            raise ValueError(
                "CSV 中需要 z_m、z_cm 或 z_mm 其中一列。"
            )
        if "B_T" in df.columns:
            B = df["B_T"].to_numpy(dtype=float)
        elif "B_mT" in df.columns:
            B = df["B_mT"].to_numpy(dtype=float) * 1e-3
        elif "B_G" in df.columns:
            B = df["B_G"].to_numpy(dtype=float) * 1e-4
        else:
            raise ValueError(
                "CSV 中需要 B_T、B_mT 或 B_G 其中一列。"
            )

        good = np.isfinite(z) & np.isfinite(B)
        z = z[good]
        B = B[good]

        order = np.argsort(z)
        z = z[order]
        B = B[order]

        # 去掉重复 z
        z_unique, idx = np.unique(z, return_index=True)
        B_unique = B[idx]
        return z_unique, B_unique

    def _make_target_on_design_grid(self):
        c = self.cfg

        if c.target_mode == "crop":
            z_fit_min = -c.fit_left_extension_m
            z_fit_max = c.L_m + c.fit_right_extension_m
            if (
                self.z_raw_m.min() > z_fit_min
                or self.z_raw_m.max() < z_fit_max
            ):
                raise ValueError(
                    f"CSV 的 z 范围 [{self.z_raw_m.min():.6f}, "
                    f"{self.z_raw_m.max():.6f}] m 不能覆盖 "
                    f"[{z_fit_min}, {z_fit_max}] m。"
                )
            return np.interp(self.z_m, self.z_raw_m, self.B_raw_T)

        # 保留整条曲线形状和两端场值，仅改变横坐标总长度
        z0 = self.z_raw_m.min()
        z1 = self.z_raw_m.max()
        z_scaled = (self.z_raw_m - z0) / (z1 - z0) * c.L_m
        return np.interp(self.z_m, z_scaled, self.B_raw_T)

    @staticmethod
    def _dipole_pair_field(z, x_i, d, moment):
        """
        上、下两个 magnetic dipoles 位于
            (x_i, +d), (x_i, -d)
        且磁矩都沿 +z。

        原子束轴上 (x=z_input, z=0) 的 z 分量为

        Bz = mu0*m/(2*pi) *
             [2 d^2 - (x-x_i)^2] /
             [(x-x_i)^2 + d^2]^(5/2)

        这里的 z 参数其实是 beam-axis coordinate；沿袭程序中 z_m 的命名，
        因为用户 CSV 用 z 表示 slower longitudinal coordinate。
        """
        dx = z - x_i
        r2 = dx * dx + d * d
        return (
            MU0
            * moment
            / (2 * np.pi)
            * (2 * d * d - dx * dx)
            / (r2 ** 2.5)
        )

    def _effective_lower_d_bound(self, n_i):
        """
        resolved 模型中，d_i 定义为单侧整个 stack 的几何中心到 beam axis 的距离。

        真空管外半径:
            R_tube = tube_outer_diameter / 2

        最靠近管壁的那块磁铁，其中心至少要满足:
            r_nearest_center >= R_tube + clearance + t/2

        对 n_i 块、中心间距 pitch 的 stack:
            r_nearest_center = d_i - (n_i-1)*pitch/2

        因此:
            d_i >= R_tube + clearance + t/2
                   + (n_i-1)*pitch/2
        """
        c = self.cfg

        if c.stack_model == "paper":
            return c.d_min_m

        pitch = c.magnet_thickness_m + c.magnet_gap_m
        half_center_span = 0.5 * max(n_i - 1, 0) * pitch
        nearest_center_min = (
            0.5 * c.tube_outer_diameter_m
            + c.tube_clearance_m
            + 0.5 * c.magnet_thickness_m
        )

        return max(
            c.d_min_m,
            nearest_center_min + half_center_span,
        )

    def _one_stack_pair_field(self, i, n_i, d_i):
        return self._one_stack_pair_field_at_z(
            self.z_m, i, n_i, d_i
        )

    def _one_stack_pair_field_at_z(self, z_m, i, n_i, d_i):
        """计算任意 longitudinal 网格上的单个固定 stack pair 磁场。"""
        z_m = np.asarray(z_m, dtype=float)
        if n_i == 0:
            return np.zeros_like(z_m)

        sign = self.orientation[i]
        x_i = self.x_stack_m[i]

        if self.cfg.stack_model == "paper":
            # 单侧 n_i 块磁铁等效为总磁矩 n_i*m，位置 ±d_i
            return self._dipole_pair_field(
                z_m,
                x_i=x_i,
                d=d_i,
                moment=sign * n_i * self.single_magnet_moment_Am2,
            )

        # resolved:
        # 单侧 n_i 块圆片沿 z 堆叠，整个 stack 中心为 +d_i；
        # 下侧是完全镜像的 stack。
        pitch = self.cfg.magnet_thickness_m + self.cfg.magnet_gap_m
        offsets = (
            np.arange(n_i, dtype=float) - 0.5 * (n_i - 1)
        ) * pitch

        radial_distances = d_i + offsets

        nearest_center_min = (
            0.5 * self.cfg.tube_outer_diameter_m
            + self.cfg.tube_clearance_m
            + 0.5 * self.cfg.magnet_thickness_m
        )
        if np.min(radial_distances) < nearest_center_min - 1e-15:
            return None

        # 向量化计算所有圆片磁铁的贡献，避免 Python 循环
        dx = (z_m - x_i)[None, :]
        q = radial_distances[:, None]
        r2 = dx * dx + q * q

        B_each = (
            MU0
            * (sign * self.single_magnet_moment_Am2)
            / (2 * np.pi)
            * (2 * q * q - dx * dx)
            / (r2 ** 2.5)
        )
        return np.sum(B_each, axis=0)

    def simulated_field_at_z(self, result, z_m):
        """不重新优化，计算当前磁铁阵列在任意 x 范围内的磁场。"""
        z_m = np.asarray(z_m, dtype=float)
        fields = [
            self._one_stack_pair_field_at_z(
                z_m,
                i,
                int(result["n"][i]),
                float(result["d_m"][i]),
            )
            for i in range(self.cfg.n_stacks)
        ]
        return np.sum(np.asarray(fields), axis=0)

    def counts_from_a(self, a):
        """
        对应 thesis Eq. (7.24):

            n_i = round[B_desired(x_i)/(B_single(x_i)*a)]

        对负场取绝对值计算块数，方向由 orientation 单独处理。
        """
        n_float = (
            np.abs(self.B_target_at_stack_T)
            / (a * self.B_single_at_d0_T)
        )
        n = np.rint(n_float).astype(int)
        # 不再设置固定磁铁数量上限。
        # 可行的 n_i 只由真空管避让条件和 d_max 决定。
        return np.maximum(n, 0)

    def _loss(self, B_sim_T):
        # 以有效区全长为主体，首尾对称加权，扩展区作为弱边界约束；
        # 另加有效区最大误差项，避免端点被 MSE 忽略。
        squared_error = (self.B_target_T - B_sim_T) ** 2
        active_mask = (self.z_m >= 0.0) & (self.z_m <= self.cfg.L_m)
        start_mask = active_mask & (
            self.z_m <= self.cfg.start_fit_region_m
        )
        end_mask = active_mask & (
            self.z_m >= self.cfg.L_m - self.cfg.end_fit_region_m
        )

        weights = np.zeros_like(self.z_m, dtype=float)
        left_extension = self.z_m < 0.0
        right_extension = self.z_m > self.cfg.L_m
        weights[left_extension] = (
            self.cfg.extension_fit_weight
            * np.exp(
                -(
                    self.z_m[left_extension]
                    / self.cfg.extension_fit_decay_m
                ) ** 2
            )
        )
        weights[right_extension] = (
            self.cfg.extension_fit_weight
            * np.exp(
                -(
                    (self.z_m[right_extension] - self.cfg.L_m)
                    / self.cfg.extension_fit_decay_m
                ) ** 2
            )
        )
        weights[active_mask] = 1.0
        weights[start_mask] = (
            self.cfg.start_fit_weight
        )
        weights[end_mask] = self.cfg.end_fit_weight

        loss = float(np.average(squared_error, weights=weights))
        if self.cfg.max_abs_error_weight > 0:
            loss += float(
                self.cfg.max_abs_error_weight
                * np.max(squared_error[active_mask])
            )
        return loss

    def _optimize_d_for_fixed_counts(self, n):
        """
        固定 {n_i}，逐个尝试 d_i ± delta_d。
        只有 loss 下降才接受，与论文描述一致。
        """
        c = self.cfg

        if (
            c.max_total_magnets is not None
            and 2 * int(np.sum(n)) > c.max_total_magnets
        ):
            return None

        d_lower = np.array(
            [self._effective_lower_d_bound(int(ni)) for ni in n],
            dtype=float,
        )

        if np.any(d_lower > c.d_max_m):
            return None

        d = np.maximum(c.d0_m, d_lower)
        d = np.minimum(d, c.d_max_m)

        contributions = []
        for i in range(c.n_stacks):
            Bi = self._one_stack_pair_field(i, int(n[i]), float(d[i]))
            if Bi is None:
                return None
            contributions.append(Bi)

        contributions = np.asarray(contributions)
        B_total = contributions.sum(axis=0)
        loss = self._loss(B_total)

        for step in c.d_steps_m:
            for _ in range(c.max_sweeps_each_step):
                moved_in_this_sweep = False

                for i in range(c.n_stacks):
                    if n[i] == 0:
                        continue

                    old_Bi = contributions[i]
                    best_loss = loss
                    best_d = d[i]
                    best_Bi = old_Bi

                    for d_try in (d[i] - step, d[i] + step):
                        if d_try < d_lower[i] - 1e-15:
                            continue
                        if d_try > c.d_max_m + 1e-15:
                            continue

                        new_Bi = self._one_stack_pair_field(
                            i, int(n[i]), float(d_try)
                        )
                        if new_Bi is None:
                            continue

                        B_try = B_total - old_Bi + new_Bi
                        loss_try = self._loss(B_try)

                        if loss_try < best_loss - 1e-20:
                            best_loss = loss_try
                            best_d = d_try
                            best_Bi = new_Bi

                    if best_d != d[i]:
                        B_total = B_total - old_Bi + best_Bi
                        contributions[i] = best_Bi
                        d[i] = best_d
                        loss = best_loss
                        moved_in_this_sweep = True

                if not moved_in_this_sweep:
                    break

        return {
            "loss": loss,
            "n": np.array(n, dtype=int),
            "d_m": d.copy(),
            "B_sim_T": B_total.copy(),
        }

    def _refine_counts_and_distances(self, initial):
        """
        从 a 扫描得到的整数 {n_i} 出发，逐 stack 尝试改变磁铁数量。

        每次 n_i 改变后都重新执行完整的 d_i coordinate descent；只有
        总场误差下降才接受。这样 x_i 保持严格固定，同时 n_i 和 d_i
        能交替收敛，而不是把所有 n_i 限制在同一个缩放参数 a 上。
        """
        current = {
            "loss": float(initial["loss"]),
            "n": initial["n"].copy(),
            "d_m": initial["d_m"].copy(),
            "B_sim_T": initial["B_sim_T"].copy(),
        }
        cache = {tuple(current["n"].tolist()): current}

        def solve_counts(n_try):
            key = tuple(np.asarray(n_try, dtype=int).tolist())
            if key not in cache:
                cache[key] = self._optimize_d_for_fixed_counts(
                    np.asarray(n_try, dtype=int)
                )
            return cache[key]

        for count_step in self.cfg.count_refinement_steps:
            count_step = int(count_step)

            for _ in range(self.cfg.max_count_refinement_sweeps):
                moved_in_this_sweep = False

                for i in range(self.cfg.n_stacks):
                    best_local = current

                    for delta_n in (-count_step, count_step):
                        n_try = current["n"].copy()
                        n_try[i] += delta_n
                        if n_try[i] < 0:
                            continue

                        candidate = solve_counts(n_try)
                        if candidate is None:
                            continue

                        if candidate["loss"] < best_local["loss"] - 1e-20:
                            best_local = candidate

                    if best_local["loss"] < current["loss"] - 1e-20:
                        current = {
                            "loss": float(best_local["loss"]),
                            "n": best_local["n"].copy(),
                            "d_m": best_local["d_m"].copy(),
                            "B_sim_T": best_local["B_sim_T"].copy(),
                        }
                        moved_in_this_sweep = True

                if not moved_in_this_sweep:
                    break

        return current

    def _polish_distances_continuously(self, initial):
        """固定整数 n_i，同时连续联合优化全部 d_i。"""
        if minimize is None:
            return initial

        n = np.asarray(initial["n"], dtype=int)
        d_lower = np.array(
            [self._effective_lower_d_bound(int(ni)) for ni in n],
            dtype=float,
        )
        bounds = list(zip(d_lower, np.full_like(d_lower, self.cfg.d_max_m)))

        best_field = initial["B_sim_T"].copy()
        best_loss = float(initial["loss"])
        best_d = np.asarray(initial["d_m"], dtype=float).copy()

        def objective(d):
            nonlocal best_field, best_loss, best_d
            fields = [
                self._one_stack_pair_field(i, int(n[i]), float(d[i]))
                for i in range(self.cfg.n_stacks)
            ]
            if any(field is None for field in fields):
                return 1e30
            B_total = np.sum(np.asarray(fields), axis=0)
            loss = self._loss(B_total)
            if loss < best_loss:
                best_loss = float(loss)
                best_field = B_total.copy()
                best_d = np.asarray(d, dtype=float).copy()
            return loss

        polished = minimize(
            objective,
            np.asarray(initial["d_m"], dtype=float),
            method="Powell",
            bounds=bounds,
            options={
                "xtol": 1e-7,
                "ftol": 1e-12,
                "maxiter": self.cfg.continuous_polish_maxiter,
                "disp": False,
            },
        )

        # Powell 的最终点不一定是整个搜索过程中最好的点；objective
        # 已经持续记录 best_field，因此这里再显式比较最终结果。
        final_d = np.asarray(polished.x, dtype=float)
        final_fields = [
            self._one_stack_pair_field(i, int(n[i]), float(final_d[i]))
            for i in range(self.cfg.n_stacks)
        ]
        final_field = np.sum(np.asarray(final_fields), axis=0)
        final_loss = self._loss(final_field)
        if final_loss < best_loss:
            best_loss = float(final_loss)
            best_field = final_field.copy()
            best_d = final_d

        return {
            "loss": best_loss,
            "n": n.copy(),
            "d_m": best_d.copy(),
            "B_sim_T": best_field.copy(),
        }

    def optimize(self, show_progress=True):
        """
        外层扫描 a；对于每个由 a 生成的整数 {n_i}，
        做一次论文式 d_i coordinate descent。
        """
        c = self.cfg

        a_values = np.arange(
            c.a_min,
            c.a_max + 0.5 * c.a_step,
            c.a_step,
        )

        best = None

        # 不同 a 有时会 round 成同一个 {n_i}，避免重复计算
        cache = {}

        iterator = tqdm(
            a_values,
            desc="Scanning a",
            disable=not show_progress,
        )

        for a in iterator:
            n = self.counts_from_a(float(a))
            key = tuple(n.tolist())

            if key in cache:
                candidate = cache[key]
            else:
                candidate = self._optimize_d_for_fixed_counts(n)
                cache[key] = candidate

            if candidate is None:
                continue

            if best is None or candidate["loss"] < best["loss"]:
                best = {
                    **candidate,
                    "a": float(a),
                }

        if best is None:
            raise RuntimeError(
                "没有找到可行解。请检查 d 范围和真空管机械约束。"
            )

        a_seed = best["a"]
        loss_before_count_refinement = best["loss"]

        if c.refine_integer_counts:
            refined = self._refine_counts_and_distances(best)
            best = {
                **refined,
                "a": a_seed,
                "counts_refined": (
                    refined["loss"]
                    < loss_before_count_refinement - 1e-20
                ),
            }
        else:
            best["counts_refined"] = False

        if c.continuous_distance_polish:
            counts_refined = best["counts_refined"]
            polished = self._polish_distances_continuously(best)
            best = {
                **polished,
                "a": a_seed,
                "counts_refined": counts_refined,
            }

        residual = best["B_sim_T"] - self.B_target_T
        active_mask = (self.z_m >= 0.0) & (self.z_m <= c.L_m)
        start_mask = active_mask & (
            self.z_m <= min(30e-3, c.L_m)
        )
        end_mask = active_mask & (
            self.z_m >= max(0.0, c.L_m - 30e-3)
        )
        middle_mask = active_mask & ~start_mask & ~end_mask
        active_residual = residual[active_mask]

        best["fit_domain_rms_error_T"] = float(
            np.sqrt(np.mean(residual ** 2))
        )
        best["rms_error_T"] = float(
            np.sqrt(np.mean(active_residual ** 2))
        )
        best["mae_T"] = float(np.mean(np.abs(active_residual)))
        best["max_abs_error_T"] = float(
            np.max(np.abs(active_residual))
        )
        best["start_30mm_rms_error_T"] = float(
            np.sqrt(np.mean(residual[start_mask] ** 2))
        )
        best["start_30mm_max_abs_error_T"] = float(
            np.max(np.abs(residual[start_mask]))
        )
        best["middle_rms_error_T"] = float(
            np.sqrt(np.mean(residual[middle_mask] ** 2))
        )
        best["end_30mm_rms_error_T"] = float(
            np.sqrt(np.mean(residual[end_mask] ** 2))
        )
        best["end_30mm_max_abs_error_T"] = float(
            np.max(np.abs(residual[end_mask]))
        )
        best["total_magnets"] = int(2 * np.sum(best["n"]))

        return best

    def result_dataframe(self, result):
        pitch = (
            self.cfg.magnet_thickness_m
            + self.cfg.magnet_gap_m
        )

        if self.cfg.stack_model == "resolved":
            nearest_center = (
                result["d_m"]
                - 0.5 * (result["n"] - 1) * pitch
            )
        else:
            nearest_center = result["d_m"].copy()

        return pd.DataFrame(
            {
                "stack_index": np.arange(1, self.cfg.n_stacks + 1),
                "x_mm": self.x_stack_m * 1e3,
                "n_magnets_each_side": result["n"],
                "total_magnets_pair": 2 * result["n"],
                "orientation": self.orientation.astype(int),
                "d_stack_center_mm": result["d_m"] * 1e3,
                "nearest_magnet_center_mm": nearest_center * 1e3,
            }
        )

    def print_summary(self, result):
        print("\n========== Zeeman slower optimization ==========")
        print(f"stack model          : {self.cfg.stack_model}")
        print(f"target mode          : {self.cfg.target_mode}")
        print(f"L                    : {self.cfg.L_m*1e3:.3f} mm")
        print(f"number of stack pairs: {self.cfg.n_stacks}")
        print(
            "total-magnet limit   : "
            f"{self.cfg.max_total_magnets}"
        )
        print(
            "fit-domain z range    : "
            f"{self.z_m.min()*1e3:.3f} ... "
            f"{self.z_m.max()*1e3:.3f} mm"
        )
        print(
            "boundary weights      : "
            f"start={self.cfg.start_fit_weight:.3g}, "
            f"end={self.cfg.end_fit_weight:.3g}, "
            f"extension={self.cfg.extension_fit_weight:.3g}, "
            f"decay={self.cfg.extension_fit_decay_m*1e3:.1f}mm, "
            f"max-error={self.cfg.max_abs_error_weight:.3g}"
        )
        print(
            "stack center x range : "
            f"{self.x_stack_m.min()*1e3:.3f} ... "
            f"{self.x_stack_m.max()*1e3:.3f} mm"
        )
        pitch = self.cfg.magnet_thickness_m + self.cfg.magnet_gap_m
        nearest_center_min = (
            0.5 * self.cfg.tube_outer_diameter_m
            + self.cfg.tube_clearance_m
            + 0.5 * self.cfg.magnet_thickness_m
        )
        if pitch > 0:
            n_geom_max = int(np.floor(
                1.0 + 2.0 * (self.cfg.d_max_m - nearest_center_min) / pitch
            ))
            n_geom_max = max(n_geom_max, 0)
            print(f"geometry-limited n_max : {n_geom_max} magnets/side")
        print(f"single magnet moment : {self.single_magnet_moment_Am2:.6f} A m^2")
        print(f"B_single(d0=50 mm)   : {self.B_single_at_d0_T*1e3:.6f} mT")
        print(f"best a seed          : {result['a']:.6g}")
        print(f"integer counts refined: {result['counts_refined']}")
        print(f"active RMS error     : {result['rms_error_T']*1e3:.6f} mT")
        print(f"MAE                  : {result['mae_T']*1e3:.6f} mT")
        print(f"max |error|          : {result['max_abs_error_T']*1e3:.6f} mT")
        print(
            "start 0-30 mm RMS    : "
            f"{result['start_30mm_rms_error_T']*1e3:.6f} mT"
        )
        print(
            "start 0-30 mm max    : "
            f"{result['start_30mm_max_abs_error_T']*1e3:.6f} mT"
        )
        print(
            "middle RMS            : "
            f"{result['middle_rms_error_T']*1e3:.6f} mT"
        )
        print(
            "end 155-185 mm RMS    : "
            f"{result['end_30mm_rms_error_T']*1e3:.6f} mT"
        )
        print(
            "end 155-185 mm max    : "
            f"{result['end_30mm_max_abs_error_T']*1e3:.6f} mT"
        )
        print(f"total magnets        : {result['total_magnets']}")
        print("===============================================\n")

        table = self.result_dataframe(result)
        with pd.option_context(
            "display.max_rows", None,
            "display.width", 140,
            "display.max_columns", None,
        ):
            print(table.to_string(index=False))

    def save_results(
        self,
        result,
        output_dir=".",
        prefix="Yb171_zeeman_15stacks",
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_df = self.result_dataframe(result)
        config_path = output_dir / f"{prefix}_magnet_config.csv"
        config_df.to_csv(config_path, index=False)

        field_df = pd.DataFrame(
            {
                "z_mm": self.z_m * 1e3,
                "B_target_mT": self.B_target_T * 1e3,
                "B_sim_mT": result["B_sim_T"] * 1e3,
                "residual_mT": (
                    result["B_sim_T"] - self.B_target_T
                ) * 1e3,
                "in_active_0_185mm": (
                    (self.z_m >= 0.0)
                    & (self.z_m <= self.cfg.L_m)
                ),
            }
        )
        field_path = output_dir / f"{prefix}_field.csv"
        field_df.to_csv(field_path, index=False)

        # 目标曲线及原拟合结果保持不变；另行输出同一磁铁阵列从
        # -20 mm 到 400 mm 的磁场，便于观察减速器之后的衰减尾场。
        z_extended_sim_m = np.linspace(-20e-3, 400e-3, 1681)
        extended_field_df = pd.DataFrame(
            {
                "z_mm": z_extended_sim_m * 1e3,
                "B_sim_mT": (
                    self.simulated_field_at_z(result, z_extended_sim_m)
                    * 1e3
                ),
            }
        )
        extended_field_path = (
            output_dir / f"{prefix}_field_to_400mm.csv"
        )
        extended_field_df.to_csv(extended_field_path, index=False)

        return config_path, field_path, extended_field_path

    def plot_results(
        self,
        result,
        output_dir=".",
        prefix="Yb171_zeeman_15stacks",
        show=True,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        z_extended_sim_m = np.linspace(-20e-3, 400e-3, 1681)
        B_extended_sim_T = self.simulated_field_at_z(
            result, z_extended_sim_m
        )

        # Figure 1: target + simulated field
        fig = plt.figure(figsize=(8, 5))
        plt.plot(
            self.z_m * 1e3,
            self.B_target_T * 1e3,
            label="B desired",
        )
        plt.plot(
            z_extended_sim_m * 1e3,
            B_extended_sim_T * 1e3,
            label="B simulated",
        )
        plt.axvline(0.0, color="0.5", linestyle="--", linewidth=1)
        plt.axvline(
            self.cfg.L_m * 1e3,
            color="0.5",
            linestyle="--",
            linewidth=1,
        )
        plt.xlabel("Longitudinal position (mm)")
        plt.ylabel("Transverse magnetic field (mT)")
        plt.xlim(-20.0, 400.0)
        plt.legend()
        plt.tight_layout()

        field_fig_path = output_dir / f"{prefix}_field.png"
        plt.savefig(field_fig_path, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)

        # Figure 2: residual
        fig = plt.figure(figsize=(8, 4))
        plt.plot(
            self.z_m * 1e3,
            (result["B_sim_T"] - self.B_target_T) * 1e3,
        )
        plt.axhline(0.0, linewidth=1)
        plt.axvline(0.0, color="0.5", linestyle="--", linewidth=1)
        plt.axvline(
            self.cfg.L_m * 1e3,
            color="0.5",
            linestyle="--",
            linewidth=1,
        )
        plt.xlabel("Longitudinal position (mm)")
        plt.ylabel("B simulated - B desired (mT)")
        plt.tight_layout()

        residual_fig_path = output_dir / f"{prefix}_residual.png"
        plt.savefig(residual_fig_path, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)

        # Figure 3: entrance-region detail, where boundary effects matter most
        start_limit_m = min(30e-3, self.cfg.L_m)
        start_mask = (self.z_m >= 0.0) & (self.z_m <= start_limit_m)
        fig = plt.figure(figsize=(8, 4))
        plt.plot(
            self.z_m[start_mask] * 1e3,
            self.B_target_T[start_mask] * 1e3,
            label="B desired",
        )
        plt.plot(
            self.z_m[start_mask] * 1e3,
            result["B_sim_T"][start_mask] * 1e3,
            label="B simulated",
        )
        plt.xlabel("Longitudinal position (mm)")
        plt.ylabel("Transverse magnetic field (mT)")
        plt.legend()
        plt.tight_layout()

        start_fig_path = output_dir / f"{prefix}_field_start_30mm.png"
        plt.savefig(start_fig_path, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)

        # Figure 4: exit-region detail
        end_start_m = max(0.0, self.cfg.L_m - 30e-3)
        end_mask = (self.z_m >= end_start_m) & (
            self.z_m <= self.cfg.L_m
        )
        fig = plt.figure(figsize=(8, 4))
        plt.plot(
            self.z_m[end_mask] * 1e3,
            self.B_target_T[end_mask] * 1e3,
            label="B desired",
        )
        plt.plot(
            self.z_m[end_mask] * 1e3,
            result["B_sim_T"][end_mask] * 1e3,
            label="B simulated",
        )
        plt.xlabel("Longitudinal position (mm)")
        plt.ylabel("Transverse magnetic field (mT)")
        plt.legend()
        plt.tight_layout()

        end_fig_path = output_dir / f"{prefix}_field_end_30mm.png"
        plt.savefig(end_fig_path, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return (
            field_fig_path,
            residual_fig_path,
            start_fig_path,
            end_fig_path,
        )


def main():
    here = Path(__file__).resolve().parent
    project_root = here.parent
    csv_path = project_root / "inputs" / "Yb171_target_B_extended.csv"
    output_dir = project_root / "runs" / "01_15stacks_unrestricted"

    cfg = ZeemanSlowerConfig(
        csv_path=str(csv_path),

        # 你的装置
        L_m=0.185,
        n_stacks=15,
        max_total_magnets=None,

        # 图纸给定的 15 个 longitudinal stack positions：严格固定，
        # 优化器只改变每组磁铁数 n_i 和横向离轴距离 d_i。
        x_stack_positions_m=tuple(
            np.array(
                [
                    9.467, 22.400, 35.333, 48.267, 61.200,
                    74.133, 87.067, 100.000, 112.933, 125.867,
                    138.800, 151.733, 164.667, 177.600, 190.533,
                ]
            ) * 1e-3
        ),
        stack_edge_extension_m=0.0,

        # 扩展数据只作弱边界约束；原 0--185 mm 是主拟合区，首尾
        # 各 30 mm 完全对称加权，并压制有效区最大单点误差。
        fit_left_extension_m=13e-3,
        fit_right_extension_m=13e-3,
        extension_fit_weight=0.02,
        # 1 m 衰减长度在当前 13 mm 外推范围内近似均匀；实验对比表明
        # 这能避免落入磁铁总数过少、有效区 RMS 更高的整数局部解。
        extension_fit_decay_m=1.0,
        start_fit_region_m=30e-3,
        start_fit_weight=20.0,
        end_fit_region_m=30e-3,
        end_fit_weight=20.0,
        max_abs_error_weight=0.05,
        n_sample=641,

        # 你的 N35 圆片磁铁
        Br_T=1.2,
        magnet_diameter_m=8e-3,
        magnet_thickness_m=3e-3,

        # 论文给的初始值
        d0_m=50e-3,

        # 目前按参考 CAD 暂定；有真实机械尺寸后请修改
        d_min_m=20e-3,
        d_max_m=100e-3,

        # crop 直接使用扩展 CSV 的 -13--198 mm 拟合曲线。
        target_mode="crop",

        # 推荐 resolved；若要更严格照 thesis 的等效 stack dipole，
        # 改成 "paper"
        stack_model="resolved",

        # 你的真空管 OD = 16 mm；暂时取磁铁与管壁 clearance = 0
        tube_outer_diameter_m=16e-3,
        tube_clearance_m=0.0,

        # a 自动扫描
        a_min=0.5,
        a_max=10.0,
        a_step=0.05,

        # d_i coordinate descent
        d_steps_m=(2e-3, 1e-3, 0.5e-3, 0.25e-3, 0.1e-3),
    )

    optimizer = PermanentMagnetZeemanSlower(cfg)

    print("Target CSV range:")
    print(
        f"z = {optimizer.z_raw_m.min()*1e3:.3f}"
        f" ... {optimizer.z_raw_m.max()*1e3:.3f} mm"
    )
    print(
        f"B = {optimizer.B_raw_T.min()*1e3:.3f}"
        f" ... {optimizer.B_raw_T.max()*1e3:.3f} mT"
    )

    result = optimizer.optimize(show_progress=True)
    optimizer.print_summary(result)

    config_path, field_path, extended_field_path = optimizer.save_results(
        result,
        output_dir=output_dir,
        prefix="Yb171_zeeman_15stacks",
    )
    field_fig, residual_fig, start_fig, end_fig = optimizer.plot_results(
        result,
        output_dir=output_dir,
        prefix="Yb171_zeeman_15stacks",
        show=True,
    )

    print("\nSaved:")
    print(config_path)
    print(field_path)
    print(extended_field_path)
    print(field_fig)
    print(residual_fig)
    print(start_fig)
    print(end_fig)


if __name__ == "__main__":
    main()
