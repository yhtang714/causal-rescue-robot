"""Dataset and metric helpers for training."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class DataValidationResult:
    total_rows: int = 0
    total_groups: int = 0
    valid_groups: int = 0
    skipped_single_candidate: int = 0
    skipped_no_target: int = 0
    skipped_all_nan: int = 0

    candidate_counts: List[int] = field(default_factory=list)

    groups_with_ties: int = 0
    total_positives: int = 0

    delta_e_ranges: List[float] = field(default_factory=list)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("数据验证摘要")
        print("=" * 60)

        print(f"\n[基本统计]")
        print(f"  总行数: {self.total_rows:,}")
        print(f"  总分组数: {self.total_groups:,}")
        print(f"  有效分组数: {self.valid_groups:,} ({100*self.valid_groups/max(self.total_groups,1):.1f}%)")

        print(f"\n[跳过的分组]")
        print(f"  单候选 (<2): {self.skipped_single_candidate:,} ({100*self.skipped_single_candidate/max(self.total_groups,1):.1f}%)")
        print(f"  无正类: {self.skipped_no_target:,} ({100*self.skipped_no_target/max(self.total_groups,1):.1f}%)")
        print(f"  全NaN: {self.skipped_all_nan:,} ({100*self.skipped_all_nan/max(self.total_groups,1):.1f}%)")

        if self.candidate_counts:
            counts = np.array(self.candidate_counts)
            print(f"\n[候选数分布]")
            print(f"  min: {counts.min()}")
            print(f"  median: {np.median(counts):.1f}")
            print(f"  mean: {counts.mean():.1f}")
            print(f"  p90: {np.percentile(counts, 90):.1f}")
            print(f"  max: {counts.max()}")

        if self.total_groups > 0:
            print(f"\n[Ties 统计]")
            print(f"  有并列最大的组: {self.groups_with_ties:,} ({100*self.groups_with_ties/max(self.valid_groups,1):.1f}%)")

        if self.delta_e_ranges:
            ranges = np.array(self.delta_e_ranges)
            print(f"\n[delta_E 组内范围 (max-min)]")
            print(f"  min: {ranges.min():.4f}")
            print(f"  median: {np.median(ranges):.4f}")
            print(f"  mean: {ranges.mean():.4f}")
            print(f"  max: {ranges.max():.4f}")

            small_range = (ranges < 1e-6).sum()
            if small_range > 0:
                print(f"  ⚠️ 范围<1e-6的组: {small_range} ({100*small_range/len(ranges):.1f}%)")

        print("=" * 60)


def validate_candidate_data(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str = "delta_E",
    candidate_id_col: Optional[str] = None,
    verbose: bool = True,
) -> DataValidationResult:
    result = DataValidationResult()
    result.total_rows = len(df)

    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"分组列 '{col}' 不存在于数据中！")

    if candidate_id_col is None:
        for col in ["candidate_id", "cube_id", "A_cube_id"]:
            if col in df.columns:
                candidate_id_col = col
                break

    if candidate_id_col is None:
        warnings.warn(
            "未找到候选ID列 (candidate_id/cube_id/A_cube_id)。"
            "数据可能只记录了执行的动作，没有候选集合。"
        )

    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于数据中！")

    grouped = df.groupby(group_cols, sort=False)
    result.total_groups = len(grouped)

    for group_key, group_df in grouped:
        n_candidates = len(group_df)

        if n_candidates < 2:
            result.skipped_single_candidate += 1
            continue

        target_values = group_df[target_col].values

        if np.all(np.isnan(target_values)):
            result.skipped_all_nan += 1
            continue

        result.valid_groups += 1
        result.candidate_counts.append(n_candidates)

        valid_targets = target_values[~np.isnan(target_values)]
        if len(valid_targets) > 1:
            range_val = valid_targets.max() - valid_targets.min()
            result.delta_e_ranges.append(range_val)

        max_val = valid_targets.max()
        n_max = (valid_targets == max_val).sum()
        if n_max > 1:
            result.groups_with_ties += 1
        result.total_positives += n_max

    if result.valid_groups == 0:
        if result.skipped_single_candidate == result.total_groups:
            raise ValueError(
                f"数据不满足候选学习要求！\n"
                f"所有 {result.total_groups} 个组都只有 1 个候选。\n"
                f"这通常意味着数据只记录了执行的动作，没有候选集合。\n"
                f"请确保数据包含每一步的所有可选候选。"
            )
        else:
            raise ValueError(
                f"没有有效的训练组！\n"
                f"跳过了 {result.skipped_single_candidate} 个单候选组，"
                f"{result.skipped_all_nan} 个全NaN组。"
            )

    if verbose:
        result.print_summary()

    return result


def check_candidate_dataset(csv_path: str, group_cols: Optional[List[str]] = None) -> DataValidationResult:
    if group_cols is None:
        group_cols = ["episode_id", "scene_id", "step_index"]

    df = pd.read_csv(csv_path)
    return validate_candidate_data(df, group_cols)


def generate_optimal_labels(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str = "delta_E",
    tie_break: str = "first",
    eps: float = 1e-6,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if verbose:
        print("\n" + "=" * 60)
        print(f"生成物理最优标签 (is_optimal) - tie_break={tie_break}")
        print("=" * 60)

    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在！")

    df["is_optimal"] = 0
    df["is_optimal_any"] = 0

    label_info = {
        "tie_break": tie_break,
        "eps": eps,
        "n_groups": 0,
        "n_positives_first": 0,
        "n_positives_any": 0,
        "n_ties": 0,
    }

    is_optimal_list = []
    is_optimal_any_list = []

    for group_key, group_indices in df.groupby(group_cols, sort=False).groups.items():
        group_df = df.loc[group_indices]
        target_values = group_df[target_col].values

        is_optimal = np.zeros(len(group_df), dtype=int)
        is_optimal_any = np.zeros(len(group_df), dtype=int)

        valid_mask = ~np.isnan(target_values)
        if not valid_mask.any():
            is_optimal_list.extend(is_optimal.tolist())
            is_optimal_any_list.extend(is_optimal_any.tolist())
            continue

        label_info["n_groups"] += 1

        valid_values = target_values[valid_mask]
        max_val = valid_values.max()

        is_max_any = valid_mask & (target_values >= max_val - eps)
        is_optimal_any = is_max_any.astype(int)
        label_info["n_positives_any"] += is_max_any.sum()

        n_max = is_max_any.sum()
        if n_max > 1:
            label_info["n_ties"] += 1

        if tie_break == "first":
            is_optimal = np.zeros_like(target_values, dtype=int)
            first_max_idx = np.where(is_max_any)[0][0]
            is_optimal[first_max_idx] = 1
            label_info["n_positives_first"] += 1
        else:
            is_optimal = is_max_any.astype(int)
            label_info["n_positives_first"] += is_max_any.sum()

        is_optimal_list.extend(is_optimal.tolist())
        is_optimal_any_list.extend(is_optimal_any.tolist())

    df["is_optimal"] = is_optimal_list
    df["is_optimal_any"] = is_optimal_any_list

    if verbose:
        print(f"\n[标签统计]")
        print(f"  有效组数: {label_info['n_groups']}")
        print(f"  is_optimal=1 数量: {(df['is_optimal'] == 1).sum()}")
        if tie_break == "first":
            print(f"  is_optimal_any=1 数量: {label_info['n_positives_any']}")
        print(f"  有并列最大的组: {label_info['n_ties']} ({100*label_info['n_ties']/max(label_info['n_groups'],1):.1f}%)")

        optimal_values = df[df["is_optimal"] == 1][target_col]
        all_values = df[target_col].dropna()
        print(f"\n[{target_col} 统计]")
        print(f"  所有候选: mean={all_values.mean():.4f}, std={all_values.std():.4f}")
        print(f"  最优候选: mean={optimal_values.mean():.4f}, std={optimal_values.std():.4f}")

    return df, label_info


@dataclass
class FeatureScaler:
    feature_cols: List[str]
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for i, col in enumerate(self.feature_cols):
            if col in df.columns:
                df[col] = (df[col] - self.mean[i]) / (self.std[i] + self.eps)
        return df

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_cols": self.feature_cols,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureScaler":
        return cls(
            feature_cols=d["feature_cols"],
            mean=np.array(d["mean"]),
            std=np.array(d["std"]),
            eps=d.get("eps", 1e-8),
        )


def compute_feature_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    verbose: bool = True,
) -> FeatureScaler:
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"以下特征列不存在于数据中: {missing_cols}")

    values = df[feature_cols].values.astype(np.float32)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)

    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)

    if verbose:
        zero_var_cols = [feature_cols[i] for i in range(len(feature_cols)) if std[i] < 1e-8]
        if zero_var_cols:
            print(f"\n[警告] 以下特征列方差为0或近似0:")
            for col in zero_var_cols[:10]:
                print(f"  - {col}")
            if len(zero_var_cols) > 10:
                print(f"  ... 及其他 {len(zero_var_cols)-10} 列")

    return FeatureScaler(
        feature_cols=feature_cols,
        mean=mean,
        std=std,
    )


def validate_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    verbose: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    info = {
        "total_features": len(feature_cols),
        "missing_cols": [],
        "zero_var_cols": [],
        "all_nan_cols": [],
    }

    valid_cols = []

    for col in feature_cols:
        if col not in df.columns:
            info["missing_cols"].append(col)
            continue

        values = df[col].values

        if np.all(np.isnan(values)):
            info["all_nan_cols"].append(col)
            continue

        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0 and np.std(valid_values) < 1e-10:
            info["zero_var_cols"].append(col)

        valid_cols.append(col)

    if verbose:
        print(f"\n[特征验证]")
        print(f"  总特征数: {info['total_features']}")
        print(f"  有效特征数: {len(valid_cols)}")

        if info["missing_cols"]:
            print(f"  ⚠️ 缺失列 ({len(info['missing_cols'])}): {info['missing_cols'][:5]}...")

        if info["all_nan_cols"]:
            print(f"  ⚠️ 全NaN列 ({len(info['all_nan_cols'])}): {info['all_nan_cols'][:5]}...")

        if info["zero_var_cols"]:
            print(f"  ⚠️ 零方差列 ({len(info['zero_var_cols'])}): {info['zero_var_cols'][:5]}...")

    if info["missing_cols"]:
        raise ValueError(
            f"特征列缺失（禁止静默补0）！\n"
            f"缺失列: {info['missing_cols']}"
        )

    return valid_cols, info


def truncate_candidates(
    features: np.ndarray,
    targets: np.ndarray,
    max_candidates: int,
    top_k: int = 48,
    random_k: int = 16,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(features)
    if n <= max_candidates:
        return features, targets

    if seed is not None:
        np.random.seed(seed)

    sorted_indices = np.argsort(-targets)

    actual_top_k = min(top_k, max_candidates)
    actual_random_k = max_candidates - actual_top_k

    top_indices = sorted_indices[:actual_top_k]
    remaining_indices = sorted_indices[actual_top_k:]

    if actual_random_k > 0 and len(remaining_indices) > 0:
        random_sample = np.random.choice(
            remaining_indices,
            size=min(actual_random_k, len(remaining_indices)),
            replace=False
        )
        selected_indices = np.concatenate([top_indices, random_sample])
    else:
        selected_indices = top_indices

    np.random.shuffle(selected_indices)

    optimal_idx = np.argmax(targets)
    if optimal_idx not in selected_indices:
        replace_pos = np.random.randint(0, len(selected_indices))
        selected_indices[replace_pos] = optimal_idx

    return features[selected_indices], targets[selected_indices]


def diagnose_training_issues(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str = "delta_E",
    feature_cols: Optional[List[str]] = None,
):
    print("\n" + "=" * 70)
    print("                    训练诊断报告")
    print("=" * 70)

    print(f"\n[1. 数据基本信息]")
    print(f"  总行数: {len(df):,}")
    print(f"  总列数: {len(df.columns)}")

    try:
        result = validate_candidate_data(df, group_cols, target_col, verbose=False)

        print(f"\n[2. 候选集统计]")
        print(f"  有效组数: {result.valid_groups:,}")
        print(f"  跳过的单候选组: {result.skipped_single_candidate:,}")

        if result.candidate_counts:
            counts = np.array(result.candidate_counts)
            print(f"  候选数: min={counts.min()}, median={np.median(counts):.1f}, max={counts.max()}")

        if result.candidate_counts:
            avg_candidates = np.mean(result.candidate_counts)
            random_baseline = 1.0 / avg_candidates
            print(f"  随机基线 P@1: {random_baseline:.4f} (1/{avg_candidates:.1f})")

    except Exception as e:
        print(f"\n[2. 候选集统计] 错误: {e}")

    print(f"\n[3. 目标列分析 ({target_col})]")
    if target_col in df.columns:
        values = df[target_col].dropna()
        print(f"  数量: {len(values):,}")
        print(f"  范围: [{values.min():.4f}, {values.max():.4f}]")
        print(f"  均值: {values.mean():.4f}")
        print(f"  标准差: {values.std():.4f}")

        grouped = df.groupby(group_cols)
        within_group_std = []
        for _, g in grouped:
            if len(g) >= 2:
                within_group_std.append(g[target_col].std())

        if within_group_std:
            avg_within_std = np.nanmean(within_group_std)
            print(f"  组内平均标准差: {avg_within_std:.4f}")

            if avg_within_std < 0.01:
                print(f"  ⚠️ 组内变化很小，可能难以区分候选")
    else:
        print(f"  ⚠️ 目标列不存在！")

    if feature_cols:
        print(f"\n[4. 特征分析]")
        print(f"  特征数: {len(feature_cols)}")

        nan_counts = df[feature_cols].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            print(f"  含NaN的特征: {len(nan_cols)}")

        const_cols = []
        for col in feature_cols:
            if df[col].std() < 1e-10:
                const_cols.append(col)
        if const_cols:
            print(f"  常数特征: {len(const_cols)} ({const_cols[:3]}...)")

    print(f"\n[5. 潜在问题检测]")
    issues = []

    if result.valid_groups == 0:
        issues.append("❌ 没有有效的训练组！")

    if result.skipped_single_candidate > result.valid_groups:
        issues.append("⚠️ 大量单候选组，数据可能只记录了执行动作")

    if result.candidate_counts and np.median(result.candidate_counts) < 5:
        issues.append("⚠️ 候选数偏少，可能影响学习效果")

    if result.groups_with_ties > result.valid_groups * 0.5:
        issues.append("⚠️ 超过50%的组有并列最大，考虑使用 tie_break=any")

    if not issues:
        print("  ✓ 未检测到明显问题")
    else:
        for issue in issues:
            print(f"  {issue}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="候选集数据验证工具")
    parser.add_argument("csv", type=str, help="CSV文件路径")
    parser.add_argument("--group_cols", type=str, default="episode_id,scene_id,step_index",
                        help="分组列（逗号分隔）")
    parser.add_argument("--target_col", type=str, default="delta_E",
                        help="目标列")
    parser.add_argument("--diagnose", action="store_true",
                        help="运行完整诊断")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    group_cols = args.group_cols.split(",")

    if args.diagnose:
        diagnose_training_issues(df, group_cols, args.target_col)
    else:
        check_candidate_dataset(args.csv, group_cols)
