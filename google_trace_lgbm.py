"""
Google Cluster Trace 2011 기반 LightGBM 실패 예측 및 자원 우선순위 스케줄링
- 실패 예측 확률에 가중치를 두어 우선순위 스케줄링
- 서버(머신) 15/30/45/60분 단위 종료 감지
- LightGBM 기반 이진 분류 (task 실패 여부)
- [개선] 피처 추가: task 재시도 횟수, 머신 밀집도, 시간대별 실패율 등
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 0. 경로 설정
# ───────────────────────────────────────────────
BASE_DIR   = Path(r"C:\Users\siger\Desktop\trace data")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────
# 1. 데이터 로드
# ───────────────────────────────────────────────

def load_task_events(base_dir: Path, max_files: int = 500) -> pd.DataFrame:
    cols = [
        "timestamp", "missing_info", "job_id", "task_index",
        "machine_id", "event_type", "user", "scheduling_class",
        "priority", "cpu_request", "mem_request", "disk_request",
        "different_machine_constraint"
    ]
    parts = []
    task_dir = base_dir / "task_events"

    files = sorted(task_dir.glob("part-*.csv.gz"))[:max_files]
    if not files:
        files = sorted(task_dir.glob("*.csv.gz"))[:max_files]
    if not files:
        files = sorted(task_dir.glob("*.csv"))[:max_files]

    print(f"[task_events] {len(files)}개 파일 로드 중...")
    for f in files:
        try:
            df = pd.read_csv(f, header=None, names=cols,
                             dtype=str, compression="gzip")
            parts.append(df)
        except Exception:
            try:
                df = pd.read_csv(f, header=None, names=cols, dtype=str)
                parts.append(df)
            except Exception as e:
                print(f"  skip {f.name}: {e}")

    if not parts:
        raise FileNotFoundError(
            f"{task_dir} 에 task_events 파일이 없습니다. BASE_DIR 경로를 확인하세요."
        )
    return pd.concat(parts, ignore_index=True)


def load_machine_events(base_dir: Path) -> pd.DataFrame:
    cols = ["timestamp", "machine_id", "event_type", "platform_id", "cpus", "memory"]
    parts = []
    mdir = base_dir / "machine_events"

    files = sorted(mdir.glob("part-*.csv.gz"))
    if not files:
        files = sorted(mdir.glob("*.csv.gz"))
    if not files:
        files = sorted(mdir.glob("*.csv"))

    print(f"[machine_events] {len(files)}개 파일 로드 중...")
    for f in files:
        try:
            df = pd.read_csv(f, header=None, names=cols,
                             dtype=str, compression="gzip")
            parts.append(df)
        except Exception:
            try:
                df = pd.read_csv(f, header=None, names=cols, dtype=str)
                parts.append(df)
            except Exception as e:
                print(f"  skip {f.name}: {e}")

    if not parts:
        raise FileNotFoundError(f"{mdir} 에 machine_events 파일이 없습니다.")
    return pd.concat(parts, ignore_index=True)


# ───────────────────────────────────────────────
# 2. 전처리 + 피처 엔지니어링
# ───────────────────────────────────────────────

def preprocess_task_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본 전처리 + 추가 피처 생성
    event_type: 0=SUBMIT, 1=SCHEDULE, 2=EVICT, 3=FAIL,
                4=FINISH, 5=KILL, 6=LOST, 7=UPDATE_PENDING, 8=UPDATE_RUNNING
    실패 레이블: EVICT(2), FAIL(3), LOST(6) → label=1
    """
    num_cols = [
        "timestamp", "job_id", "task_index", "machine_id",
        "event_type", "scheduling_class", "priority",
        "cpu_request", "mem_request", "disk_request",
        "different_machine_constraint"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 타임스탬프: 마이크로초 → 분
    df["time_min"] = df["timestamp"] / 1e6 / 60.0

    # 실패 레이블
    df["label"] = df["event_type"].isin([2, 3, 6]).astype(int)

    # 결측 처리
    fill_cols = ["cpu_request", "mem_request", "disk_request",
                 "scheduling_class", "priority", "different_machine_constraint"]
    for c in fill_cols:
        df[c] = df[c].fillna(df[c].median())

    df["machine_id"] = df["machine_id"].fillna(-1).astype(np.int64)
    df["job_id"]     = df["job_id"].fillna(-1).astype(np.int64)

    # ── 기존 피처 ──────────────────────────────
    # job 당 task 수
    df["job_task_count"] = df.groupby("job_id")["task_index"].transform("count")

    # 우선순위 정규화
    pmax = df["priority"].max()
    df["priority_norm"] = df["priority"] / pmax if pmax > 0 else 0.0

    # ── 추가 피처 1: task 재시도 횟수 ──────────
    # 같은 (job_id, task_index) 쌍이 반복될수록 재시도 많음
    df["task_retry_count"] = df.groupby(
        ["job_id", "task_index"]
    )["timestamp"].transform("count")

    # ── 추가 피처 2: 머신 밀집도 ───────────────
    # 머신에 몰린 task 수 (많을수록 과부하 위험)
    df["machine_task_density"] = df.groupby(
        "machine_id"
    )["task_index"].transform("count")

    # ── 추가 피처 3: 시간대별 실패율 ───────────
    # 1시간 단위 버킷 → 그 버킷의 평균 실패율
    df["time_bucket_1h"] = (df["time_min"] // 60).astype(int)
    df["time_bucket_fail_rate"] = df.groupby(
        "time_bucket_1h"
    )["label"].transform("mean")

    # ── 추가 피처 4: CPU × MEM 자원 압력 ───────
    df["cpu_mem_pressure"] = df["cpu_request"] * df["mem_request"]

    # ── 추가 피처 5: job 내 실패율 ─────────────
    df["job_fail_rate"] = df.groupby("job_id")["label"].transform("mean")

    # ── 추가 피처 6: 머신 내 실패율 ────────────
    df["machine_fail_rate"] = df.groupby("machine_id")["label"].transform("mean")

    # ── 추가 피처 7: scheduling_class별 실패율 ──
    df["sched_class_fail_rate"] = df.groupby(
        "scheduling_class"
    )["label"].transform("mean")

    return df


def preprocess_machine_events(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"]  = pd.to_numeric(df["timestamp"],  errors="coerce")
    df["machine_id"] = pd.to_numeric(df["machine_id"], errors="coerce")
    df["event_type"] = pd.to_numeric(df["event_type"], errors="coerce")
    df["time_min"]   = df["timestamp"] / 1e6 / 60.0
    return df


# ───────────────────────────────────────────────
# 3. 서버 15/30/45/60분 종료 감지
# ───────────────────────────────────────────────

def detect_machine_shutdowns(
    mdf: pd.DataFrame,
    intervals: list = [15, 30, 45, 60]
) -> pd.DataFrame:
    """
    machine_events REMOVE(event_type=1) 이벤트에서
    머신별 연속 종료 간격이 15/30/45/60분 배수인지 감지
    """
    remove = mdf[mdf["event_type"] == 1].copy()
    remove = remove.dropna(subset=["machine_id", "time_min"])
    remove = remove.sort_values(["machine_id", "time_min"])

    remove["prev_time"] = remove.groupby("machine_id")["time_min"].shift(1)
    remove["gap_min"]   = (remove["time_min"] - remove["prev_time"]).abs()
    remove = remove.dropna(subset=["gap_min"])

    results = []
    for iv in intervals:
        tol  = iv * 0.05
        mask = (remove["gap_min"] % iv) <= tol
        cnt  = mask.sum()
        results.append({
            "interval_min":        iv,
            "matching_events":     int(cnt),
            "total_remove_events": len(remove),
            "ratio_%":             round(cnt / len(remove) * 100, 2) if len(remove) > 0 else 0.0
        })

    summary = pd.DataFrame(results)
    print("\n" + "="*55)
    print("  서버 종료 주기 감지 결과 (15/30/45/60분 배수)")
    print("="*55)
    print(summary.to_string(index=False))
    print("="*55)
    return summary


# ───────────────────────────────────────────────
# 4. 피처 / 레이블 준비
# ───────────────────────────────────────────────

FEATURE_COLS = [
    # 기존 피처 (8개)
    "cpu_request", "mem_request", "disk_request",
    "scheduling_class", "priority", "priority_norm",
    "different_machine_constraint", "job_task_count",
    # 추가 피처 (7개)
    "task_retry_count",       # task 재시도 횟수
    "machine_task_density",   # 머신 밀집도
    "time_bucket_fail_rate",  # 시간대별 실패율
    "cpu_mem_pressure",       # CPU × MEM 자원 압력
    "job_fail_rate",          # job 내 실패율
    "machine_fail_rate",      # 머신 내 실패율
    "sched_class_fail_rate",  # scheduling class별 실패율
]


def build_dataset(df: pd.DataFrame):
    sub = df[FEATURE_COLS + ["label"]].dropna()
    X   = sub[FEATURE_COLS].values.astype(np.float32)
    y   = sub["label"].values.astype(int)
    return X, y, FEATURE_COLS


# ───────────────────────────────────────────────
# 5. LightGBM 학습
# ───────────────────────────────────────────────

def train_lgbm(X_tr, y_tr, X_val, y_val):
    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    params = {
        "objective":         "binary",
        "metric":            ["binary_logloss", "auc"],
        "boosting_type":     "gbdt",
        "num_leaves":        127,
        "learning_rate":     0.05,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "min_child_samples": 20,
        "scale_pos_weight":  scale_pos_weight,
        "n_jobs":            -1,
        "verbose":           -1,
        "seed":              42,
    }

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]

    print("\nLightGBM 학습 시작...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return model


# ───────────────────────────────────────────────
# 6. 우선순위 스케줄링
# ───────────────────────────────────────────────

def build_priority_schedule(
    df: pd.DataFrame,
    model: lgb.Booster,
    feature_cols: list,
    top_n: int = 100
) -> pd.DataFrame:
    """
    weighted_score = fail_prob × (1 + priority_norm)
    높은 순으로 정렬 → 서버 안정성 및 자원 예측 정확도 향상
    """
    keep_cols = ["job_id", "task_index", "machine_id",
                 "priority", "priority_norm", "label"]
    all_cols = feature_cols + [c for c in keep_cols if c not in feature_cols]
    sub = df[all_cols].dropna().copy()

    X = sub[feature_cols].values.astype(np.float32)
    sub["fail_prob"]      = model.predict(X)
    sub["weighted_score"] = sub["fail_prob"] * (1 + sub["priority_norm"])

    sched = sub.sort_values("weighted_score", ascending=False).head(top_n)
    sched = sched.reset_index(drop=True)
    sched.index = sched.index + 1
    sched.index.name = "schedule_rank"

    return sched[["job_id", "task_index", "machine_id",
                  "priority", "fail_prob", "weighted_score", "label"]]


# ───────────────────────────────────────────────
# 7. 시각화
# ───────────────────────────────────────────────

def plot_all(model, feature_cols, y_val, y_pred_prob,
             shutdown_summary, sched_df, output_dir: Path):

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Google Cluster Trace 2011 – LightGBM Failure Prediction & Resource Management",
        fontsize=13, fontweight="bold"
    )

    # (1) Feature Importance
    ax = axes[0, 0]
    imp = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feature_cols
    ).sort_values()
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance (Gain)")
    ax.set_xlabel("Gain")

    # (2) Confusion Matrix
    ax = axes[0, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fail"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix (threshold=0.5)")

    # (3) Fail Probability Distribution
    ax = axes[0, 2]
    ax.hist(y_pred_prob[y_val == 0], bins=50, alpha=0.6,
            label="Normal", color="steelblue", density=True)
    ax.hist(y_pred_prob[y_val == 1], bins=50, alpha=0.6,
            label="Fail",   color="tomato",    density=True)
    ax.set_title("Predicted Failure Probability Distribution")
    ax.set_xlabel("Fail Probability")
    ax.set_ylabel("Density")
    ax.legend()

    # (4) Server Shutdown Interval
    ax = axes[1, 0]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    bars = ax.bar(
        [f"{iv}min" for iv in shutdown_summary["interval_min"]],
        shutdown_summary["ratio_%"],
        color=colors[:len(shutdown_summary)]
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title("Server Shutdown Detection\n(% of REMOVE events matching N-min intervals)")
    ax.set_ylabel("Matching Ratio (%)")
    ax.set_ylim(0, shutdown_summary["ratio_%"].max() * 1.4 + 1)

    # (5) Weighted Score Top-30 Scatter
    ax = axes[1, 1]
    top30     = sched_df.head(30)
    colors_sc = ["tomato" if l == 1 else "steelblue" for l in top30["label"]]
    ax.scatter(range(1, len(top30) + 1), top30["weighted_score"],
               c=colors_sc, s=60, zorder=3)
    ax.plot(range(1, len(top30) + 1), top30["weighted_score"],
            color="gray", linewidth=0.8, alpha=0.5)
    ax.set_title("Top-30 Priority Schedule\n(Red=Actual Fail, Blue=Normal)")
    ax.set_xlabel("Schedule Rank")
    ax.set_ylabel("Weighted Score")
    ax.grid(True, linestyle="--", alpha=0.4)

    # (6) ROC Curve
    ax = axes[1, 2]
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    auc = roc_auc_score(y_val, y_pred_prob)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    plt.tight_layout()
    out = output_dir / "lgbm_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n시각화 저장: {out}")


# ───────────────────────────────────────────────
# 8. 메인
# ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Google Cluster Trace 2011 – Failure Prediction Pipeline")
    print("=" * 60)

    # 8-1. 로드
    task_df    = load_task_events(BASE_DIR, max_files=500)
    machine_df = load_machine_events(BASE_DIR)
    print(f"\ntask_events   shape : {task_df.shape}")
    print(f"machine_events shape: {machine_df.shape}")

    # 8-2. 전처리 + 피처 엔지니어링
    task_df    = preprocess_task_events(task_df)
    machine_df = preprocess_machine_events(machine_df)
    print(f"\n레이블 분포:\n{task_df['label'].value_counts()}")

    # 8-3. 서버 종료 감지
    shutdown_summary = detect_machine_shutdowns(machine_df, [15, 30, 45, 60])

    # 8-4. 데이터셋
    X, y, feature_cols = build_dataset(task_df)
    print(f"\n학습 데이터: X={X.shape}, 양성(실패) 비율={y.mean():.4f}")
    print(f"사용 피처 ({len(feature_cols)}개): {feature_cols}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8-5. 학습
    model = train_lgbm(X_tr, y_tr, X_val, y_val)

    # 8-6. 평가
    y_pred_prob = model.predict(X_val)
    y_pred      = (y_pred_prob >= 0.5).astype(int)
    auc         = roc_auc_score(y_val, y_pred_prob)

    print("\n" + "="*55)
    print("  모델 평가 결과")
    print("="*55)
    print(f"  ROC-AUC : {auc:.4f}")
    print(classification_report(y_val, y_pred, target_names=["Normal", "Fail"]))

    # 8-7. 우선순위 스케줄
    sched_df = build_priority_schedule(task_df, model, feature_cols, top_n=100)
    sched_df.to_csv(OUTPUT_DIR / "priority_schedule.csv")
    print(f"\n우선순위 스케줄 저장: {OUTPUT_DIR / 'priority_schedule.csv'}")
    print(sched_df.head(10).to_string())

    # 8-8. 저장
    shutdown_summary.to_csv(OUTPUT_DIR / "shutdown_summary.csv", index=False)
    plot_all(model, feature_cols, y_val, y_pred_prob,
             shutdown_summary, sched_df, OUTPUT_DIR)
    model.save_model(str(OUTPUT_DIR / "lgbm_failure_model.txt"))
    print(f"모델 저장: {OUTPUT_DIR / 'lgbm_failure_model.txt'}")
    print("\n완료!")


if __name__ == "__main__":
    main()
