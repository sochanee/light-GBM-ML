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
