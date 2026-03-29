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
