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
