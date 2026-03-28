[README.md](https://github.com/user-attachments/files/26324721/README.md)
# 🖥️ Google Cluster Trace 2011 – Failure Prediction & Resource Management

LightGBM 기반 태스크 실패 예측 및 자원 우선순위 스케줄링 시스템입니다.  
Google Cluster Trace 2011 데이터를 활용하여 서버 안정성을 높이고, 자원 예측 정확도를 개선합니다.

---

## 📌 주요 기능

- **태스크 실패 예측** — LightGBM 이진 분류 (EVICT / FAIL / LOST → label=1)
- **가중치 기반 우선순위 스케줄링** — 실패 확률 × 우선순위로 자원 배분 순서 결정
- **서버 종료 주기 감지** — 15 / 30 / 45 / 60분 단위 종료 패턴 자동 감지
- **피처 엔지니어링** — task 재시도 횟수, 머신 밀집도, 시간대별 실패율 등 15개 피처

---

## 📁 레포지토리 구조

```
google-cluster-trace-lgbm/
│
├── google_trace_lgbm.py       # 메인 실행 파일
│
├── data/
│   ├── task_events/           # Google Trace task_events 파일 위치
│   │   └── (part-*.csv.gz)
│   └── machine_events/        # Google Trace machine_events 파일 위치
│       └── (part-*.csv.gz)
│
├── outputs/                   # 실행 결과 자동 저장
│   ├── lgbm_results.png           # 시각화 결과
│   ├── priority_schedule.csv      # 우선순위 스케줄
│   ├── shutdown_summary.csv       # 서버 종료 주기 감지 결과
│   └── lgbm_failure_model.txt     # 저장된 LightGBM 모델
│
├── requirements.txt           # 의존 패키지 목록
└── README.md
```

---

## 📊 데이터

[Google Cluster Trace 2011 (clusterdata-2011-2)](https://github.com/google/cluster-data)

### 다운로드 방법

```bash
# Google Cloud SDK 설치 후
gsutil -m cp -r gs://clusterdata-2011-2/task_events/    ./data/task_events/
gsutil -m cp -r gs://clusterdata-2011-2/machine_events/ ./data/machine_events/
```

> ⚠️ 전체 데이터는 수십 GB입니다. 테스트 시 일부 파일만 사용하는 것을 권장합니다.

### 사용 스키마

**task_events**

| 컬럼 | 설명 |
|---|---|
| timestamp | 이벤트 발생 시각 (마이크로초) |
| job_id | 작업 ID |
| task_index | 태스크 인덱스 |
| machine_id | 머신 ID |
| event_type | 이벤트 유형 (0=SUBMIT ~ 8=UPDATE_RUNNING) |
| priority | 우선순위 (0~11) |
| cpu_request | CPU 요청량 |
| mem_request | 메모리 요청량 |
| disk_request | 디스크 요청량 |

**machine_events**

| 컬럼 | 설명 |
|---|---|
| timestamp | 이벤트 발생 시각 (마이크로초) |
| machine_id | 머신 ID |
| event_type | 0=ADD, 1=REMOVE, 2=UPDATE |

---

## ⚙️ 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 경로 설정

`google_trace_lgbm.py` 상단의 `BASE_DIR`을 데이터 경로에 맞게 수정합니다.

```python
# Windows
BASE_DIR = Path(r"C:\Users\yourname\Desktop\trace data")

# Linux / Mac
BASE_DIR = Path("./data")
```

### 3. 실행

```bash
python google_trace_lgbm.py
```

---

## 🔧 피처 엔지니어링

| 피처 | 설명 |
|---|---|
| `cpu_request` | CPU 요청량 |
| `mem_request` | 메모리 요청량 |
| `disk_request` | 디스크 요청량 |
| `scheduling_class` | 스케줄링 클래스 |
| `priority` | 태스크 우선순위 |
| `priority_norm` | 정규화된 우선순위 (0~1) |
| `different_machine_constraint` | 다른 머신 배치 제약 |
| `job_task_count` | job 당 태스크 수 |
| `task_retry_count` | 태스크 재시도 횟수 |
| `machine_task_density` | 머신 당 태스크 밀집도 |
| `time_bucket_fail_rate` | 1시간 단위 시간대별 실패율 |
| `cpu_mem_pressure` | CPU × MEM 복합 자원 압박 지표 |
| `job_fail_rate` | job 내 실패 비율 |
| `machine_fail_rate` | 머신별 누적 실패율 |
| `sched_class_fail_rate` | scheduling class별 실패율 |

---

## 🏆 우선순위 스케줄링 방식

```
weighted_score = fail_prob × (1 + priority_norm)
```

- `fail_prob` — LightGBM이 예측한 태스크 실패 확률 (0~1)
- `priority_norm` — 정규화된 태스크 우선순위 (0~1)
- **점수가 높은 태스크부터 자원을 우선 배분**하여 실패 위험을 사전에 차단합니다.

---

## 🖥️ 서버 종료 주기 감지

`machine_events`의 REMOVE(event_type=1) 이벤트를 분석하여  
머신별 연속 종료 간격이 **15 / 30 / 45 / 60분 배수**에 해당하는 비율을 계산합니다.

```
허용 오차: 각 주기의 ±5%
```

---

## 📈 출력 결과

### `lgbm_results.png`

| 그래프 | 설명 |
|---|---|
| Feature Importance | 피처별 중요도 (Gain 기준) |
| Confusion Matrix | 실제 vs 예측 결과 |
| Fail Probability Distribution | 실패 확률 분포 |
| Server Shutdown Detection | 15/30/45/60분 종료 감지 비율 |
| Top-30 Priority Schedule | 우선순위 상위 30개 태스크 |
| ROC Curve | AUC 기반 모델 성능 |

### `priority_schedule.csv`

| 컬럼 | 설명 |
|---|---|
| schedule_rank | 우선순위 순위 |
| job_id | 작업 ID |
| task_index | 태스크 인덱스 |
| machine_id | 배정된 머신 ID |
| priority | 원본 우선순위 |
| fail_prob | 예측 실패 확률 |
| weighted_score | 가중 점수 |
| label | 실제 실패 여부 |

---

## 🛠️ 모델 설정

| 파라미터 | 값 | 설명 |
|---|---|---|
| objective | binary | 이진 분류 |
| num_leaves | 127 | 트리 복잡도 |
| learning_rate | 0.05 | 학습률 |
| early_stopping | 50 round | 과적합 방지 |
| scale_pos_weight | 자동 계산 | 클래스 불균형 보정 |

---

## 📋 실험 결과 요약

| 파일 수 | 데이터 건수 | 실패 비율 | ROC-AUC |
|---|---|---|---|
| 1개 | 450,146 | 1.09% | 0.9076 |
| 21개 | 9,766,805 | 22.04% | 0.7117 |
| 500개 (전체) | ~2억+ | — | — |

---

## 📦 requirements.txt

```
lightgbm
scikit-learn
pandas
numpy
matplotlib
```

---

## 📄 라이선스

MIT License

---

## 🔗 참고

- [Google Cluster Data GitHub](https://github.com/google/cluster-data)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Google Cluster Trace 2011 논문](https://research.google/pubs/pub40325/)
