# 인사이트 시그널 — 소수의견 관찰 모듈

김동엽 인터뷰의 투자 원칙을 시스템화한 **관찰 전용** 모듈.
기존 페이퍼 트레이딩 매매 로직에 일절 개입하지 않고, 시그널만 수집·기록하여
실제 수익률이 검증된 뒤 스코어링 승격을 판단한다.

## 원칙 → 구현 매핑

| 인터뷰 원칙 | 구현 |
|---|---|
| "사상 최대·게임 체인저 키워드 뉴스" | 한경 RSS + 네이버 금융 뉴스 제목 키워드 스캔 → 상장사 매핑 |
| "임원들이 자사주 사면 항상 올라" | DART D002 공시(임원·주요주주 소유상황보고) 매수 건 집계 |
| "힘 있는 소수를 따르라 (소수의견)" | KIS 투자자별 수급: 기관+외인 매집 & 개인 이탈 = 가점, 개인 쏠림 = 감점 |
| "귀에 들리면 이미 늦다" | 뉴스 단독보다 뉴스+DART 동시 포착에 시너지 가중 (1.25배) |

## 설치 (3단계)

1. **폴더 복사** — 이 패키지 내용물을 프로젝트 루트에 그대로 복사:
   ```
   D:\sub-agent-project_퀀트봇\
   ├── insight_signals\            ← 통째로 복사
   ├── config\insight_signals.yaml ← 기존 config 폴더에 추가
   ├── run_insight_signals.bat     ← 루트에 추가
   └── run_insight_track.bat       ← 루트에 추가
   ```
2. **DART 키 발급** — https://opendart.fss.or.kr → 인증키 신청 (무료, 즉시).
   `.env`에 한 줄 추가:
   ```
   DART_API_KEY=발급받은40자리키
   ```
3. **KIS 키 (선택)** — `.env`의 KIS 변수명이 `KIS_APP_KEY` / `KIS_APP_SECRET`가
   아니면 `config/insight_signals.yaml`의 `env_names`만 실제 변수명으로 수정.
   KIS 키가 없어도 뉴스+DART 시그널은 정상 동작 (수급 필터만 생략됨).

## 실행

```bat
run_insight_signals.bat            :: 일일 수집 (장 마감 후)
run_insight_signals.bat --dry-run  :: 동작 확인 (뉴스만, 키 최소 요구)
run_insight_track.bat              :: 누적 성과 단독 평가
run_insight_backtest.bat           :: D002 과거 백테스트 (--months 6 기본)
```

Lightsail(cron) 배포 시 — 정본은 cron이므로 crontab에 별도 라인으로 추가.
⚠ 실행 시각은 **기존 단계(16:30 D, 17:00 J 등)와 겹치지 않게** 서버 스케줄을
아는 쪽(퀀트봇 Claude)이 최종 결정할 것. RAM 2GB 공유 환경이므로 다른 단계와
동시 실행 금지. 참고용 예시 (17:00 J 단계가 충분히 끝난 뒤):
```cron
# 인사이트 시그널 (평일, 시각은 서버 스케줄 확인 후 확정)
40 17 * * 1-5 cd /home/ubuntu/quantum-master && venv/bin/python -u -X utf8 -m insight_signals.agent.run_daily >> logs/cron_insight.log 2>&1
```
⚠ 기존 `run_bat.sh` 단계(A~HEALTH)는 건드리지 말 것.

## KIS 토큰 캐시 공유 (필수 확인)

KIS 토큰은 발급 횟수 제한이 있어 기존 봇과 이 모듈이 **각자 발급하면
rate limit 충돌**이 난다 (7/7 실측 사례 있음). 반드시
`config/insight_signals.yaml`의 `flow.kis_token_cache`에 기존 봇이 쓰는
토큰 캐시 파일 경로를 지정해 공유할 것.
- 우리 포맷(`{"access_token","issued_at"}`)과 KIS 공식 예제
  포맷(`{"token","valid-date"}`) 모두 읽는다.
- **남의 포맷 파일은 절대 덮어쓰지 않는다** (기존 봇 캐시 보호 —
  이 경우 토큰 만료 시 이 모듈이 새로 발급은 하되 저장하지 않으므로,
  가급적 기존 봇 포맷 경로를 지정해 재발급 자체가 없게 할 것).

## D002 백테스트 (승격 판단은 이걸 먼저)

임원 매수 시그널은 forward 관찰 30픽을 기다릴 필요 없이 과거 공시로
즉시 검증 가능하다 (백테스트 우선 원칙):

```bat
run_insight_backtest.bat --months 6
run_insight_backtest.bat --months 12 --baseline-sample 5
```

- 진입: 공시일 T+1 종가 (선견 편향 방지) / 수익률: +5/+10/+20 거래일
- **기저선 = 같은 날 유니버스 무작위 진입** (이벤트당 K종목, seed 고정 재현 가능)
- 판정: 초과 평균·초과 중앙값 모두 양수 + 승률 차 +5%p 이상, 세 구간 방향 일치
- 시세는 네이버 fchart(키 불필요), DART 상세 조회는 회사당 1콜 + 0.15s 간격

## 산출물

| 파일 | 내용 |
|---|---|
| `data/insight_signals/signals_YYYY-MM-DD.json` | 당일 전체 시그널 (소스·점수·근거) |
| `data/insight_signals/picks_log.csv` | 누적 픽 로그 + 픽 시점 가격 |
| `data/insight_signals/report_YYYY-MM-DD.md` | 일일 리포트 (픽 + 근거 + 누적 성과) |
| `data/insight_signals/performance_*.csv` | +5/+10/+20일 수익률 평가 |

## 스코어링 승격 기준 (v0.2 교정판)

관찰/백테스트 데이터가 쌓인 뒤 아래를 **모두** 만족하는 소스만 실제
종목 선정 스코어에 가중치로 편입할 것:

1. 표본 수 n ≥ 30 (해당 소스가 기여한 픽 기준)
2. +10일 평균 수익률이 **같은 날 유니버스 무작위 진입(비이벤트 기저선)
   대비 우위** — 페이퍼 평균 비교는 레짐 베타에 속으므로 금지
   (backtest_dart의 기저선 방식과 동일 기준)
3. 승률이 기저선 승률 대비 +5%p 이상
4. 특정 1~2 종목이 평균을 끌어올린 게 아닐 것 (중앙값도 양수)
5. D002 소스는 forward 관찰 이전에 백테스트(`run_insight_backtest.bat`)
   초과성과 확인을 선행할 것

승격 시에도 처음엔 보조 가중치(전체 스코어의 10~20%)로 시작 권장.

## 한계 및 주의

- **뉴스 종목 매핑은 제목 부분일치** 기반이라 오탐 가능
  (지주사 짧은 이름 등). `name_blacklist`로 튜닝할 것.
- DART D002에는 스톡옵션 행사·상속 등 '매수 아닌 증가'도 섞임.
  v2에서 취득방법 필드 파싱으로 정밀화 예정.
- 네이버 금융 HTML 구조가 바뀌면 해당 소스만 자동 스킵됨 (RSS는 유지).
- KIS 토큰은 `data/insight_signals/kis_token.json`에 캐시 —
  기존 봇의 토큰 캐시와 별개이므로 발급 횟수 제한에 주의
  (기존 봇이 토큰 파일을 쓰고 있다면 그 경로를 공유하도록 v2에서 통합 가능).

## 아키텍처

```
insight_signals/
├── entities.py          # Signal, DailyPick, InsiderFiling ... (의존성 없음)
├── use_cases/
│   ├── collect.py       # 점수화·합산 로직 (entities만 import)
│   └── evaluate.py      # 성과 평가
├── adapters/
│   ├── dart_client.py   # DART OpenAPI
│   ├── news_client.py   # RSS + 네이버 (requests + stdlib만)
│   ├── kis_flow_client.py  # KIS 수급/현재가 (키 없으면 자동 스킵)
│   └── price_client.py  # 시세 폴백 (KIS → 네이버)
└── agent/
    ├── run_daily.py     # 진입점 (조립)
    ├── track_performance.py
    ├── report.py
    └── _env.py          # .env/config 로더
```
안쪽 계층은 바깥 계층을 import 하지 않는다 (프로젝트 규칙 준수).
