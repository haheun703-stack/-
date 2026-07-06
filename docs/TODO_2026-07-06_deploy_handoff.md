# US FV v1 배포 핸드오프 (사무실 재개용) — 2026-07-06

> 아침 세션에서 미국판 미래가치 엔진 v1을 **착수→VPS검증→백테스트→적대검수→커밋+push→BAT-D 배선**까지 완료.
> **남은 건 VPS 배포 1스텝뿐**(장중 금지라 대기). 이 문서만 보고 사무실에서 마무리 가능.

## ✅ 완료 (로컬·GitHub 동기화, `f6a991c`)
- 커밋 3건 전부 push 완료: `208eeda`(엔진 3스택+백테스트) → `117d334`(load_dotenv) → `f6a991c`(BAT-D G5.7 배선).
- 코드/스크립트 6+2파일 전부 VPS에서 **실측 검증 완료**(scan_consensus_us 47종·PER밴드·엔진 47종·러너 스냅샷).
- 세션 메모리 기록: `session_state_7_6_fv_us_v1`.

## ⏳ 남은 1스텝 — VPS 배포 (장마감 15:30 후 ~ BAT-D 16:30 전)
**왜 대기**: VPS `git pull`은 장중(09:00~15:30) 금지(11:30 BAT-H 등 장중 cron이 새 코드 물 위험).
현재 VPS는 아직 옛 코드(`ecf2c5d`) + scp 검증본(untracked). 아래 배포로 `f6a991c` 정본화 + G5.7 활성화.

**★필요 조건**: SSH pem이 있는 머신에서 실행.
`pem = D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem` (이 로컬 머신에 위치).
사무실이 **다른 머신**이면 pem 복사 필요, 또는 이 머신에 원격 접속해서 실행.

### 1) 배포 명령 (한 줄)
```bash
ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" ubuntu@13.209.153.221 \
"cd ~/quantum-master && \
 git checkout -- src/use_cases/valuation_band.py src/use_cases/index_regime.py && \
 rm -f scripts/scan_consensus_us.py scripts/run_future_value_us.py \
       scripts/research/per_band_backtest_us.py \
       src/use_cases/valuation_band_history_us.py src/use_cases/future_value_engine_us.py \
       data/consensus_us_history.jsonl && \
 git pull origin main && rm -rf src/use_cases/__pycache__ && \
 git log --oneline -1 && grep -c run_future_value_us scripts/cron/run_bat.sh"
```
(scp 검증본 정리 → `git pull`로 `f6a991c` 반영 → `__pycache__` 삭제 → 테스트 스냅샷 삭제해 오늘밤 종가 스냅샷을 첫 기록으로)

### 2) 검증
- `git log` HEAD = **`f6a991c`** 확인
- `grep -c run_future_value_us` = **1** (G5.7 배선 확인)
- 러너 1회 수동 실행:
  ```bash
  ssh -i "<pem>" ubuntu@13.209.153.221 "cd ~/quantum-master && ./venv/bin/python3.11 scripts/run_future_value_us.py 2>&1 | tail -5"
  ```
  → `data/shadow/future_value_us.json`(47종) + `data/consensus_us_history.jsonl`(첫 스냅샷·장마감가) 생성 확인

### 3) 확인 사항
- **오늘밤 BAT-D(16:30)부터 US FV 자동 산출** 시작 → 매일 목표가 스냅샷 축적 → **20거래일 뒤 forward 검증**(컨센서스 괴리축이 실제 SPY 초과했나).
- **미배포해도 무해**: 다음 배포일 BAT-D부터 축적 시작(하루 늦어질 뿐).

## 📌 배경 (엔진 핵심 판정 — 잊지 말 것)
- **US PER밴드·저PER = 무효**(백테스트 t=0.61·reliable 서브셋 -0.61%p 역전). US 성장장에선 가치 틸트 안 통함.
- → 엔진 실점수는 **사이클(매수적기)+실적가속**만. 밸류는 관측 태그. 컨센서스 괴리는 forward 검증 중.
- **shadow_unvalidated**: 실주문0·매매 미배선. 승격은 20일 forward 게이트 통과 후(퐝가님 승인).

## 🔜 다음 후보 (배포 후, 여유 시)
- forward 검증 리더 스크립트(`consensus_us_history.jsonl` 20일 뒤 D+20 SPY초과 집계) — 지금 미리 만들어두면 20일 뒤 바로 검증 가능.
- 한국 미결: 메인A 리밸런스 SCAN/AA 잔존구멍(퐝가님 결정 대기) · BRAIN BULL 비대칭 완화.
