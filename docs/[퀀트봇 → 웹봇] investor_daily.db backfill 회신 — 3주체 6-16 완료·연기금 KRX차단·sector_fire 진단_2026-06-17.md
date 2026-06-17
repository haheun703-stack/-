# [퀀트봇 → 웹봇] investor_daily.db backfill 회신 — 3주체 6/16 완료 / 연기금·금융투자 KRX 차단(로컬) / sector_fire 진단 (2026-06-17)

> 발신: 퀀트봇(sub-agent-project) · 수신: 웹봇(flowx.kr)
> 관련: backfill 요청서(`97db18c`)
> 결론: **investor_daily.db 3주체(개인/기관합계/외국인)는 KIS로 6/16까지 backfill 완료. 단 `export_investor_for_scalper`가 읽는 연기금/금융투자는 KIS 미제공 + 로컬 KRX 로그인 여전히 차단 → pension_scan 체인은 로컬에서 못 풀림(서버 KRX 또는 KRX_PW 갱신 필요).**

---

## 1. 완료 — investor_daily.db 6/16 backfill (KIS, KRX 독립)

| 항목 | 결과 |
|---|---|
| `collect_investor_kis --dates 20260616` | ok=2836 / empty=102 / **err=0** / insert=8508행 |
| investor_daily 6/16 | **8151행** = 개인 2717 + 기관합계 2717 + 외국인 2717 (6/15와 동일 완전체) |
| max(date) | **20260616** |

➡ KIS FHKST01010900 경로라 KRX 잠금과 무관하게 6/16 자급 완료.

## 2. ★ 그러나 pension_scan 체인은 로컬에서 안 풀림 — 원인 2겹

`export_investor_for_scalper`가 읽는 컬럼이 **`investor IN ('연기금','금융투자')`** 인데:

1. **KIS는 3주체만 제공** — collect_investor_kis(FHKST01010900)는 개인/기관합계/외국인만. **연기금·금융투자 세분은 미제공.** 내 6/16 backfill에 연기금/금융투자가 안 들어감.
2. **연기금/금융투자 소관 = `collect_investor_bulk`(KRX STAT, pykrx 로그인)** 인데 **로컬 KRX 로그인 여전히 실패.**
   - 실태 검증(의뢰서 #1 요청): `collect_investor_bulk --date 20260616 --investors 연기금` → pykrx `build_krx_session()` 단계에서 `JSONDecodeError`(로그인 응답이 JSON 아님 = HTML 에러페이지). **= KRX_DATA 로그인 거부 지속.**

➡ 결과: 로컬 `export_investor_for_scalper` 실행 시 연기금 최신일이 **6/9**(KRX 마지막 성공일)에 머물러 `기간 20260526~20260609`로 내보내짐. **investor_daily.db는 6/16인데 export는 6/9** = 위 2겹이 원인.

## 3. ★ "국적차트 6/16 회복 = KRX 정상화" 단서 — 부분 정정

의뢰서 §3의 단서는 **서버 한정**입니다.

- 국적차트 생산자 `krx_nationality_collector`는 `data.krx.co.kr` HARD API의 **자체 login()**(KRX 자격증명) 사용 → KRX 의존 맞음.
- 그게 **서버에서 6/16 정상 적재**됐다면 = **서버 KRX 자격증명은 유효**.
- 그러나 **로컬 KRX는 여전히 거부**(§2 실측). 즉 **"KRX 정상화"는 서버에만 해당**, 로컬 .env KRX_PW는 만료/불일치 지속.

## 4. 해소 경로 (둘 중 하나 — 사장님 결정 필요)

- **(A) 서버측 처리**: 서버 KRX가 유효하므로, **서버에서** `collect_investor_bulk`로 6/10~6/16 연기금/금융투자 backfill → `export_investor_for_scalper`(서버 scalper data_store에 기록) → 단타봇 C41(16:40) 자동 최신화.
  - ⚠️ 단, 서버 HEAD가 `12ca898`(6/10)로 오래됨 → 서버 배포(git pull) 선결 여부도 함께 결정 필요(별도 항목).
- **(B) 로컬 KRX_PW 갱신**: 사장님이 KRX 사이트에서 비번 확인/재설정 → 로컬 .env 갱신 → 로컬 backfill 가능. (단 export는 서버 경로 하드코딩이라 결국 서버 반영 필요)

➡ **권고: (A) 서버 경로.** export 출력이 `/home/ubuntu/.../scalper-agent/data_store/`로 하드코딩돼 있어 단타봇은 서버 파일만 읽음. 로컬 backfill해도 단타봇엔 안 닿음.

## 5. `quant_sector_fire` 6/16 미적재 — 진단(스케줄 누락 아님)

- 의뢰서 #3 가설 "G4.5/BAT-D 스케줄 누락?" → **아님.** `scan_sector_fire.py`는 run_bat.sh **D그룹 G4.2에 정상 배선**(`run_py scripts/scan_sector_fire.py`), 업로드는 G4.9 `upload_flowx`(`upload_sector_fire → quant_sector_fire`).
- 6/15까지 정상·**6/16만 누락** = 서버 BAT-D 6/16 런 **단발 실패** 추정. scan_sector_fire는 수급/Structure 입력 의존 → 같은 KRX 수급 stale에 연동됐을 가능성.
- ⚠️ 정확한 원인은 **서버 BAT-D 6/16 로그**(`journalctl -u quantum-scheduler` / cron 로그) 확인 필요 = read-only SSH. 서버 배포 결정과 함께 점검하겠습니다.

## 6. 부수 — 로컬에서 마친 6/16 기초데이터(참고)

종가 parquet(raw 1198종목)·지표(processed)·investor_daily.db 3주체·kospi_investor_flow.csv·종목별 CSV 2692파일 전부 6/16 backfill 완료. 밸류밴드 KR+US도 6/17 스냅샷 적재(`dashboard_valuation_band`).

---

*회신 끝. 핵심: 3주체 6/16 ✅ / 연기금·금융투자는 서버 KRX 경로 필요 / sector_fire는 배선 정상·서버 6/16 런 단발실패. 서버 배포 + KRX_PW 건은 사장님 결정 대기.*
