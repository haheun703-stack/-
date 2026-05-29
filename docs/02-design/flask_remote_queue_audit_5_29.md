# Flask remote_queue 전수 감사 2차 — 5/29(금)

> **상태**: read-only 2차 감사 (코드 변경 0건, B-8 우선순위 재평가)
> **계기**: 5/29 사장님 결단 — "Flask 시작/정지 큐도 P3 audit 후보로 등록 + 전수 감사"
> **상위 문서**: `docs/02-design/deletion-quarantine-audit-5_29.md` §3-1 + `docs/01-plan/v2-backlog-5_29.md` B-11
> **검수 방식**: Explore 서브에이전트 분업 (12,855 파일 전수 탐색)

---

## 0. 한 줄 결론

> **Flask remote_queue 명령 3건 (start / stop / liquidate) 모두 polling 소비자 0건 dead-letter queue 확정. 1차 audit "청산만" → 2차 audit "3건 모두" QUARANTINE 권장. B-8 diff 초안을 3건 전체로 확장 필요.**

---

## 1. polling 클라이언트 추적 결과 ★

### 1-1. 추적 범위
- 전체 파이썬 파일 **12,855개**
- `/api/remote/queue` 문자열 grep
- `/api/remote/exec` 문자열 grep
- `requests.(get|post).*remote` 패턴 grep
- `remote_queue.json` 직접 참조 grep
- VPS crontab + bodyhunter (텔레그램 봇) + flowx-web (Next.js) + static HTML/JS 모두 포함

### 1-2. 결과
- **polling 클라이언트 존재 여부: NO**
- `/api/remote/queue` GET endpoint 호출 코드: **0건** (정의 1건만 — `website/flask_app.py:510`)
- `/api/remote/exec` POST endpoint 호출 코드: **0건** (정의 1건만 — `website/flask_app.py:494`)
- `data/remote_queue.json` 파일 자체 미생성 (현재 시점)

### 1-3. 의미
- L513 주석 "로컬 머신이 폴링"은 **의도된 설계만 명시, 실제 구현 0건**
- canonical 우회 청산/제어 경로는 **현재 활성화되어 있지 않음**
- 그러나 polling 클라이언트가 향후 추가되면 **즉시 활성화 위험** 존재

---

## 2. Flask remote_queue 명령 전수 매트릭스

### 2-1. 큐 등록 패턴 grep 결과

| cmd | 라우트 라인 | 도움말 노출 | UI 노출 | polling 소비자 | 트리거 동작 (의도) | 5분류 |
|---|---|---|---|---|---|---|
| **`'start'`** | L789-792 | ✅ L947 "시작" | ❌ | ❌ | KILL_SWITCH 해제 (의도) | **QUARANTINE** |
| **`'stop'`** | L804-807 | ✅ L948 "정지" | ❌ | ❌ | KILL_SWITCH 생성 (의도) | **QUARANTINE** |
| **`'liquidate'`** | L956-968 | ✅ L950 "청산" | ❌ | ❌ | 전량 시장가 매도 (의도) | **QUARANTINE** |

### 2-2. 대조 — 다른 cmd (KEEP 23+ 개)
- 스캔 그룹: '스캔' / '스윙스캔' / '사전감지' (조회만, 매매 0)
- AI 그룹: '종목선정' / 'AI모니터' / '뉴스AI' (조회만)
- 계좌 그룹: '현재잔고' / '포트폴리오' (조회만)
- 유틸 그룹: '도움' / '상태' / '로그' (정보 조회)

→ 큐 등록 패턴 + 트리거 의도가 **매매/제어**인 명령만 QUARANTINE 대상.

---

## 3. UI 노출 지점 매트릭스

| UI 파일 | `/api/remote/exec` 또는 `/api/remote/queue` 참조 | start | stop | liquidate |
|---|---|---|---|---|
| `static/jarvis.html` | 0건 | - | - | - |
| `quantum_master_dashboard.html` | 0건 | - | - | - |
| `flowx-web/` (Next.js) | 0건 | - | - | - |
| 그 외 HTML/JS/TSX | 0건 | - | - | - |

**결론**: Flask 도움말 메뉴 (L928-954) 에는 3 명령 노출되지만 **외부 UI에서 호출되지 않음**.

---

## 4. 5분류 종합 결단

### 4-1. QUARANTINE (3건 ★ — 1차 audit 1건 → 2차 audit 3건 확장)

| # | 명령 | 위험 시나리오 | 격리 방안 |
|---|---|---|---|
| 1 | `'start'` | polling 클라이언트 추가 시 KILL_SWITCH 무인 해제 (canonical 우회 자동매매 재가동) | L789-792 라우트 비활성화 + L947 도움말 제거 |
| 2 | `'stop'` | polling 클라이언트 추가 시 KILL_SWITCH 무인 생성 (의도된 정지지만 audit 로그 부재) | L804-807 라우트 비활성화 + L948 도움말 제거 |
| 3 | `'liquidate'` | polling 클라이언트 추가 시 canonical 우회 전량 청산 | L956-968 라우트 비활성화 + L950 도움말 제거 |

### 4-2. KEEP (조회/분석 명령 23+ 건)
- 매매/제어 트리거 없음, polling 클라이언트 없어도 위험 0
- `/api/remote/exec` 자체는 유지 (조회 명령용)

### 4-3. KEEP+GUARD / DELETE / MERGE
- 해당 없음

---

## 5. B-8 (Flask 청산 라우트 비활성화) 우선순위 재평가

### 5-1. 1차 audit (5/29 오후) 결단
- 대상: `'liquidate'` 1건만
- diff 초안: `docs/02-design/flask-liquidate-route-disable-diff-5_29.md`

### 5-2. 2차 audit (본 문서) 결단
- 대상: **`'start'` + `'stop'` + `'liquidate'` 3건 전체**
- diff 초안 확장 필요:
  - L789-792 `'start'` 라우트 비활성화
  - L804-807 `'stop'` 라우트 비활성화
  - L956-968 `'liquidate'` 라우트 비활성화 (1차와 동일)
  - L947-950 도움말 메뉴에서 3 명령 모두 제거
- **권장**: 1차 diff 초안을 별도 PDCA에서 확장 (사장님 별도 결단 후)

### 5-3. 우선순위 변경
- **이전**: P2 (Codex 회신 + 별도 승인 후 적용)
- **변경 후**: P2 유지 (위험도 동일 — polling 클라이언트 추가 시점이 위험 발생 시점)
- 단, diff 범위 확장 → 작업량 증가 (~30분 → ~45분)

---

## 6. 잔여 위험 (정직 명시)

### 6-1. polling 클라이언트가 미래 추가될 가능성
- 시나리오: 누군가 (사장님 본인 포함) `requests.get(API/remote/queue)` 폴링 스크립트를 작성하면 dead-letter → 활성 큐로 즉시 전환
- 차단 방안: L494 `/api/remote/exec` 라우트에서 `cmd in ('start', 'stop', 'liquidate')` 강제 거부 (큐 등록 자체 차단)

### 6-2. 텔레그램 대체 경로
- 텔레그램 `/시작` (L398 `_execute_auto_start`) → KILL_SWITCH 해제 (P1-A4 완료)
- 텔레그램 `/정지` (L1203 `_cmd_stop`) → STOP.signal + KILL_SWITCH 생성 (P1-A4 완료)
- 텔레그램 `/청산` (L926 `_cmd_liquidate` → L589 `_execute_liquidate`) → canonical 매도 (P1-A4 완료)
- → **Flask 큐 비활성화해도 텔레그램 경로 안전 유지**

### 6-3. `data/remote_queue.json` 파일 자체
- 현재 시점: 미생성
- B-8 적용 후: 큐 등록 코드 제거 → 파일 영구 미생성
- 이미 존재한다면: 파일 삭제 권장 (별도 결단)

---

## 7. 적용 금지 (본 감사 후)

- ❌ 즉시 라우트 비활성화 X (Codex 회신 + 사장님 별도 승인 후만)
- ❌ `data/remote_queue.json` 임시 삭제 X
- ❌ Flask 서버 재시작 X
- ❌ 단독 commit X

---

## 8. 표현 룰

### 사용 가능
- "Flask remote_queue 전수 감사 1차 완료"
- "polling 클라이언트 0건 + 명령 3건 QUARANTINE 권장"
- "B-8 diff 범위 확장 필요 (1건 → 3건)"

### 사용 금지
- "dead-letter 영구 확정" X (polling 추가 시 활성화 가능성)
- "Flask 우회 경로 완전 차단" X (코드 미적용)
- "운영 안전 완성" X

---

## 9. 다음 단계 권장

1. 사장님께 본 결과 보고 (3건 → 3건 QUARANTINE 확장 결단 요청)
2. 결단 후 B-8 diff 초안 확장 (3 라우트 + 도움말 3 라인)
3. Codex 회신 ①+② 도달 + 사장님 별도 승인 시 적용
4. 적용 후 회귀 25/25 + preflight 10/10 PASS 재확인
5. (선택) `tests/test_flask_queue_disabled.py` 신규 회귀 작성 (3 라우트 'error' 응답 검증)

---

## 10. 연결 문서
- `docs/02-design/deletion-quarantine-audit-5_29.md` §3-1 (1차 결과)
- `docs/02-design/flask-liquidate-route-disable-diff-5_29.md` (1차 diff 초안 — 확장 필요)
- `docs/01-plan/v2-backlog-5_29.md` B-8, B-11
- `website/flask_app.py:480-530` (큐 인프라)
- `website/flask_app.py:789-808, 956-968` (3 라우트)
- `website/flask_app.py:928-954` (도움말 메뉴)
- `src/telegram_command_handler.py:620, 1203, 589, 926` (텔레그램 대체 경로 — P1-A4 완료)
