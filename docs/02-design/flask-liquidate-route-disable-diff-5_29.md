# Flask "청산" 라우트 비활성화 diff 초안 — 5/29(금)

> **⚠️ SUPERSEDED BY `flask-queue-3items-disable-diff-5_29-v2.md`** — 본 1차는 `liquidate` 1건만 비활성화. 2차 전수 감사 결과 `start`/`stop`/`liquidate` 3건 모두 dead-letter 확정 → 2차가 3건 전체로 확장 대체. 이 문서는 이력 보존용. 적용은 2차 기준.
>
> **상태**: diff 초안 (코드 미적용, Codex 회신 후 사장님 별도 승인 시 적용)
> **계기**: 5/29 사장님 결단 — "Flask 청산 큐 라우트 비활성화 방향 승인 / 코드 변경은 HOLD"
> **상위 문서**: `docs/02-design/deletion-quarantine-audit-5_29.md` §3-1

---

## 0. 한 줄 요약

> **`website/flask_app.py` "청산" 라우트 (L956-968) + 도움말 메뉴 (L950)를 비활성화하여 dead-letter queue를 통한 canonical 우회 청산 경로를 차단하는 diff 초안. 적용은 Codex 회신 + 사장님 별도 승인 후.**

---

## 1. 변경 위치

| 위치 | 라인 | 현 상태 | 변경 후 |
|---|---|---|---|
| `website/flask_app.py:956-968` | "청산" 라우트 핸들러 | 큐에 `{'cmd': 'liquidate'}` 등록 + 안내 응답 | "기능 중단" 응답으로 교체, 큐 등록 코드 제거 |
| `website/flask_app.py:950` | 도움말 메뉴 안내 | `청산 — 전량 청산 (위험!)` | 해당 라인 제거 |

---

## 2. diff 초안

### 2-1. 라우트 비활성화 (L956-968)

```diff
@@ -956,15 +956,16 @@
     elif cmd == '청산':
-        q = _load_remote_queue()
-        q.append({'cmd': 'liquidate', 'ts': datetime.now().isoformat()})
-        _save_remote_queue(q)
+        # P3 (5/29 사장님 결단): Flask 청산 큐 비활성화.
+        # 사유: remote_queue.json 'liquidate' 명령의 polling 소비자가 없는 dead-letter queue 상태.
+        # 향후 polling 클라이언트 추가 시 canonical (order_intents_gate 10중 가드) 우회 위험.
+        # 청산이 필요하면: 텔레그램 "청산" 명령 (P1-A4 완료 경로) 또는 `python main.py emergency-stop` 사용.
         return {
-            'title': '💀 전량 청산 명령',
+            'title': '⛔ 청산 명령 비활성화',
             'lines': [
-                '🛑 청산 명령이 큐에 등록되었습니다.',
-                '로컬 시스템이 명령을 수신하면 전량 시장가 매도됩니다.',
+                '🛑 Flask 웹 청산 명령은 비활성화되었습니다.',
+                '청산이 필요하면 텔레그램 "청산" 명령 또는 emergency-stop CLI를 사용하세요.',
                 '',
-                '⚠️ 이 작업은 되돌릴 수 없습니다!',
-                '💡 텔레그램에서 "청산" 명령도 동일하게 작동합니다.',
+                '⚠️ 사유: canonical 매매 경로 (order_intents_gate 10중 가드) 우회 위험.',
+                '💡 P1-A4 완료된 텔레그램/CLI 경로만 사용하세요.',
             ]
         }
```

### 2-2. 도움말 메뉴 정리 (L946-953)

```diff
@@ -946,11 +946,10 @@
             '🎛️ [제어]',
             '  시작 — 자동매매 ON',
             '  정지 — 자동매매 OFF',
             '  상태 — 시스템 현황',
-            '  청산 — 전량 청산 (위험!)',
             '',
-            '💡 텔레그램 봇에서도 동일 명령 사용 가능',
+            '💡 텔레그램 봇에서도 동일 명령 사용 가능',
+            '⚠️ 청산은 텔레그램 또는 emergency-stop CLI에서만 가능',
         ]
         return {'title': '❓ 도움말', 'lines': lines}
```

---

## 3. 변경 영향 분석

### 3-1. 위험 차단
- ✅ Flask 라우트가 `remote_queue.json`에 `'liquidate'` 명령을 등록하는 경로 제거
- ✅ 향후 polling 클라이언트가 추가되어도 canonical 우회 경로 활성화 불가
- ✅ 도움말 메뉴에서 청산 옵션 제거 → 사용자 혼동 방지

### 3-2. 영향 없음
- Flask `시작` / `정지` 명령은 그대로 유지 (L790-792, L805-807 동일 큐 패턴이지만 매매 영향 없음)
- 텔레그램 `청산` 명령은 그대로 동작 (P1-A4 완료된 canonical 경로)
- `python main.py emergency-stop` CLI는 그대로 동작 (P2 완료된 canonical 경로)
- `data/remote_queue.json` 파일 자체는 유지 (시작/정지 명령용)

### 3-3. 잔존 위험 (정직 명시)
- Flask `시작` / `정지` 명령도 동일 큐 패턴 + 소비자 0건 → dead-letter
- 단, `시작` / `정지`는 KILL_SWITCH 토글만 (직접 매매 X) → canonical 우회 위험 낮음
- 별도 P3 audit 권장: Flask `시작` / `정지` 큐도 polling 소비자 추가 시 canonical 정합 확인 필요

---

## 4. 회귀 영향

### 4-1. 기존 회귀 영향 0건 (예상)
- 본 변경은 응답 문자열 + 큐 등록 코드만 수정
- 시그니처 변경 0건
- import 변경 0건
- 다른 파일 영향 0건

### 4-2. 신규 회귀 권장 (P3)
- `tests/test_flask_liquidate_disabled.py` (선택):
  - "청산" 명령 → 응답에 `'비활성화'` 포함 확인
  - `remote_queue.json` `'liquidate'` 등록 0건 확인

---

## 5. 적용 절차 (사장님 별도 승인 후)

```
[현재] diff 초안 작성 완료 (본 문서)
   ↓
[필수] Codex 회신 ①+② 도달
   ↓
[필수] 사장님 commit 결단
   ↓
[필수] 사장님 Flask 비활성화 별도 승인
   ↓
[적용] Edit으로 L946-968 변경 적용
   ↓
[필수] 회귀 25/25 + preflight 10/10 PASS 재확인
   ↓
[선택] 신규 회귀 `test_flask_liquidate_disabled.py` 작성
   ↓
[필수] Flask 서버 재시작 (사장님 명시 승인 후만)
```

---

## 6. 적용 금지 (본 diff 초안 작성 후)

- ❌ 본 diff 즉시 적용 X (Codex 회신 + 사장님 별도 승인 전까지)
- ❌ Flask 서버 재시작 X
- ❌ remote_queue.json 임시 비우기 X
- ❌ 도움말 메뉴 즉시 수정 X

---

## 7. 표현 룰

### 사용 가능
- "Flask 청산 라우트 비활성화 diff 초안 작성 완료"
- "dead-letter queue → 안전 응답 교체 방향"
- "적용은 Codex 회신 + 사장님 별도 승인 후"

### 사용 금지
- "Flask 비활성화 완료" X
- "canonical 우회 경로 차단 완료" X
- "운영 안전 완성" X

---

## 8. 연결 문서
- `docs/02-design/deletion-quarantine-audit-5_29.md` §3-1
- `docs/02-design/p1-truth-pack-5-29.md`
- `website/flask_app.py:946-969` (변경 대상)
- `src/telegram_command_handler.py:589, 926, 1442` (대체 경로 — P1-A4 완료)
- `main.py:611-645` (대체 경로 — P2 완료)
