# Flask remote_queue 3건 비활성화 diff 초안 (2차 확장) — 5/29(금)

> **상태**: diff 초안 (코드 미적용, Codex 회신 + 사장님 별도 승인 후만)
> **계기**: 5/29 사장님 결단 — "Flask queue 3건 확장 승인 / liquidate만이 아니라 start, stop, liquidate 3건 전체를 B-8 범위로 확장"
> **상위 문서**:
> - `docs/02-design/flask-liquidate-route-disable-diff-5_29.md` (1차, liquidate만)
> - `docs/02-design/flask_remote_queue_audit_5_29.md` (2차 전수 감사 결과)
> - `docs/01-plan/v2-backlog-5_29.md` B-8

---

## 0. 한 줄 요약

> **1차 diff는 `liquidate` 1건만 비활성화. 2차 전수 감사 결과 `start`/`stop`/`liquidate` 3건 모두 polling 소비자 0건 dead-letter 확정 → 본 2차 diff는 3 라우트 전체 + 도움말 3 라인 제거 기준으로 확장. 적용은 Codex 회신 + 사장님 별도 승인 후만.**

---

## 1. 1차 → 2차 확장 사유

| 항목 | 1차 (5/29 오후) | 2차 (5/29 저녁) |
|---|---|---|
| 대상 | `liquidate` 1건 | `start` + `stop` + `liquidate` **3건** |
| 근거 | 1차 audit (간이) | 2차 전수 감사 (12,855 파일 전수 grep) |
| polling 소비자 확인 | 추정 | **확정 0건** |
| 도움말 제거 | L950 1줄 | **L947 + L948 + L950 3줄** |
| 작업량 | ~30분 | ~45분 |
| 우선순위 | P2 | P2 유지 (위험도 동일) |

**확장 근거**: 3 명령 모두 동일 dead-letter 패턴 (큐 등록 + polling 소비자 0건). polling 클라이언트가 미래 추가될 경우 3 명령 모두 canonical 우회 경로로 전환 가능.

---

## 2. 변경 위치 (3 라우트 + 도움말 3 라인)

| 위치 | 라인 | 현 상태 | 변경 후 |
|---|---|---|---|
| `website/flask_app.py:789-792` | `'시작'` 라우트 핸들러 | 큐 `{'cmd': 'start'}` 등록 + 안내 응답 | "기능 중단" 응답 + 큐 등록 제거 |
| `website/flask_app.py:804-807` | `'정지'` 라우트 핸들러 | 큐 `{'cmd': 'stop'}` 등록 + 안내 응답 | "기능 중단" 응답 + 큐 등록 제거 |
| `website/flask_app.py:956-968` | `'청산'` 라우트 핸들러 | 큐 `{'cmd': 'liquidate'}` 등록 + 안내 응답 | "기능 중단" 응답 + 큐 등록 제거 |
| `website/flask_app.py:947` | 도움말 메뉴 "시작" 안내 | `'  시작 — 자동매매 ON'` | 해당 라인 제거 |
| `website/flask_app.py:948` | 도움말 메뉴 "정지" 안내 | `'  정지 — 자동매매 OFF'` | 해당 라인 제거 |
| `website/flask_app.py:950` | 도움말 메뉴 "청산" 안내 | `'  청산 — 전량 청산 (위험!)'` | 해당 라인 제거 |

---

## 3. diff 초안

### 3-1. `'시작'` 라우트 비활성화 (L789-802)

```diff
@@ -789,14 +789,17 @@
     elif cmd == '시작':
-        q = _load_remote_queue()
-        q.append({'cmd': 'start', 'ts': datetime.now().isoformat()})
-        _save_remote_queue(q)
+        # B-8 (5/29 사장님 결단 2차): Flask remote_queue 'start' 비활성화.
+        # 사유: polling 소비자 0건 dead-letter queue. polling 클라이언트 추가 시
+        #       canonical 우회 KILL_SWITCH 해제 경로 활성화 위험.
+        # 대체: 텔레그램 "시작" 명령 (P1-A4 완료) 또는 KILL_SWITCH 파일 직접 삭제.
         return {
-            'title': '🟢 자동매매 시작 명령',
+            'title': '⛔ Flask 시작 명령 비활성화',
             'lines': [
-                '✅ 시작 명령이 큐에 등록되었습니다.',
-                '로컬 시스템이 명령을 수신하면 KILL_SWITCH가 해제됩니다.',
+                '🛑 Flask 웹 시작 명령은 비활성화되었습니다.',
+                '자동매매 시작이 필요하면 텔레그램 "시작" 명령을 사용하세요.',
                 '',
-                '💡 텔레그램에서 "시작" 명령도 동일하게 작동합니다.',
+                '⚠️ 사유: canonical 매매 경로 우회 위험 (polling 소비자 0건 dead-letter).',
+                '💡 P1-A4 완료된 텔레그램 경로만 사용하세요.',
             ],
-            'live_status': True
+            'live_status': False
         }
```

### 3-2. `'정지'` 라우트 비활성화 (L804-815)

```diff
@@ -804,12 +804,16 @@
     elif cmd == '정지':
-        q = _load_remote_queue()
-        q.append({'cmd': 'stop', 'ts': datetime.now().isoformat()})
-        _save_remote_queue(q)
+        # B-8 (5/29 사장님 결단 2차): Flask remote_queue 'stop' 비활성화.
+        # 사유: polling 소비자 0건 dead-letter queue.
+        # 대체: 텔레그램 "정지" 명령 (P1-A4 완료, STOP.signal + KILL_SWITCH 생성).
         return {
-            'title': '🔴 자동매매 정지 명령',
+            'title': '⛔ Flask 정지 명령 비활성화',
             'lines': [
-                '✅ 정지 명령이 큐에 등록되었습니다.',
-                # (기존 안내 라인들)
+                '🛑 Flask 웹 정지 명령은 비활성화되었습니다.',
+                '자동매매 정지가 필요하면 텔레그램 "정지" 명령을 사용하세요.',
+                '',
+                '⚠️ 사유: polling 소비자 0건 dead-letter (canonical 우회 위험).',
+                '💡 P1-A4 완료된 텔레그램 경로만 사용하세요.',
             ]
         }
```
**※** L808 이후 기존 응답 라인 (확인 필요 — 본 diff 작성 시 일부만 표시. 실제 Edit 시 전체 라인 확인 의무)

### 3-3. `'청산'` 라우트 비활성화 (L956-968) — 1차 diff와 동일

```diff
@@ -956,15 +956,16 @@
     elif cmd == '청산':
-        q = _load_remote_queue()
-        q.append({'cmd': 'liquidate', 'ts': datetime.now().isoformat()})
-        _save_remote_queue(q)
+        # B-8 (5/29 사장님 결단 2차): Flask remote_queue 'liquidate' 비활성화.
+        # 사유: polling 소비자 0건 dead-letter queue.
+        # 대체: 텔레그램 "청산" (P1-A4 완료) 또는 `python main.py emergency-stop` (P2 완료).
         return {
-            'title': '💀 전량 청산 명령',
+            'title': '⛔ Flask 청산 명령 비활성화',
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

### 3-4. 도움말 메뉴 정리 (L946-953) — 3 라인 제거

```diff
@@ -946,12 +946,12 @@
             '🎛️ [제어]',
-            '  시작 — 자동매매 ON',
-            '  정지 — 자동매매 OFF',
             '  상태 — 시스템 현황',
-            '  청산 — 전량 청산 (위험!)',
             '',
-            '💡 텔레그램 봇에서도 동일 명령 사용 가능',
+            '💡 시작/정지/청산은 텔레그램 봇 또는 emergency-stop CLI에서만 가능',
+            '⚠️ Flask 웹 시작/정지/청산 라우트는 비활성화 상태',
         ]
         return {'title': '❓ 도움말', 'lines': lines}
```

---

## 4. 변경 영향 분석

### 4-1. 위험 차단 (3 라우트)
- ✅ `start`: KILL_SWITCH 무인 해제 경로 차단
- ✅ `stop`: KILL_SWITCH 무인 생성 경로 차단 (텔레그램 대체 경로 유지)
- ✅ `liquidate`: canonical 우회 전량 청산 경로 차단
- ✅ 도움말 메뉴에서 3 명령 모두 제거 → 사용자 혼동 방지
- ✅ 향후 polling 클라이언트가 추가되어도 큐 등록 자체 불가 → 3 명령 모두 활성화 방지

### 4-2. 영향 없음
- Flask `/api/remote/exec` POST endpoint 자체는 유지 (조회 명령 KEEP 23+ 건)
- Flask `/api/remote/queue` GET endpoint 자체는 유지 (큐가 비어있어도 동작)
- 텔레그램 `/시작` / `/정지` / `/청산` 경로 그대로 동작 (P1-A4 완료)
- `python main.py emergency-stop` CLI 그대로 동작 (P2 완료)
- KILL_SWITCH 파일 직접 조작 (`touch data/KILL_SWITCH` 등) 가능

### 4-3. 잔존 위험 (정직 명시)
- L494 `/api/remote/exec` POST에 인증 (`@api_key_required`) 적용된 상태 — 외부 무단 호출 차단 중
- 단, API key 유출 시 dead-letter 명령 큐 등록은 여전히 가능 → 본 diff 적용 후에는 큐 등록 자체 차단
- 잔존: `시작`/`정지`/`청산` 외 다른 조회 명령 (23+ 건)도 polling 소비자 추가 시 dead-letter 활성화 가능 — 단, 매매/제어 트리거 0건이라 위험 낮음 (별도 P3 검토)

---

## 5. 회귀 영향

### 5-1. 기존 회귀 영향 0건 (예상)
- 본 변경은 응답 문자열 + 큐 등록 코드만 수정
- 시그니처 변경 0건
- import 변경 0건
- 다른 파일 영향 0건

### 5-2. 신규 회귀 권장 (P3, B-8 적용 후)
- `tests/test_flask_queue_disabled.py` (선택):
  - 3 명령 (`시작`/`정지`/`청산`) 응답에 `'비활성화'` 또는 `'⛔'` 포함 확인
  - `remote_queue.json` 파일 또는 메모리 큐에 3 명령 등록 0건 확인

---

## 6. rollback 방법

### 6-1. commit 단위 rollback
```bash
git revert <flask-queue-disable-commit-hash>
# 효과: 3 라우트 + 도움말 3 라인 모두 원복
```

### 6-2. 부분 rollback (필요 시)
- 1 명령만 활성화 복구하고 싶을 경우:
  ```bash
  git checkout <pre-disable-commit> -- website/flask_app.py
  # 그 후 수동으로 원하는 라우트만 복구 + 다른 2개는 그대로 비활성화
  # → 별도 commit
  ```

### 6-3. rollback 후 안전 확인
- Flask 서버 재시작 + 응답 확인 (사장님 별도 승인 후)
- 회귀 25/25 + preflight 10/10 재확인
- canonical 우회 경로 차단 효과 확인 (`/api/remote/queue` GET 결과 빈 배열)

---

## 7. 적용 절차 (사장님 별도 승인 후)

```
[현재] 2차 diff 초안 작성 완료 (본 문서)
   ↓
[필수] Codex 회신 ①+② 도달
   ↓
[필수] 사장님 묶음 A/B/C commit 결단
   ↓
[필수] 사장님 Flask 큐 3건 비활성화 적용 별도 승인
   ↓
[적용] Edit으로 L789-815, L956-968, L946-953 변경 적용
   ↓
[필수] 회귀 25/25 + preflight 10/10 PASS 재확인
   ↓
[선택] 신규 회귀 `test_flask_queue_disabled.py` 작성 (P3)
   ↓
[필수] Flask 서버 재시작 (사장님 명시 승인 후만)
   ↓
[검증] /api/remote/exec POST {'cmd': '시작'} → "비활성화" 응답 확인 (curl 또는 수동)
[검증] /api/remote/exec POST {'cmd': '정지'} → "비활성화" 응답 확인
[검증] /api/remote/exec POST {'cmd': '청산'} → "비활성화" 응답 확인
[검증] data/remote_queue.json (있다면) 3 cmd 등록 0건 확인
```

---

## 8. 적용 금지 (본 2차 diff 작성 후)

- ❌ 본 diff 즉시 적용 X (Codex 회신 + 사장님 별도 승인 전까지)
- ❌ Flask 서버 재시작 X
- ❌ `data/remote_queue.json` 임시 삭제 X
- ❌ 도움말 메뉴 즉시 수정 X
- ❌ 1차 diff 초안 파일 삭제 X (이력 보존)
- ❌ 단독 commit X

---

## 9. 표현 룰

### 사용 가능
- "Flask queue 3건 비활성화 diff 2차 확장 완료"
- "1차 → 2차 확장 (liquidate 1건 → start/stop/liquidate 3건)"
- "polling 소비자 0건 dead-letter 3건 모두 격리 권장"
- "적용은 Codex 회신 + 사장님 별도 승인 후"

### 사용 금지
- "Flask queue 비활성화 완료" X (diff만 작성)
- "canonical 우회 경로 차단 완료" X (코드 미적용)
- "운영 안전 완성" X
- "재가동 가능" X

---

## 10. 연결 문서
- `docs/02-design/flask-liquidate-route-disable-diff-5_29.md` (1차 diff, liquidate만)
- `docs/02-design/flask_remote_queue_audit_5_29.md` (2차 전수 감사 — 본 diff의 근거)
- `docs/02-design/deletion-quarantine-audit-5_29.md` §3-1 (1차 audit)
- `docs/01-plan/v2-backlog-5_29.md` B-8, B-11
- `docs/02-design/quant_commit_bundle_plan_5_29.md` 묶음 E (적용 시 commit 후보)
- `website/flask_app.py:789-815, 956-968, 946-953` (변경 대상)
- `src/telegram_command_handler.py:620 (_execute_auto_start), L1203 (_cmd_stop), L589/L926 (_cmd_liquidate)` (대체 경로 — P1-A4 완료)
- `main.py:611-645 (cmd_emergency_stop)` (대체 경로 — P2 완료)
