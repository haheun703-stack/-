# [퀀트봇 → 단타봇] httpx 텔레그램 토큰 로그 마스킹 P1 (5/28)

> **작성일**: 2026-05-28 15:00 KST
> **계기**: 사용자 14:50 실측 보고 — 단타봇 journalctl에 Telegram bot URL이 토큰 포함 형태로 노출
> **우선순위**: P1 (보안, 비즈니스 차단 X)

## 1. 사건 요약

VPS `bodyhunter-bot.service` journalctl에 다음 패턴 노출 발견:

```
May 28 15:03:55 ip-172-26-2-140 python3.11[115302]: 15:03:55 [httpx] INFO: HTTP Request: POST https://api.telegram.org/bot<TOKEN_FULL>/getUpdates "HTTP/1.1 200 OK"
```

`<TOKEN_FULL>` 위치에 실제 텔레그램 봇 토큰이 그대로 출력됨.

## 2. 영향

- **위험**: journalctl 로그 접근 권한자가 봇 토큰 탈취 가능
- **공격 시나리오**:
  - VPS 침입 후 journalctl 읽기 → 봇 토큰 추출
  - 동일 봇으로 사용자 채팅에 가짜 매매 시그널 발송
  - 또는 봇 명령으로 자동매매 트리거 (단, 단타봇은 현재 manual_orders_allowed만)

## 3. 원인

- `httpx` 라이브러리 기본 INFO 로깅
- HTTP request URL을 INFO 레벨로 출력
- 텔레그램 API는 URL에 토큰 포함 (`https://api.telegram.org/bot{TOKEN}/method`)

## 4. Fix 패턴 (3가지 옵션)

### Option A: httpx 로거 INFO → WARNING 격하 (가장 단순)

```python
# bodyhunter 봇 진입점 (예: scalper-agent/main.py 상단)
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
```

- 장점: 1줄 fix
- 단점: httpx 다른 INFO 로그도 함께 사라짐 (디버깅 정보 손실)

### Option B: 로깅 Filter로 토큰만 마스킹

```python
import logging
import re

class TelegramTokenMaskFilter(logging.Filter):
    """텔레그램 봇 토큰 URL을 마스킹."""
    PATTERN = re.compile(r"bot\d{8,12}:[A-Za-z0-9_-]+")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.PATTERN.sub("bot<TOKEN_MASKED>", record.msg)
        if record.args:
            record.args = tuple(
                self.PATTERN.sub("bot<TOKEN_MASKED>", str(a)) if isinstance(a, str) else a
                for a in record.args
            )
        return True

# 등록
logging.getLogger("httpx").addFilter(TelegramTokenMaskFilter())
logging.getLogger("urllib3").addFilter(TelegramTokenMaskFilter())  # urllib3도 비슷한 패턴
```

- 장점: INFO 레벨 유지 + 토큰만 마스킹 (디버깅 정보 보존)
- 단점: 코드 추가 (~15줄)

### Option C: python-telegram-bot 측 옵션 (라이브러리 의존)

`python-telegram-bot` 사용 중이면 라이브러리 자체 설정 활용. (라이브러리 버전 확인 필요)

## 5. 권장 — Option B (Filter)

이유:
- INFO 레벨 정보 보존 (HTTP 200/400 등 디버깅 가치)
- 토큰만 정확히 마스킹
- 단타봇 외 다른 봇에서도 재사용 가능
- 퀀트봇은 `requests` 사용 + URL 미로깅이라 영향 X (참조용)

## 6. 적용 위치 (단타봇 측 검토 필요)

| 후보 | 확인 |
|------|------|
| `scalper-agent/main.py` 상단 | 봇 진입점 |
| `scalper-agent/bot/__init__.py` | 패키지 초기화 |
| `bodyhunter-bot.service` 환경변수 | systemd 측 추가 |
| `scripts/setup_logging.py` (있다면) | 공통 로깅 설정 |

## 7. 검증

Fix 후 확인 명령:

```bash
sudo journalctl -u bodyhunter-bot --since "5 minutes ago" --no-pager | grep -E 'api\.telegram\.org/bot[0-9]'
```

- **Fix 전**: `bot{실제토큰}/method` 노출
- **Fix 후**: `bot<TOKEN_MASKED>/method` 또는 INFO 자체 출력 X

## 8. 퀀트봇 측 점검 결과 (참조)

- `src/telegram_sender.py`: `requests` 라이브러리 사용
- HTTP URL 미로깅 (`logger.error("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다"`만)
- 결론: **퀀트봇 토큰 노출 0건** (단, 5/28 c7d53d6 시점)
- 향후 httpx/aiohttp 도입 시 동일 Filter 적용 권장

## 9. 일정 권장

- 5/28 (오늘): Option B 적용
- 5/29: VPS 배포 + 검증
- 토큰 강제 회전: 노출 확인됐으므로 단타봇 봇 토큰 재발급 권장 (사용자 결단)
- BotFather에서 `/revoke` 후 새 토큰 발급 + .env 갱신

## 10. 코덱스 검수 의뢰 (단타봇 측)

본 fix 적용 후 단타봇 측 코덱스 검수:
1. Filter 패턴 정확성 (`bot\d{8,12}:[A-Za-z0-9_-]+`)
2. urllib3 외 추가 라이브러리 점검
3. 토큰 회전 시점 결단
4. systemd journal 기존 로그 보존 정책 (rotation/cleanup)
