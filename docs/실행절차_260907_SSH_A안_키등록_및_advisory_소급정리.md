# 실행 절차 — SSH A안 키 등록 · advisory 소급 정리 (2026-09-07 승인분)

퐝가님이 승인하신 4건 중 **2건은 제가 실행했고**(내재가치 엔진 상태 표시, 봇 회신 발신), **2건은 제 도구 정책이 원격 상태 변경을 막아 실행하지 못했습니다.** 우회하지 않고 바로 실행하실 수 있게 정리합니다. 둘 다 명령 한 줄씩입니다.

---

## 1. advisory 소급 정리 — VPS에서 한 줄

스크립트는 저장소에 넣어 배포했습니다(`scripts/backfill_advisory_contract.py`). **기본이 dry-run**이라 `--apply` 없이는 아무것도 쓰지 않습니다.

```bash
ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" ubuntu@13.209.153.221
cd ~/quantum-master

# ① 먼저 조회만 (쓰지 않음) — 대상 1,352행이 나오면 정상
./venv/bin/python3.11 -u -X utf8 scripts/backfill_advisory_contract.py

# ② 실제 정리
./venv/bin/python3.11 -u -X utf8 scripts/backfill_advisory_contract.py --apply
```

**무엇을 하나**

| 대상 | 처리 |
|---|---|
| `reasoning` | 허용키 9종만 남기고 나머지 18종 제거(ETF 추천·픽 목록·top5 6종 등) |
| `related_tickers` | 빈 배열 |
| `title`·`body` | **금지 어휘가 있는 것만** 최소 문구로 교체 |
| 행 자체 | **지우지 않음** — 수신 이력은 남긴다 |

**범위**: 계약 시행일(7/27) 이후, 우리 생산분, 9/7 11:25 배포 이전 = **1,352행**.
계약 이전(5/18~7/26, 약 2,160행)은 당시 규칙상 정상이라 손대지 않습니다. 타 봇 적재분 2행도 제외입니다.

**안전장치**: 실행 전 전량을 `data/advisory_backfill_backup_20260907.json`에 백업하고, 단일 트랜잭션이라 실패 시 롤백되며, 끝나면 같은 판정으로 사후 검증해 잔존 위반 0을 확인합니다. 단타봇이 응답·평가를 기록한 행은 **0건**이라 덮어써서 잃는 것이 없습니다(사전 조회 확인).

---

## 2. SSH A안 — 퀀트봇 전용 키 등록

키쌍은 만들어 두었습니다. **저장소 밖**에 있습니다.

| 항목 | 값 |
|---|---|
| 개인키 | `D:/Prophet_Agent_System_예언자/_tmp_zips/quantbot_ed25519` |
| 공개키 | `D:/Prophet_Agent_System_예언자/_tmp_zips/quantbot_ed25519.pub` |
| 지문 | `SHA256:OZg4TNqY1+6gXkOYDeI5+/hLAGPMIZZuE3ENgdJI3ZQ` (ED25519) |
| 주석 | `quant-bot@flowx-2026-09-07` |

**서버 현황**(오늘 실측): `~/.ssh/authorized_keys`에 키가 **1개뿐**이고 그것이 `LightsailDefaultKeyPair`(RSA 2048, 지문 `SHA256:Ndcd…SM5Q`)입니다. 퀀트봇·단타봇·정보봇이 전부 이 하나를 쓰고 있습니다. 웹봇이 말한 "공유키"가 사실은 **인스턴스 기본 키페어**였습니다.

**등록 — 추가만 하므로 아무도 잠기지 않습니다**

```bash
ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/lightsail_60gb.pem" ubuntu@13.209.153.221

# ① 백업
cp ~/.ssh/authorized_keys ~/.ssh/authorized_keys.bak_20260907

# ② 퀀트봇 공개키 추가 (아래 한 줄을 그대로)
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBv/CXMXxM+gzZ9ky0tuNTEOW4c1evgUf2X+PKyO/XGd quant-bot@flowx-2026-09-07' >> ~/.ssh/authorized_keys

# ③ 확인 (2개가 나와야 정상)
ssh-keygen -lf ~/.ssh/authorized_keys
exit
```

**③ 검증 — 기존 세션을 닫지 말고 새 창에서**

```bash
ssh -i "D:/Prophet_Agent_System_예언자/_tmp_zips/quantbot_ed25519" ubuntu@13.209.153.221 'hostname'
# ip-172-26-2-140 이 나오면 성공
```

성공하면 알려주십시오. 제가 `CLAUDE.md`의 접속 경로를 새 키로 바꾸고 원장 1줄을 남기겠습니다.

★ **기존 키는 지우지 마십시오.** 단타봇·정보봇까지 전용 키로 전환하고 넷 다 검증한 뒤가 제거 시점입니다. 지금 지우면 다른 봇이 잠깁니다.

⚠️ Windows에서 `chmod 600`이 NTFS에 반영되지 않아 SSH가 개인키 권한을 거부할 수 있습니다. `UNPROTECTED PRIVATE KEY FILE` 오류가 나면 파일 속성 → 보안에서 본인 계정 외 권한을 제거하시면 됩니다.

---

## 3. 왜 제가 못 했는지

두 작업 모두 **원격 서버의 상태를 바꾸는 것**이라 제 도구 정책이 차단했습니다. DB의 1,352행 UPDATE와 SSH 인증 파일 수정입니다. 차단을 우회하는 방법은 있었지만 쓰지 않았습니다 — 그런 우회는 승인의 의미를 없앱니다. 대신 실행 가능한 형태로 만들어 두었고, 스크립트는 저장소에 있어 언제든 다시 쓸 수 있습니다.

원하시면 다음 세션에서 권한을 열어 주시거나(`/config`의 Bash 권한 규칙), 위 두 줄을 직접 실행해 주시면 됩니다.
