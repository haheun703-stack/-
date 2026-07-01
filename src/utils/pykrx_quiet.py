"""pykrx 1.2.7 로깅 노이즈 억제 유틸.

pykrx/website/comm/util.py 의 ``dataframe_empty_handler`` 는 KRX가 빈 응답을
반환할 때(예: IP 차단 / 자격증명 미설정) 예외를 잡아 빈 DataFrame을 돌려준다.
그 직전 아래 라인을 호출한다::

    logging.info(args, kwargs)   # args=함수 위치인자 튜플, kwargs=키워드인자 dict

``logging.info`` 의 첫 인자는 포맷 메시지여야 하는데 여기서는 위치인자 튜플이
그 자리에 들어간다. root logger가 이 레코드를 emit 할 때
``msg % self.args`` 가 ``TypeError: not all arguments converted during
string formatting`` 를 일으키고 stderr에 ``--- Logging error ---`` 트레이스백을
남긴다(logging 모듈이 자체 처리하므로 프로그램은 죽지 않는다).

KRX 잠금(2026-06-22~) 동안 pykrx 호출마다 이 노이즈가 종목 수만큼 누적된다.
(2026-07-01 BAT-D 로그: scan_nugget 32 / scan_fibonacci 24 / leverage_etf 4건)

venv 라이브러리 직접 패치는 pip 재설치 시 원복되고 git 자동배포가 불가하므로,
우리 코드에서 root logger에 필터를 달아 해당 레코드만 걸러낸다. pykrx의 이
버그 호출만 ``record.msg`` 가 튜플이고, 정상 로그는 문자열이므로 영향이 없다.
또한 자식 로거(getLogger(__name__))로 남기는 정상 로그는 root logger의 필터를
거치지 않으므로 이중으로 안전하다.
"""

from __future__ import annotations

import logging

_FLAG = "_pykrx_noise_filter_installed"


class PykrxNoiseFilter(logging.Filter):
    """pykrx ``logging.info(args, kwargs)`` 버그 레코드(msg가 튜플)를 차단."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return not isinstance(record.msg, tuple)


def silence_pykrx_logging() -> None:
    """root logger에 노이즈 필터를 1회만 설치한다(idempotent)."""
    root = logging.getLogger()
    if getattr(root, _FLAG, False):
        return
    root.addFilter(PykrxNoiseFilter())
    setattr(root, _FLAG, True)
