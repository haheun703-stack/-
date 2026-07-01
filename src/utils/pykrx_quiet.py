"""pykrx 로깅/print 노이즈 억제 유틸.

이 프로젝트의 pykrx는 KRX 로그인 기능이 얹힌 커스텀 빌드다. KRX를 실제로는
쓰지 않지만(OHLCV=FDR / 수급=KIS) pykrx가 두 갈래로 노이즈를 뿜는다.

1) pykrx/website/comm/util.py 의 ``dataframe_empty_handler`` 는 KRX 빈 응답 시
   예외를 잡아 빈 DataFrame을 돌려주기 직전 ``logging.info(args, kwargs)`` 를
   호출한다. 위치인자 튜플이 포맷 메시지 자리에 들어가 root logger가 이를
   emit 할 때 ``TypeError: not all arguments converted`` 트레이스백을 낸다.

2) pykrx/website/comm/auth.py 는 import 시점과 조회 시마다 KRX 자동 로그인을
   시도하며 실패 메시지를 전부 ``print`` 로 뿜는다("KRX 로그인 시도...",
   "자격 증명을 확인하세요." 등). logging이 아니라 print라 필터로는 못 막는다.

두 노이즈 모두 기능과 무관하다(예외는 각 스캐너의 try/except가 처리, KRX 결과는
버려지고 FDR/KIS로 대체됨). venv 라이브러리 직접 패치는 pip 재설치 시 원복되고
git 자동배포가 불가하므로, 진입 스크립트에서 이 유틸을 1회 호출해 잡는다.

- (1)은 root logger에 "msg가 튜플인 레코드"만 걸러내는 필터로 차단.
- (2)는 pykrx.auth 모듈을 미리 로드하되 그동안만 builtins.print를 억제하고,
  이후 auth 모듈의 print 를 영구 no-op으로 덮는다. 스캐너들이 pykrx를 함수
  내부에서 지연 import 하므로, 모듈 로드 시점의 이 선점이 유효하다.
"""

from __future__ import annotations

import logging

_FLAG = "_pykrx_noise_filter_installed"
_AUTH_FLAG = "_pykrx_auth_print_silenced"


class PykrxNoiseFilter(logging.Filter):
    """pykrx의 root 직접 로깅 노이즈를 차단.

    - util.py:19 ``logging.info(args, kwargs)`` → msg가 튜플(TypeError 유발)
    - util.py:20 ``logging.info(e)`` → msg가 예외 객체
    둘 다 root logger 직접 호출이라, 자식 logger(getLogger(__name__))로 남기는
    우리 정상 로그(exception 포함)에는 영향이 없다.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return not isinstance(record.msg, (tuple, BaseException))


def _silence_logging() -> None:
    root = logging.getLogger()
    if getattr(root, _FLAG, False):
        return
    root.addFilter(PykrxNoiseFilter())
    setattr(root, _FLAG, True)


def _silence_auth_prints() -> None:
    """pykrx.auth 의 KRX 로그인 print 노이즈를 무력화(idempotent)."""
    import builtins

    _orig_print = builtins.print
    builtins.print = lambda *a, **k: None  # auth import 중 노이즈 선점 억제
    try:
        import pykrx.website.comm.auth as _auth

        if getattr(_auth, _AUTH_FLAG, False):
            return
        _auth.print = lambda *a, **k: None  # 이후 조회 시 노이즈 영구 억제
        setattr(_auth, _AUTH_FLAG, True)
    except Exception:
        # pykrx 미설치/구조 변경 등은 조용히 무시(기능 무관)
        pass
    finally:
        builtins.print = _orig_print


def silence_pykrx_logging() -> None:
    """pykrx의 logging TypeError 노이즈와 KRX 로그인 print 노이즈를 모두 억제.

    진입 스크립트에서 pykrx 를 (지연) import 하기 전에 1회 호출한다.
    """
    _silence_logging()
    _silence_auth_prints()
