# 기존 import 경로 호환성을 유지하기 위한 facade 모듈
# 실제 구현은 modules.assembly 패키지 아래로 분리되어 있다.
from modules.assembly import *  # noqa: F401,F403
from modules.assembly import __all__
