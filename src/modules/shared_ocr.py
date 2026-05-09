import gc
import logging
import warnings

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

_SHARED_VARCO_PROCESSOR = None
_SHARED_VARCO_MODEL = None


def get_shared_varco_components():
    """직접 실행 모드에서 재사용할 VARCO 모델을 반환한다."""
    global _SHARED_VARCO_PROCESSOR, _SHARED_VARCO_MODEL

    if _SHARED_VARCO_PROCESSOR is None or _SHARED_VARCO_MODEL is None:
        print("[SharedOCR] VARCO-VISION OCR 모델 로드 중...")
        varco_model_id = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
        _SHARED_VARCO_PROCESSOR = AutoProcessor.from_pretrained(varco_model_id)
        _SHARED_VARCO_MODEL = LlavaOnevisionForConditionalGeneration.from_pretrained(
            varco_model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        _SHARED_VARCO_MODEL.eval()
        print("[SharedOCR] VARCO 모델 로드 완료!\n")

    return _SHARED_VARCO_PROCESSOR, _SHARED_VARCO_MODEL


def release_shared_varco_components() -> None:
    """직접 실행 모드에서 재사용한 VARCO 자원을 해제한다."""
    global _SHARED_VARCO_PROCESSOR, _SHARED_VARCO_MODEL

    _SHARED_VARCO_PROCESSOR = None
    _SHARED_VARCO_MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[SharedOCR] VARCO 메모리 해제 완료")
