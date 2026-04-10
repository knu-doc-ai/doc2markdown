import logging
import warnings

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# VARCO OCR 모델은 문서 파이프라인 전체에서 공유한다.
_SHARED_VARCO_PROCESSOR = None
_SHARED_VARCO_MODEL = None


def get_shared_varco_components():
    """파이프라인 전역에서 재사용할 VARCO processor/model을 반환한다."""
    global _SHARED_VARCO_PROCESSOR, _SHARED_VARCO_MODEL

    if _SHARED_VARCO_PROCESSOR is None or _SHARED_VARCO_MODEL is None:
        print("[SharedOCR] VARCO-VISION OCR 모델 로드 중...")
        varco_model_id = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
        _SHARED_VARCO_PROCESSOR = AutoProcessor.from_pretrained(varco_model_id)
        _SHARED_VARCO_MODEL = LlavaOnevisionForConditionalGeneration.from_pretrained(
            varco_model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto"
        )
        _SHARED_VARCO_MODEL.eval()
        print("[SharedOCR] VARCO 모델 로드 완료!\n")

    return _SHARED_VARCO_PROCESSOR, _SHARED_VARCO_MODEL
