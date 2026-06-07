from __future__ import annotations

import json
from typing import Any


def build_prompt(task: str, payload: dict[str, Any]) -> str:
    if task == "semantic_enrichment":
        return build_semantic_enrichment_prompt(payload)
    if task == "content_repair":
        return build_content_repair_prompt(payload)

    return (
        f"작업: {task}\n"
        "요청 schema 정확히 준수. 원문 의미 보존.\n"
        f"입력 JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

def build_semantic_enrichment_prompt(payload: dict[str, Any]) -> str:
    return (
        "작업: semantic_enrichment\n"
        "역할: OCR/PDF layout 결과에서 제목, 본문, 캡션, 주석 후보를 재분류.\n"
        "출력은 반드시 JSON 객체 하나만 반환.\n"
        '출력 형식: {"semantic_decisions":[{"id":"...","kind":"text|heading|caption|note","heading_level":null,"confidence":0.9}],"caption_links":[{"caption_id":"...","target_id":"...","confidence":0.9}]}\n'
        "semantic_decisions에는 kind 또는 heading_level을 바꿀 필요가 있는 후보만 포함.\n"
        "제목 판단 기준:\n"
        "- 1., 1.2, 1.2.3 같은 번호 제목은 heading 후보.\n"
        "- 번호 depth 기준 heading_level: 1.은 1, 1.2는 2, 1.2.3은 3.\n"
        "- 입력 candidate의 semantic_hint와 semantic_reason은 휴리스틱 참고값이며, 최종 판단은 text 기준.\n"
        "- 짧고 명사구에 가까운 줄은 제목일 수 있음.\n"
        "- 마침표로 끝나는 긴 설명문, 완전한 본문 문장은 text 유지.\n"
        "- 표/그림 설명처럼 '표 1', 'Figure 2', '그림 3'으로 시작하면 caption 후보.\n"
        "- confidence는 0.5 이상 1.0 이하로 반환. 확신이 낮으면 해당 후보는 응답에서 제외.\n"
        "예시:\n"
        "- 1.2 조사 범위 -> kind=heading, heading_level=2\n"
        "- 3.1 주요 결과 요약 -> kind=heading, heading_level=2\n"
        "- 5. 결론 및 향후 과제 -> kind=heading, heading_level=1\n"
        "- 본 문서는 주요 내용을 설명합니다. -> kind=text, 응답 제외\n"
        f"입력 JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

def build_content_repair_prompt(payload: dict[str, Any]) -> str:
    return (
        "작업: content_repair\n"
        "역할: OCR/PDF 줄바꿈 병합 때문에 한국어 단어 내부에 삽입된 잘못된 공백 보정.\n"
        "출력은 반드시 JSON 객체 하나만 반환.\n"
        '출력 형식: {"repairs":[{"node_id":"...","text":"..."}]}\n'
        "모든 입력 item에 대해 repairs 항목 하나를 반환.\n"
        "공백만 추가/삭제. 비공백 문자 시퀀스는 원문과 완전히 동일해야 함.\n"
        "각 item의 non_space_signature는 원문 text에서 공백을 제거한 기준 문자열이며, 반환 text에서 공백을 제거한 값이 반드시 이 문자열과 정확히 같아야 함.\n"
        "content_repair에서는 confidence를 사용하지 않으므로 반환하지 않음.\n"
        "오타, 단어, 조사, 숫자, 영어, 기호, 괄호, 콜론, 마침표는 수정하지 말 것. 닫는 괄호나 문장부호를 삭제하지 말 것.\n"
        "문단 전체의 정상 띄어쓰기를 제거하지 말 것. 여러 어절 공백을 한꺼번에 없애는 출력은 잘못된 출력임.\n"
        "정상 어절 사이 공백, 영어, 숫자, 기호, 괄호, URL, 코드처럼 보이는 조각은 보존.\n"
        "의심스러운 한국어 내부 공백은 제거하고, 의심스러운 내부 공백이 없으면 원문 text를 그대로 반환.\n"
        "spacing_suspect=true인 짧은 item은 한글 단어 내부가 잘렸을 가능성이 높으므로 특히 확인하되, 고정 단어 목록에 의존하지 말고 문맥과 자연스러운 한국어 어절 기준으로 판단.\n"
        "예시(특정 문서 단어가 아니라 OCR 내부 공백 패턴 예시이며, 단어를 새로 만들지 말고 공백만 조정):\n"
        "- 작 성되었습니다 -> 작성되었습니다\n"
        "- 경 험을 -> 경험을\n"
        "- 내 용을 -> 내용을\n"
        "- 문 서를 -> 문서를\n"
        "- 검 토합니다 -> 검토합니다\n"
        "- 자료 분석 -> 자료 분석\n"
        "- 검토 의견 -> 검토 의견\n"
        f"입력 JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
