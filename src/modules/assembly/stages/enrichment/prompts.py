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
        '출력 형식: {"semantic_decisions":[{"id":"...","kind":"text|heading|caption|note","heading_level":null,"confidence":0.0}],"caption_links":[{"caption_id":"...","target_id":"...","confidence":0.0}]}\n'
        "semantic_decisions에는 kind 또는 heading_level을 바꿀 필요가 있는 후보만 포함.\n"
        "제목 판단 기준:\n"
        "- 1., 1.2, 1.2.3 같은 번호 제목은 heading 후보.\n"
        "- 번호 depth 기준 heading_level: 1.은 1, 1.2는 2, 1.2.3은 3.\n"
        "- 입력 candidate의 semantic_hint와 semantic_reason은 휴리스틱 참고값이며, 최종 판단은 text 기준.\n"
        "- 짧고 명사구에 가까운 줄은 제목일 수 있음.\n"
        "- 마침표로 끝나는 긴 설명문, 완전한 본문 문장은 text 유지.\n"
        "- 표/그림 설명처럼 '표 1', 'Figure 2', '그림 3'으로 시작하면 caption 후보.\n"
        "- 확신이 낮으면 해당 후보는 응답에서 제외.\n"
        "예시:\n"
        "- 1.2 범위 및 제약사항 -> kind=heading, heading_level=2\n"
        "- 3.1 테일러 급수 전개 (공학 함수 근사) -> kind=heading, heading_level=2\n"
        "- 5. 비기능 요구사항 (Non-Functional Requirements) -> kind=heading, heading_level=1\n"
        "- 본 문서는 웹 기반 계산기 애플리케이션의 요구사항을 정의합니다. -> kind=text, 응답 제외\n"
        f"입력 JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

def build_content_repair_prompt(payload: dict[str, Any]) -> str:
    return (
        "작업: content_repair\n"
        "역할: OCR/PDF 줄바꿈 때문에 한국어 단어 내부에 삽입된 잘못된 공백 보정.\n"
        "출력은 반드시 JSON 객체 하나만 반환.\n"
        '출력 형식: {"repairs":[{"node_id":"...","text":"...","confidence":0.0}]}\n'
        "모든 입력 item에 대해 repairs 항목 하나를 반환.\n"
        "공백만 변경. 비공백 문자 시퀀스는 원문과 완전히 동일해야 함.\n"
        "정상 띄어쓰기, 영어, 숫자, 기호, 괄호, URL, 코드처럼 보이는 조각은 보존.\n"
        "확신이 낮으면 원문 text를 그대로 반환하고 confidence를 0.5로 설정.\n"
        "예시:\n"
        "- 기 능적 -> 기능적\n"
        "- 작 성되었습니다 -> 작성되었습니다\n"
        "- 준 수하여 -> 준수하여\n"
        "- 경 험 -> 경험\n"
        f"입력 JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
