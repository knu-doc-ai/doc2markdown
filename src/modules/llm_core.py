class ContentEnricher:
    """
    레이아웃 기반 텍스트를 LLM에 전달하여 형식을 가다듬거나, Alt-text 등을 생성하는 Enricher 클래스입니다.
    현재 팀원이 구현 중이므로 임시로 빈 모의(Mock) 클래스를 둡니다.
    """
    def __init__(self):
        pass

    def enrich(self, layout_elements, table_results=None, config=None):
        # 현재는 변환 없이 그대로 반환
        return layout_elements
