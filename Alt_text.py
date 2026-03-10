import os
import uuid
import base64
import ollama
from openai import OpenAI

# Llama-3.2-Vision을 사용하는 방식 (무료임)
# ollama 설치(1.2G) 후 ollama run llama3.2-vision 을 'cmd'에 입력하여 모델 다운(7.8G) 필요 

client = OpenAI(base_url='http://localhost:11434/v1',
                api_key='ollama',) # 로컬 Ollama 서버 주소로 설정

# 표나 이미지를 받을 때 이미지파일로 받는다
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def process_image_with_alt_text(crop_image_path, save_dir="./images"):
    # 이미지를 저장하고 내용 설명(Alt-text)이 포함된 마크다운 태그를 생성
    # 파일 저장 및 경로 생성 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    import shutil
    unique_name = f"img_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join(save_dir, unique_name)
    shutil.copy(crop_image_path, save_path)
    
    # Alt-text 생성
    base64_image = encode_image(save_path)
    prompt = "Write a one-line concise alt-text (maximum 10 words) for this image. Output ONLY the description text without any labels or bullet points."

    response = client.chat.completions.create(
        model="llama3.2-vision",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ],
        }],extra_body={"keep_alive": 0}
    )
    alt_text = response.choices[0].message.content.strip()
    # 마크다운 이미지 태그 반환
    return f"![{alt_text}]({save_path})\n"

# 실행 예시
import time
start_time = time.time()  # 시작 시간 기록
end_time = time.time()    # 종료 시간 기록
image_md = process_image_with_alt_text("test_image.png")
print(image_md)
print(f"소요 시간: {end_time - start_time:.2f}초")