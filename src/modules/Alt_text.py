import os
import uuid
import base64
import ollama
import requests
from openai import OpenAI

MODEL_NAME = "gemma4:e4b"

# 성능이 좋지 않은 llama-vision을 버리고, gemma4:e4b를 사용하는 버전.
def ensure_model_exists(model_name):
    try:
        ollama.show(model_name)
    except Exception:
        print(f"[{model_name}] 모델 다운로드를 시작합니다...")
        ollama.pull(model_name)

ensure_model_exists(MODEL_NAME)
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

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
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ],
        }],
        extra_body={"keep_alive": "5m"} 
    )
    alt_text = response.choices[0].message.content.strip()
    return f"![{alt_text}]({save_path})\n"
