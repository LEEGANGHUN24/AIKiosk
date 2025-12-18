from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "yanolja/YanoljaNEXT-EEVE-Instruct-2.8B"

# 1) 토크나이저 / 모델 로드
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,   # 모델 쪽에서 커스텀 코드 쓰는 경우를 대비
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # RTX 40시리즈 BF16 잘됨
    device_map="auto",           # 자동으로 GPU/CPU 배치
    trust_remote_code=True,
)

# 2) 프롬프트 템플릿
prompt_template = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: {prompt}\n"
    "Assistant:\n"
)

# 3) 실제 질문 (한국어)
user_question = "커피란 무엇인지 한국어로 자세히 설명해줘."

full_prompt = prompt_template.format(prompt=user_question)

inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

# 4) 생성
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,      # 샘플링 (랜덤성 약간)
        temperature=0.7,
        top_p=0.9,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5) 전체 텍스트에서 Assistant 부분만 잘라내기
if "Assistant:" in generated:
    answer = generated.split("Assistant:")[-1].strip()
else:
    # 혹시 템플릿이 조금 다르게 동작한 경우 대비
    answer = generated

print("=== 모델 전체 출력 ===")
print(generated)
print("\n=== Assistant 답변만 ===")
print(answer)
