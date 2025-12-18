import os
import sys
import json  # ✅ LLM JSON 응답 파싱용

# 🛑 PyTorch 인덕터 / 다이너모 끄기 (cl 컴파일러 문제 방지 + 안정성)
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import time
import torch
import torchaudio
import sounddevice as sd
import numpy as np

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_DIR = r"C:\\Users\\ouner\\Desktop\\kiosk\\lee_kiosk"
MELO_PATH = os.path.join(BASE_DIR, "MeloTTS")

# 패키지 경로를 PYTHONPATH에 추가
if MELO_PATH not in sys.path:
    sys.path.append(MELO_PATH)

from melo.api import TTS  # ✅ Melo TTS

# ------------------------------------------------
# 0. 공통 설정
# ------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")

# ✅ 허용 옵션 상수 (LLM + 후처리에서 같이 사용)
ALLOWED_OPTIONS = {
    "아몬드밀크변경",
    "시나몬(O)",
    "시나몬(X)",
    "헤이즐넛시럽추가",
    "오트밀크변경",
    "샷 추가",
    "2샷 추가",
    "바닐라시럽추가",
    "카라멜시럽추가",
    "휘핑(O)",
    "휘핑(X)",
    "제로사이다",
    "스태비아 추가",
}


# ------------------------------------------------
# 1. 모델들 미리 로드 (프로그램 시작 시 1번만)
# ------------------------------------------------

print("🔊 Whisper 모델 로드 중 (small, fp16/int8)...")
whisper = WhisperModel(
    "small",
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8"  # CPU이면 int8로
)

print("🤖 EEVE LLM 로드 중...")
LLM_NAME = "yanolja/YanoljaNEXT-EEVE-Instruct-2.8B"
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

print("🎵 Melo TTS 로드 중...")
MELO_SPEED = 1.5  # 말하는 속도
MELO_DEVICE = "cuda:0" if DEVICE == "cuda" else "cpu"

melo_tts = TTS(language="KR", device=MELO_DEVICE)
melo_speaker_ids = melo_tts.hps.data.spk2id  # 예: melo_speaker_ids["KR"]

print("✅ 모든 모델 로드 완료\n")


# ------------------------------------------------
# 2. 유틸 함수들
# ------------------------------------------------

def record_to_file(filename: str, sec: float = 3.0, samplerate: int = 16000):
    """Enter 누르면 부르는 함수: 마이크 녹음 -> WAV 저장"""
    print("🎙 녹음 시작... (말하세요!)")
    audio = sd.rec(
        int(sec * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("🎙 녹음 종료")

    audio_np = np.squeeze(audio)
    audio_tensor = torch.tensor(audio_np).unsqueeze(0)  # [1, T]
    torchaudio.save(filename, audio_tensor, samplerate)


@torch.inference_mode()
def stt_whisper(audio_path: str) -> str:
    """Whisper로 음성 -> 텍스트"""
    t0 = time.time()
    segments, info = whisper.transcribe(
        audio_path,
        beam_size=1,               # 빠르게
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        without_timestamps=True,
        language="ko"              # 한국어 고정
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"⏱ Whisper 인식 시간: {time.time() - t0:.2f}s")
    return text


def extract_json_from_text(text: str) -> dict:
    """
    혹시라도 LLM이 다른 텍스트를 섞었을 경우,
    문자열 중에서 { ... } 부분만 뽑아서 JSON 파싱 시도.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        print(f"🔍 JSON 후보 추출:\n{candidate}\n")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            print(f"❗ 후보 JSON 파싱 실패: {e}")
            raise
    else:
        raise json.JSONDecodeError("No JSON object found", text, 0)


@torch.inference_mode()
def chat_with_eeve(user_text: str) -> dict:
    """
    EEVE LLM에게 JSON 형식으로

    {
      "items": [
        {
          "menu_name": "...",
          "menu_quantity": 2,
          "menu_option": "..."
        },
        ...
      ],
      "assistant_response": "..."
    }

    를 받은 뒤, 파이썬에서 menu_quantity만큼
    1잔짜리 item으로 분해해서 돌려준다.
    """

    # 🔹 시스템 프롬프트 (압축 버전 + "수량은 정수" 규칙)
    system_prompt = """
너는 카페 키오스크의 음성 주문 도우미이자 JSON 포맷터다.

항상 사용자의 발화를 분석해서 아래 구조의 JSON 객체 한 개만 출력한다.

{
  "items": [
    { "menu_name": 문자열, "menu_quantity": 정수, "menu_option": 문자열 },
    ...
  ],
  "assistant_response": 문자열
}

규칙:

1) items의 각 객체는 "한 종류의 메뉴"를 나타낸다.
   - menu_name : 음료 이름 (예: "아이스 아메리카노", "레몬차", "레몬 아이스티")
   - menu_quantity : 그 메뉴의 총 잔 수 (1, 2, 3 ...)
   - menu_option : 아래 허용 옵션 중 하나 또는 ""(빈 문자열)

   예: "레몬차 2잔 아이스 아메리카노 1잔" →
       items = [
         { "menu_name": "레몬차", "menu_quantity": 2, "menu_option": "" },
         { "menu_name": "아이스 아메리카노", "menu_quantity": 1, "menu_option": "" }
       ]

2) 여러 종류의 메뉴가 함께 말해진 경우,
   반드시 메뉴별로 item을 분리한다.

3) 수량 표현 예시:
   - "한 잔, 하나" → 1
   - "두 잔, 둘, 2잔" → 2
   - "세 잔, 셋, 3잔" → 3
   - "네 잔, 넷, 4잔" → 4
   수량이 명시되지 않은 메뉴는 1잔으로 간주한다.

4) 허용 옵션 (이 중 하나만 사용):
   - 아몬드밀크변경
   - 시나몬(O)
   - 시나몬(X)
   - 헤이즐넛시럽추가
   - 오트밀크변경
   - 샷 추가
   - 2샷 추가
   - 바닐라시럽추가
   - 카라멜시럽추가
   - 휘핑(O)
   - 휘핑(X)
   - 제로사이다
   - 스태비아 추가

   위 목록에 없는 단어는 절대로 menu_option에 넣지 말고,
   다른 메뉴 이름일 가능성이 있으면 새로운 item으로 분리한다.

   예: "레몬차 제로사이다 변경 아이스 아메리카노 한 잔" →
       - 레몬차: menu_option = "제로사이다"
       - 아이스 아메리카노: menu_option = ""

5) 시나몬/휘핑 규칙:
   - "넣어줘, 추가해줘, 넣어, 올려줘, 있게 해줘" ⇒ (O)
   - "빼줘, 빼고, 없이, 없게, 넣지 말아줘" ⇒ (X)

6) menu_option에는 위 허용 옵션 하나 또는 ""만 넣는다.
   허용되지 않는 옵션은 menu_option에는 넣지 말고,
   assistant_response에
   "죄송합니다, 말씀하신 옵션은 없는 옵션이므로 기본으로 주문해드리겠습니다."
   라는 문장을 포함시켜 알려 준다.

7) assistant_response:
   - 사용자의 표현을 자연스럽게 살려 주문 내용을 다시 말한다.
   - 여러 개 주문이면 "아이스 아메리카노 샷 추가 2잔과 레몬차 1잔"처럼 말한다.
   - 마지막은 반드시 "그대로 주문 도와드리겠습니다."로 끝낸다.
   - 존댓말, 한두 문장.

8) "듬뿍 넣어주세요", "많이 넣어주세요", "넉넉히 넣어주세요" 등은 사용하지 말고,
   내부적으로는 해당 옵션을 (O)로 판단하되,
   assistant_response에서는 "추가해서 그대로 주문 도와드리겠습니다." 정도로 표현한다.

9) 절대 JSON 바깥에 다른 문장, 설명, 마크다운, 예시를 출력하지 말고
   오직 JSON 한 개만 출력한다.
"""

    # 🔹 few-shot 예시 1: 단일 메뉴 + 옵션
    example_user1 = "아이스 아메리카노 샷 추가"
    example_assistant1 = """{
  "items": [
    {
      "menu_name": "아이스 아메리카노",
      "menu_quantity": 1,
      "menu_option": "샷 추가"
    }
  ],
  "assistant_response": "손님, 말씀하신 아이스 아메리카노 샷 추가 1잔 그대로 주문 도와드리겠습니다."
}"""

    # 🔹 few-shot 예시 2: 휘핑(X)
    example_user2 = "모카 하나 휘핑크림은 빼줘"
    example_assistant2 = """{
  "items": [
    {
      "menu_name": "모카",
      "menu_quantity": 1,
      "menu_option": "휘핑(X)"
    }
  ],
  "assistant_response": "손님, 말씀하신 모카 1잔, 휘핑크림은 빼서 그대로 주문 도와드리겠습니다."
}"""

    # 🔹 few-shot 예시 3: 제로사이다 변경 (단일 메뉴)
    example_user3 = "레몬차 제로사이다 변경"
    example_assistant3 = """{
  "items": [
    {
      "menu_name": "레몬차",
      "menu_quantity": 1,
      "menu_option": "제로사이다"
    }
  ],
  "assistant_response": "손님, 레몬차 1잔을 제로사이다로 변경하여 그대로 주문 도와드리겠습니다."
}"""

    # 🔹 few-shot 예시 4: 수량 (2잔)
    example_user4 = "아이스 아메리카노 샷 추가 2잔"
    example_assistant4 = """{
  "items": [
    {
      "menu_name": "아이스 아메리카노",
      "menu_quantity": 2,
      "menu_option": "샷 추가"
    }
  ],
  "assistant_response": "손님, 아이스 아메리카노 샷 추가 2잔 그대로 주문 도와드리겠습니다."
}"""

    # 🔹 few-shot 예시 5: 여러 메뉴 + 수량
    example_user5 = "레몬차 2잔 아이스 아메리카노 1잔"
    example_assistant5 = """{
  "items": [
    {
      "menu_name": "레몬차",
      "menu_quantity": 2,
      "menu_option": ""
    },
    {
      "menu_name": "아이스 아메리카노",
      "menu_quantity": 1,
      "menu_option": ""
    }
  ],
  "assistant_response": "손님, 레몬차 2잔과 아이스 아메리카노 1잔 그대로 주문 도와드리겠습니다."
}"""

    # 🔹 실제 프롬프트 구성 (EEVE 채팅 템플릿 사용)
    full_prompt = (
        "<|im_start|>system\n" + system_prompt.strip() + "<|im_end|>\n"
        "<|im_start|>user\n" + example_user1 + "<|im_end|>\n"
        "<|im_start|>assistant\n" + example_assistant1 + "<|im_end|>\n"
        "<|im_start|>user\n" + example_user2 + "<|im_end|>\n"
        "<|im_start|>assistant\n" + example_assistant2 + "<|im_end|>\n"
        "<|im_start|>user\n" + example_user3 + "<|im_end|>\n"
        "<|im_start|>assistant\n" + example_assistant3 + "<|im_end|>\n"
        "<|im_start|>user\n" + example_user4 + "<|im_end|>\n"
        "<|im_start|>assistant\n" + example_assistant4 + "<|im_end|>\n"
        "<|im_start|>user\n" + example_user5 + "<|im_end|>\n"
        "<|im_start|>assistant\n" + example_assistant5 + "<|im_end|>\n"
        "<|im_start|>user\n" + user_text.strip() + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    t0 = time.time()
    inputs = tokenizer(full_prompt, return_tensors="pt").to(llm.device)

    # 🔧 결정론적으로 생성 (JSON 구조 유지 유도) + ⏱ max_new_tokens 120으로 더 축소
    output = llm.generate(
        **inputs,
        max_new_tokens=120,          # ✅ 160 → 120 (속도 더 줄임)
        do_sample=False,             # 샘플링 비활성화
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output[0][inputs["input_ids"].shape[1]:]
    answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print(f"⏱ LLM 응답 시간: {time.time() - t0:.2f}s")
    print(f"🤖 LLM 원문 응답:\n{answer_text}\n")

    # JSON 파싱 (직접 시도 + {..} 부분만 추출 시도)
    try:
        data = json.loads(answer_text)
    except json.JSONDecodeError:
        print("❗ 전체 텍스트 JSON 파싱 실패, 중간에서 { ... } 부분만 추출 시도합니다.")
        try:
            data = extract_json_from_text(answer_text)
        except json.JSONDecodeError:
            print("❗ 최종 JSON 파싱 실패, fallback 사용.")
            data = {
                "items": [
                    {
                        "menu_name": "",
                        "menu_quantity": 1,
                        "menu_option": ""
                    }
                ],
                "assistant_response": "죄송합니다, 주문 인식에 실패했습니다. 다시 한번 말씀해주시겠어요? 그대로 주문 도와드리겠습니다."
            }

    # -------- 후처리 공통 --------
    items = data.get("items", [])
    if not isinstance(items, list):
        items = []

    # 1차: 각 item 보정 (수량 정수화, 옵션 문자열 보정) + "1잔 단위 리스트"로 분해
    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue

        name = str(item.get("menu_name", "")).strip()
        option = str(item.get("menu_option", "")).strip()
        qty = item.get("menu_quantity", 1)

        # menu_quantity 정리 (정수 변환)
        try:
            qty = int(qty)
        except (TypeError, ValueError):
            qty = 1
        if qty <= 0:
            qty = 1

        # 옵션: 허용되지 않은 옵션이면 강제로 "" 처리
        if option not in ALLOWED_OPTIONS:
            option = ""

        # qty 만큼 item 복제 (각각 menu_quantity=1)
        for _ in range(qty):
            normalized_items.append({
                "menu_name": name,
                "menu_quantity": 1,
                "menu_option": option,
            })

    # 🔧 제로사이다 옵션 후처리 (특정 음료 제한 X, 일반 규칙)
    text_no_space = user_text.replace(" ", "")
    if "제로사이다" in text_no_space:
        has_zero = any(it.get("menu_option") == "제로사이다" for it in normalized_items)
        if not has_zero and normalized_items:
            # LLM이 아무 item에도 제로사이다를 안 넣었으면,
            # 일단 첫 번째 item의 빈 옵션에 제로사이다를 붙여준다.
            if not normalized_items[0].get("menu_option"):
                normalized_items[0]["menu_option"] = "제로사이다"

    # assistant_response 후처리
    assistant_response = data.get("assistant_response", "")

    # "듬뿍", "많이", "넉넉히" 제거 + 끝에 문장 정리
    if any(x in assistant_response for x in ["듬뿍", "많이", "넉넉히"]):
        for x in ["듬뿍", "많이", "넉넉히"]:
            assistant_response = assistant_response.replace(x, "")
        if not assistant_response.strip().endswith("그대로 주문 도와드리겠습니다."):
            if not assistant_response.strip().endswith("다.") and not assistant_response.strip().endswith("요."):
                assistant_response = assistant_response.strip() + " "
            assistant_response = assistant_response.strip() + " 그대로 주문 도와드리겠습니다."

    # "시나몬은" → "시나몬 추가하여" 같은 식으로 자연스럽게
    if "시나몬은" in assistant_response:
        assistant_response = assistant_response.replace("시나몬은", "시나몬 추가하여")

    # 제로사이다 표현 보정
    if "제로사이다변경" in assistant_response:
        assistant_response = assistant_response.replace("제로사이다변경", "제로사이다")
    if "제로 사이다" in assistant_response:
        assistant_response = assistant_response.replace("제로 사이다", "제로사이다")

    # assistant_response가 비어있거나 "그대로 주문..."으로 안 끝나면 강제로 끝 문장 추가
    if not assistant_response.strip():
        assistant_response = "주문해 주셔서 감사합니다. 그대로 주문 도와드리겠습니다."
    elif not assistant_response.strip().endswith("그대로 주문 도와드리겠습니다."):
        assistant_response = assistant_response.strip() + " 그대로 주문 도와드리겠습니다."

    data["items"] = normalized_items
    data["assistant_response"] = assistant_response

    return data


@torch.inference_mode()
def speak_with_melo(text: str, out_path: str, speed: float = MELO_SPEED):
    """Melo TTS로 한국어 문장을 음성(WAV)으로 저장"""
    # 너무 길면 앞부분만 TTS (속도 절약)
    if len(text) > 80:
        print("🔁 TTS 텍스트가 길어서 앞 80자만 읽습니다.")
        text = text[:80]

    t0 = time.time()
    speaker_id = melo_speaker_ids["KR"]  # 기본 한국어 화자

    melo_tts.tts_to_file(
        text,
        speaker_id,
        out_path,
        speed=speed,
    )

    print(f"⏱ Melo TTS 시간: {time.time() - t0:.2f}s")


# ------------------------------------------------
# 3. 메인 루프 (단독 실행용 데모)
# ------------------------------------------------

if __name__ == "__main__":
    print("==============================")
    print("🎤 음성 대화형 키오스크 데모 (Whisper + EEVE + Melo TTS)")
    print("Enter 키를 누르면 녹음합니다. (약 5초)")
    print("종료하려면 Ctrl+C")
    print("==============================\n")

    try:
        while True:
            input("➡ Enter 를 누르면 녹음을 시작합니다...")

            # 1) 녹음
            wav_path = os.path.join(BASE_DIR, "input.wav")
            record_to_file(wav_path, sec=5.0)

            # 2) STT
            user_text = stt_whisper(wav_path)
            print(f"📝 Whisper 인식 결과: {user_text}")

            if not user_text:
                print("❗ 인식된 내용이 없습니다. 다시 시도하세요.\n")
                continue

            # 3) LLM 응답 (JSON)
            result = chat_with_eeve(user_text)

            items = result.get("items", [])
            assistant_response = result.get("assistant_response", "")

            print("📦 파싱된 주문 정보:")
            if not items:
                print("  (items 비어 있음)")
            else:
                for i, it in enumerate(items, start=1):
                    mn = it.get("menu_name", "")
                    mq = it.get("menu_quantity", 1)
                    mo = it.get("menu_option", "")
                    print(f"  - item #{i}")
                    print(f"      menu_name     : {mn}")
                    print(f"      menu_quantity : {mq}")
                    print(f"      menu_option   : {mo}")
            print(f"  - assistant_resp: {assistant_response}")

            # TODO: 여기서 items를 실제 키오스크 주문 로직과 연결하면 됨.
            #       (각 item마다 menu_name / menu_option / 1잔)

            # 4) TTS (Melo) → assistant_response만 읽기
            out_wav = os.path.join(BASE_DIR, "response.wav")
            speak_with_melo(assistant_response, out_wav)
            print(f"📁 음성 파일 저장 완료: {out_wav}\n")

    except KeyboardInterrupt:
        print("\n👋 종료합니다.")
