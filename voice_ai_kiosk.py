# voice_ai_kiosk.py
import os
import sys
import time
import json
import csv
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional, Any

from dotenv import load_dotenv
from openai import OpenAI

# 🛑 PyTorch 인덕터 / 다이너모 끄기 (cl 컴파일러 문제 방지 + 안정성)
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
import torchaudio
import sounddevice as sd
import numpy as np

from faster_whisper import WhisperModel

# ====== 기존 코드의 BASE_DIR / PROJECT_DIR 유지 ======
BASE_DIR = r"C:\\Users\\ouner\\Desktop\\kiosk\\lee_kiosk"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ wav 폴더를 음성 입출력 표준 위치로 사용
WAV_DIR = os.path.join(PROJECT_DIR, "wav")
os.makedirs(WAV_DIR, exist_ok=True)

# ✅ MeloTTS 경로도 프로젝트 기준으로
MELO_PATH = os.path.join(PROJECT_DIR, "MeloTTS")
if MELO_PATH not in sys.path:
    sys.path.append(MELO_PATH)

from melo.api import TTS  # ✅ Melo TTS

# ------------------------------------------------
# 0. 공통 설정
# ------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")

# HOT/ICE 관련 키워드
HOT_KEYWORDS = {"핫", "뜨거운", "따뜻한"}
ICE_KEYWORDS = {"아이스", "차가운", "시원한"}

# ------------------------------------------------
# 1. 모델들 미리 로드 (프로그램 시작 시 1번만)
# ------------------------------------------------
print("🔊 Whisper 모델 로드 중 (small, fp16/int8)...")
whisper = WhisperModel(
    "small",
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8"
)

print("🎵 Melo TTS 로드 중...")
MELO_SPEED = 1.5
MELO_DEVICE = "cuda:0" if DEVICE == "cuda" else "cpu"

melo_tts = TTS(language="KR", device=MELO_DEVICE)
melo_speaker_ids = melo_tts.hps.data.spk2id  # 예: melo_speaker_ids["KR"]

print("✅ 모든 모델 로드 완료\n")

# ------------------------------------------------
# 2. OpenAI client (EEVE 대신)
# ------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PARSE_ORDER_TOOL = {
    "type": "function",
    "name": "parse_order",
    "description": "한국어 카페 주문 문장을 JSON 주문 객체로 변환한다.",
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "menu_name": {"type": "string", "minLength": 1},
                        "menu_quantity": {"type": "integer", "minimum": 1},
                        "menu_option": {"type": "string"},
                    },
                    "required": ["menu_name", "menu_quantity", "menu_option"],
                    "additionalProperties": False,
                },
            },
            "assistant_response": {"type": "string", "minLength": 1},
        },
        "required": ["items", "assistant_response"],
        "additionalProperties": False,
    },
}

# ------------------------------------------------
# 3. 유틸 함수들 (녹음 / STT / TTS)
# ------------------------------------------------
def record_to_file(filename: str, sec: float = 3.0, samplerate: int = 16000):
    """마이크 녹음 -> WAV 저장"""
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
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        without_timestamps=True,
        language="ko",
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"⏱ Whisper 인식 시간: {time.time() - t0:.2f}s")
    return text


@torch.inference_mode()
def speak_with_melo(text: str, out_path: str, speed: float = MELO_SPEED):
    """Melo TTS로 한국어 문장을 음성(WAV)으로 저장"""
    if len(text) > 120:
        print("🔁 TTS 텍스트가 길어서 앞 120자만 읽습니다.")
        text = text[:120]

    t0 = time.time()
    speaker_id = melo_speaker_ids["KR"]
    melo_tts.tts_to_file(text, speaker_id, out_path, speed=speed)
    print(f"⏱ Melo TTS 시간: {time.time() - t0:.2f}s")

# ------------------------------------------------
# 4. CSV 로딩 (인코딩 robust) - 라이트 RAG에 사용
# ------------------------------------------------
def open_text_robust(path: str, newline: str = ""):
    encodings = ("utf-8-sig", "cp949", "euc-kr", "utf-8")
    last_err = None
    for enc in encodings:
        f = None
        try:
            f = open(path, "r", encoding=enc, newline=newline)
            f.readline()   # 헤더 디코딩 강제
            f.seek(0)
            return f
        except UnicodeDecodeError as e:
            last_err = e
            if f:
                f.close()
    raise last_err


def load_menu_names(menu_csv_path: str) -> List[str]:
    names: List[str] = []
    with open_text_robust(menu_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "이름" not in reader.fieldnames:
            raise ValueError(f"data.csv에 '이름' 컬럼이 없음. fieldnames={reader.fieldnames}")
        for row in reader:
            n = (row.get("이름") or "").strip()
            if n:
                names.append(n)

    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def load_option_names(option_csv_path: str) -> List[str]:
    names: List[str] = []
    with open_text_robust(option_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "kor_name" not in reader.fieldnames:
            raise ValueError(f"drink_price.csv에 'kor_name' 컬럼이 없음. fieldnames={reader.fieldnames}")
        for row in reader:
            n = (row.get("kor_name") or "").strip()
            if n:
                names.append(n)

    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


MENU_CSV = os.path.join(PROJECT_DIR, "DATA", "data.csv")
OPTION_CSV = os.path.join(PROJECT_DIR, "DATA", "drink_price.csv")

# ✅ import 시 1번만 로드 (기존처럼 “초기 로딩” 스타일 유지)
MENU_NAMES: List[str] = load_menu_names(MENU_CSV)
OPTION_NAMES: List[str] = load_option_names(OPTION_CSV)

# ------------------------------------------------
# 5. 라이트 RAG(Top-K 후보 주입) + 규칙 후처리
# ------------------------------------------------
_KO_CLEAN_RE = re.compile(r"[^0-9a-zA-Z가-힣\s]")

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = _KO_CLEAN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def top_k_candidates(query: str, candidates: List[str], k: int = 15) -> List[str]:
    nq = normalize_text(query).replace(" ", "")
    scored: List[Tuple[float, str]] = []
    for c in candidates:
        nc = normalize_text(c).replace(" ", "")
        base = similarity(nq, nc)
        if nc and nc in nq:
            base += 0.25
        if nq and nq in nc:
            base += 0.10
        scored.append((base, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]

def ensure_candidates(cands: List[str], must: List[str], universe: List[str]) -> List[str]:
    s = set(cands)
    out = list(cands)
    uni = set(universe)
    for m in must:
        if m in uni and m not in s:
            out.insert(0, m)
            s.add(m)
    return out

NEG_WORDS = ["빼", "빼줘", "빼고", "없이", "없게", "제외", "넣지", "넣지마", "말아", "안 넣", "안넣"]
POS_WORDS = ["넣어", "넣어줘", "추가", "추가해", "추가해줘", "올려", "올려줘", "있게"]

def _contains_any(text: str, words: List[str]) -> bool:
    t = normalize_text(text).replace(" ", "")
    return any(normalize_text(w).replace(" ", "") in t for w in words)

def should_force_zero_cider(user_text: str) -> bool:
    t = normalize_text(user_text).replace(" ", "")
    if "제로사이다" not in t:
        return False
    return any(x in t for x in ["변경", "바꿔", "바꾸", "대신", "으로", "로"])

def intent_whip(user_text: str) -> Optional[str]:
    t = normalize_text(user_text).replace(" ", "")
    if ("휘핑" not in t) and ("휘핑크림" not in t):
        return None
    if _contains_any(user_text, NEG_WORDS):
        return "휘핑(X)"
    if _contains_any(user_text, POS_WORDS):
        return "휘핑(O)"
    return "휘핑(O)"

def intent_cinnamon(user_text: str) -> Optional[str]:
    t = normalize_text(user_text).replace(" ", "")
    if "시나몬" not in t:
        return None
    if _contains_any(user_text, NEG_WORDS):
        return "시나몬(X)"
    if _contains_any(user_text, POS_WORDS):
        return "시나몬(O)"
    return "시나몬(O)"

def best_match(value: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    nv = normalize_text(value)
    scored = [(similarity(nv, normalize_text(c)), c) for c in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]

def attach_option_first(items: List[Dict[str, Any]], option_value: str):
    if items:
        items[0]["menu_option"] = option_value

def apply_hot_ice_preference(user_text: str, menu_universe: List[str], items: List[Dict[str, Any]]) -> None:
    t = normalize_text(user_text).replace(" ", "")
    hot = any(normalize_text(x).replace(" ", "") in t for x in HOT_KEYWORDS)
    ice = any(normalize_text(x).replace(" ", "") in t for x in ICE_KEYWORDS)

    uni = set(menu_universe)
    for it in items:
        name = (it.get("menu_name") or "").strip()
        if not name:
            continue

        if hot and not name.endswith("(HOT)"):
            hot_name = f"{name}(HOT)"
            if hot_name in uni:
                it["menu_name"] = hot_name

        if ice and name.endswith("(HOT)"):
            base = name[:-5]  # remove "(HOT)"
            if base in uni:
                it["menu_name"] = base

def expand_to_single_cups(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        mn = str(it.get("menu_name", "")).strip()
        mo = str(it.get("menu_option", "")).strip()
        mq = it.get("menu_quantity", 1)
        try:
            mq = int(mq)
        except (TypeError, ValueError):
            mq = 1
        if mq < 1:
            mq = 1

        for _ in range(mq):
            out.append({"menu_name": mn, "menu_quantity": 1, "menu_option": mo})
    return out

def option_to_spoken(user_text: str, opt: str) -> str:
    t = normalize_text(user_text).replace(" ", "")
    opt = (opt or "").strip()

    if opt == "휘핑(O)":
        return "휘핑크림 넣어줘" if "휘핑크림" in t else "휘핑 넣어줘"
    if opt == "휘핑(X)":
        return "휘핑크림 빼줘" if "휘핑크림" in t else "휘핑 빼줘"

    if opt == "시나몬(O)":
        return "시나몬 넣어줘"
    if opt == "시나몬(X)":
        return "시나몬 빼줘"

    if opt == "제로사이다" and should_force_zero_cider(user_text):
        return "제로사이다로 변경"

    return opt

def build_assistant_response(user_text: str, items_single_cup: List[Dict[str, Any]]) -> str:
    if not items_single_cup:
        return "죄송합니다, 주문을 확인하지 못했어요. 다시 한 번 말씀해주시겠어요? 그대로 주문 도와드리겠습니다."

    counter: Dict[Tuple[str, str], int] = {}
    order_keys: List[Tuple[str, str]] = []

    for it in items_single_cup:
        name = (it.get("menu_name") or "").strip()
        opt = (it.get("menu_option") or "").strip()
        key = (name, opt)
        if key not in counter:
            counter[key] = 0
            order_keys.append(key)
        counter[key] += 1

    parts: List[str] = []
    for (name, opt) in order_keys:
        qty = counter[(name, opt)]
        spoken_opt = option_to_spoken(user_text, opt)

        if not spoken_opt:
            parts.append(f"{name} {qty}잔")
        else:
            parts.append(f"{name} {spoken_opt} {qty}잔")

    if len(parts) == 1:
        mid = parts[0]
    elif len(parts) == 2:
        mid = f"{parts[0]}과 {parts[1]}"
    else:
        mid = ", ".join(parts[:-1]) + f" 그리고 {parts[-1]}"

    return f"손님, 말씀하신 {mid} 그대로 주문 도와드리겠습니다."

def postprocess_with_candidates(
    user_text: str,
    data: Dict[str, Any],
    menu_candidates: List[str],
    option_candidates: List[str],
    menu_universe: List[str],
) -> Dict[str, Any]:
    items = data.get("items")
    if not isinstance(items, list):
        items = []

    fixed: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        mn = str(it.get("menu_name", "")).strip()
        mo = str(it.get("menu_option", "")).strip()
        mq = it.get("menu_quantity", 1)

        try:
            mq = int(mq)
        except (TypeError, ValueError):
            mq = 1
        if mq < 1:
            mq = 1

        if mn not in menu_candidates:
            mn = best_match(mn, menu_candidates) or (menu_candidates[0] if menu_candidates else mn)

        if mo not in option_candidates:
            mo = best_match(mo, option_candidates) or ""

        fixed.append({"menu_name": mn, "menu_quantity": mq, "menu_option": mo})

    if not fixed:
        fallback_menu = menu_candidates[0] if menu_candidates else "아메리카노"
        fixed = [{"menu_name": fallback_menu, "menu_quantity": 1, "menu_option": ""}]

    # HOT/ICE 의도 보정
    apply_hot_ice_preference(user_text, menu_universe, fixed)

    # 규칙 기반 옵션 강제
    if should_force_zero_cider(user_text) and "제로사이다" in option_candidates:
        attach_option_first(fixed, "제로사이다")

    w = intent_whip(user_text)
    if w and w in option_candidates:
        attach_option_first(fixed, w)

    c = intent_cinnamon(user_text)
    if c and c in option_candidates:
        attach_option_first(fixed, c)

    fixed_single = expand_to_single_cups(fixed)
    assistant_response = build_assistant_response(user_text, fixed_single)

    return {"items": fixed_single, "assistant_response": assistant_response}

def parse_order_light_rag(
    text: str,
    menu_names: List[str],
    option_names: List[str],
    model: str = "gpt-4o-mini",
    menu_top_k: int = 20,
    option_top_k: int = 20,
) -> Dict[str, Any]:
    menu_candidates = top_k_candidates(text, menu_names, k=menu_top_k)
    option_candidates = top_k_candidates(text, option_names, k=option_top_k)

    # Top-K에 안 들어가도 규칙용 옵션은 후보에 강제로 포함
    must_opts = ["제로사이다", "휘핑(O)", "휘핑(X)", "시나몬(O)", "시나몬(X)"]
    option_candidates = ensure_candidates(option_candidates, must_opts, option_names)

    option_candidates_plus = option_candidates + [""]  # 빈 옵션 허용

    instructions = f"""
너는 카페 키오스크 주문 파서다.
- 반드시 parse_order 함수를 호출해서만 답해라. (말로 설명 금지)
- items는 비우지 마라. 최소 1개 채워라.
- 수량이 없으면 menu_quantity=1
- 옵션이 없으면 menu_option=""

[중요: 후보 제한]
- menu_name은 아래 [MENU_CANDIDATES] 목록 중 하나로만 선택해라.
- menu_option은 아래 [OPTION_CANDIDATES] 목록 중 하나로만 선택해라.
- 목록에 없는 값은 절대 만들어내지 마라. 애매하면 가장 가까운 후보를 선택해라.

[MENU_CANDIDATES]
{json.dumps(menu_candidates, ensure_ascii=False)}

[OPTION_CANDIDATES]
{json.dumps(option_candidates_plus, ensure_ascii=False)}
""".strip()

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=text,
        tools=[PARSE_ORDER_TOOL],
        tool_choice={"type": "function", "name": "parse_order"},
        temperature=0.0,
    )

    for out in resp.output:
        if out.type == "function_call" and out.name == "parse_order":
            data = json.loads(out.arguments)
            return postprocess_with_candidates(
                user_text=text,
                data=data,
                menu_candidates=menu_candidates,
                option_candidates=option_candidates_plus,
                menu_universe=menu_names,
            )

    raise RuntimeError("parse_order tool call not found")

# ------------------------------------------------
# 6. 키오스크용 래퍼 클래스 (메서드/이름 유지)
# ------------------------------------------------
class VoiceAIKiosk:
    """
    키오스크 메인 코드(kiosk_t.py)에서 사용할 래퍼.
    - Whisper + Melo TTS는 import 시 이미 로드됨
    - EEVE 대신 OpenAI 라이트 RAG 사용
    """

    def __init__(self):
        print("✅ VoiceAIKiosk 초기화 (모델 이미 로드됨)")

    def record_and_stt(self, sec: float = 3.0, in_filename: str = "input.wav") -> str:
        wav_path = os.path.join(WAV_DIR, in_filename)
        record_to_file(wav_path, sec=sec)
        text = stt_whisper(wav_path)
        return text

    def parse_menu_json(self, user_text: str) -> dict:
        """
        user_text → OpenAI(라이트 RAG) →
        {
          "items": [ {menu_name, menu_quantity(항상1), menu_option}, ... ],
          "assistant_response": "..."
        }
        """
        try:
            result = parse_order_light_rag(
                text=user_text,
                menu_names=MENU_NAMES,
                option_names=OPTION_NAMES,
                model="gpt-4o-mini",
            )

            if not isinstance(result, dict):
                raise ValueError("LLM result is not a dict")

            # items 타입 안전장치
            items = result.get("items", [])
            if not isinstance(items, list):
                result["items"] = []

            # assistant_response 안전장치
            ar = result.get("assistant_response", "")
            if not isinstance(ar, str) or not ar.strip():
                result["assistant_response"] = "손님, 주문해 주셔서 감사합니다. 그대로 주문 도와드리겠습니다."

            return result

        except Exception as e:
            print("❗ parse_menu_json 실패:", e)
            return {
                "items": [
                    {
                        "menu_name": "",
                        "menu_quantity": 1,
                        "menu_option": ""
                    }
                ],
                "assistant_response": "죄송합니다, 주문 인식에 실패했습니다. 다시 한 번 말씀해주시겠어요? 그대로 주문 도와드리겠습니다."
            }

    def make_tts(self, text: str, out_filename: str = "response.wav", speed: float = MELO_SPEED) -> str:
        out_path = os.path.join(WAV_DIR, out_filename)
        speak_with_melo(text, out_path, speed=speed)
        return out_path

    def run_voice_order_once(self, record_sec: float = 3.0) -> dict:
        wav_path = os.path.join(WAV_DIR, "input.wav")
        record_to_file(wav_path, sec=record_sec)
        user_text = stt_whisper(wav_path)
        print("📝 STT 결과:", user_text)

        if not user_text:
            resp_text = "음성을 잘 듣지 못했어요. 다시 한 번 말씀해주시겠어요? 그대로 주문 도와드리겠습니다."
            return {
                "ok": False,
                "reason": "stt_empty",
                "stt_text": "",
                "items": [],
                "assistant_response": resp_text,
                "tts_path": "",
            }

        result = self.parse_menu_json(user_text)
        items = result.get("items", [])
        assistant_response = result.get("assistant_response", "")

        out_wav = os.path.join(WAV_DIR, "response.wav")
        speak_with_melo(assistant_response, out_wav)

        return {
            "ok": True,
            "reason": "ok",
            "stt_text": user_text,
            "items": items,
            "assistant_response": assistant_response,
            "tts_path": out_wav,
        }
