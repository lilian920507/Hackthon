# -*- coding: utf-8 -*-
from datetime import datetime
import os, re, json, time, traceback, logging
from flask import Flask, request, jsonify, abort, send_from_directory
from flask_cors import CORS
import openai
import speech_recognition as sr
from pydub import AudioSegment

# ===== LINEBOT SDK =====
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage, PushMessageRequest, AudioMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.exceptions import InvalidSignatureError
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = r".\ffmpeg.exe"
AudioSegment.ffprobe   = r".\ffprobe.exe"

# ------------------ 基本設定 ------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
AUDIO_DIR = UPLOAD_FOLDER  # 音檔就放同一個 uploads
os.makedirs(AUDIO_DIR, exist_ok=True)

# 可從環境變數帶入 (建議)；或直接填值
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "您的openai金鑰")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
# 例如 "https://xxxx-yy-zz.ngrok.io"
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "")

openai.api_key = OPENAI_API_KEY
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
# Logging 設定

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------ in-memory DB ------------------
owner_profiles = {}
owner_line_map = {}
trust_db = {}
trust_reverse = {}
import re

def sanitize_filename(filename):
    # 只允許 英數字、底線、點、減號
    return re.sub(r'[^A-Za-z0-9._-]', '_', filename)

def make_owner_key(name: str, phone: str) -> str:
    return f"{name.strip()}|{phone.strip()}"

# ------------------ 工具：推送警示 ------------------
def push_alert(to_id: str, name: str, phone: str, message_text: str, reason: str,
               audio_url: str = None, duration_ms: int = None):
    try:
        logging.debug(f"準備推播給 {to_id}, name={name}, phone={phone}, reason={reason}")
        payload = (
            f"⚠️ 你被設定為 {name} 的信任人\n\n"
            f"當事人姓名：{name}\n電話：{phone}\n\n"
            f"模型判斷：詐騙\n理由：{reason}\n\n"
            "請立即提醒當事人不要匯款或提供個資。"
        )

        messages = [TextMessage(text=payload)]
        if audio_url and duration_ms:
            logging.debug(f"附加音檔: {audio_url}, duration={duration_ms}")
            messages.append(AudioMessage(original_content_url=audio_url, duration=duration_ms))

        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            api.push_message(PushMessageRequest(to=to_id, messages=messages))
        logging.info(f"✅ 成功推播給 {to_id}")
    except Exception:
        logging.error("推播失敗: %s", traceback.format_exc())

def notify_trusts_and_owner(owner_k: str, text: str, reason: str,
                            audio_url: str = None, duration_ms: int = None):
    name = owner_profiles.get(owner_k, {}).get("name", "當事人")
    phone = owner_profiles.get(owner_k, {}).get("phone", "未知")

    logging.debug(f"通知 owner={owner_k}, 信任人數={len(trust_db.get(owner_k, []))}")

    # 信任人
    for t in trust_db.get(owner_k, []):
        push_alert(t["id"], name, phone, text, reason, audio_url, duration_ms)

    # 本人
    owner_line = owner_line_map.get(owner_k)
    if owner_line:
        push_alert(owner_line, name, phone, text, reason, audio_url, duration_ms)

# ------------------ 語音處理 ------------------
def convert_audio_to_wav(file_path: str) -> str:
    logging.debug(f"轉換音檔為 wav: {file_path}")
    audio = AudioSegment.from_file(file_path)
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def get_audio_duration_ms(file_path: str) -> int:
    audio = AudioSegment.from_file(file_path)
    return len(audio)

def speech_to_text_google(wav_path: str) -> str:
    logging.debug(f"語音轉文字: {wav_path}")
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data, language="zh-TW")
        logging.debug(f"語音辨識結果: {text}")
        return text
    except Exception as e:
        logging.error(f"語音辨識失敗: {e}")
        return f"（語音辨識失敗: {e}）"

def analyze_audio_with_openai(filename: str, timestamp: str, text: str) -> dict:
    logging.debug(f"丟給 OpenAI 分析: filename={filename}, time={timestamp}")
    try:
        system_prompt = (
            "你是一個專門識別詐騙風險的專家。"
            "請輸出 JSON："
            "{\"filename\":..., \"timestamp\":..., \"scam_status\":\"詐騙\"或\"非詐騙\", "
            "\"scam_type\":\"交友詐騙\"或\"投資詐騙\"或\"網購詐騙\"或\"N/A\", "
            "\"reason\":\"理由\"}"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
        raw = resp.choices[0].message.content.strip()
        logging.debug(f"OpenAI 回應: {raw}")
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {"scam_status": "未知", "reason": "無法解析"}
    except Exception:
        logging.error("OpenAI 分析失敗: %s", traceback.format_exc())
        return {"scam_status": "未知", "reason": "分析失敗"}

# ------------------ 對外提供音檔 ------------------
@app.route("/audios/<filename>")
def serve_audio(filename):
    file_path = os.path.join(AUDIO_DIR, filename)
    if filename.lower().endswith(".mp3"):
        mimetype = "audio/mpeg"
    elif filename.lower().endswith(".wav"):
        mimetype = "audio/wav"
    else:
        mimetype = "application/octet-stream"

    return send_from_directory(AUDIO_DIR, filename, mimetype=mimetype)

# ------------------ API: 上傳 ------------------
@app.route('/upload', methods=['POST'])
def upload_audio():
    logging.info("收到 /upload 請求")
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    allowed = ('.amr', '.mp3', '.m4a', '.ogg', '.flac', '.wav')
    if not file.filename.lower().endswith(allowed):
        return jsonify({"error": "Unsupported file format"}), 400

    # 🔹 檔名消毒
    safe_filename = sanitize_filename(file.filename)
    save_path = os.path.join(AUDIO_DIR, safe_filename)
    file.save(save_path)
    app.logger.debug(f"檔案已保存: {save_path}")

    time.sleep(1)  # 確保寫入完成
    current_time = datetime.now().strftime("%Y/%m/%d %H:%M")

    try:
        wav_path = convert_audio_to_wav(save_path)
        duration_ms = get_audio_duration_ms(save_path)
        text = speech_to_text_google(wav_path)
        analysis = analyze_audio_with_openai(file.filename, current_time, text)

        logging.info(f"分析結果: {analysis}")

        if analysis.get("scam_status") == "詐騙":
            # 用消毒過的檔名，避免空白、括號導致 LINE API 拒絕
            audio_url = f"{PUBLIC_BASE_URL}/uploads/{safe_filename}"
            app.logger.debug(f"推播用 URL: {audio_url}")
            for ok in list(owner_profiles.keys()):
                notify_trusts_and_owner(ok, text, analysis.get("reason", ""), audio_url, duration_ms)

        return jsonify(analysis), 200
    except Exception:
        logging.error("處理語音失敗: %s", traceback.format_exc())
        return jsonify({"error": "processing failed"}), 500

# ------------------ LINE 綁定 ------------------
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    uid = event.source.user_id
    text = event.message.text.strip()
    logging.info(f"收到 LINE 訊息: {text} from {uid}")

    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)

        m_owner = re.match(r"^我是當事人\s+(.+?)\s+電話\s+([0-9+\- ]+)$", text)
        if m_owner:
            name = m_owner.group(1).strip()
            phone = re.sub(r"[^\d+]", "", m_owner.group(2))
            ok = make_owner_key(name, phone)
            owner_profiles[ok] = {"name": name, "phone": phone}
            owner_line_map[ok] = uid
            logging.debug(f"綁定當事人: {ok} -> {uid}")
            api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=f"✅ 已綁定當事人：{name}（{phone}）")]
            ))
            return

        m_trust = re.match(r"^我是信任人\s+(.+?)\s+負責\s+(.+?)\s+電話\s+([0-9+\- ]+)$", text)
        if m_trust:
            trust_name = m_trust.group(1).strip()
            owner_name = m_trust.group(2).strip()
            owner_phone = re.sub(r"[^\d+]", "", m_trust.group(3))
            ok = make_owner_key(owner_name, owner_phone)
            owner_profiles.setdefault(ok, {"name": owner_name, "phone": owner_phone})
            trust_db.setdefault(ok, []).append({"id": uid, "trust_name": trust_name})
            trust_reverse[uid] = ok
            logging.debug(f"綁定信任人: {trust_name} -> {ok}")
            api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=f"✅ 你是 {owner_name}（{owner_phone}）的信任人 {trust_name}")]
            ))
            return

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True,threaded=True)