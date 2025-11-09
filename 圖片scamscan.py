# -*- coding: utf-8 -*-
import os, re, json, traceback, logging
from datetime import datetime
from flask import Flask, request, jsonify, abort, send_from_directory
from flask_cors import CORS
from paddleocr import PaddleOCR
import openai
import cv2
import jieba.posseg as pseg

OPENAI_API_KEY = "您的openai金鑰"
LINE_CHANNEL_ACCESS_TOKEN = ""
LINE_CHANNEL_SECRET = ""
PUBLIC_BASE_URL = ""

# ===== LINEBOT SDK =====
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage, PushMessageRequest, ImageMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.exceptions import InvalidSignatureError

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)

# ------------------ Flask ------------------
app = Flask(__name__)
CORS(app)

# ------------------ 上傳設定 ------------------
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ OCR ------------------
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# ------------------ API Keys (請改成你的) ------------------


openai.api_key = OPENAI_API_KEY
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ------------------ 資料庫 (in-memory demo) ------------------
owner_profiles = {}   # owner_key -> {name, phone}
owner_line_map = {}   # owner_key -> line user id
trust_db = {}         # owner_key -> [ {id, trust_name} ]
trust_reverse = {}    # trust id -> owner_key

def make_owner_key(name: str, phone: str) -> str:
    return f"{name.strip()}|{phone.strip()}"

# ------------------ OCR Function ------------------
def process_ocr(image_path):
    logging.debug(f"開始 OCR: {image_path}")
    try:
        results = ocr.ocr(image_path, cls=True)
        extracted_text = ""
        # PaddleOCR 回傳結果可能為 list(list(...))
        if results and isinstance(results, list):
            # results 可能是 [ [ [box, (txt,score)], ... ] ]
            for block in results:
                if not isinstance(block, list):
                    continue
                for line in block:
                    try:
                        info = line[1]
                        text = info[0] if isinstance(info, (list, tuple)) else info
                        extracted_text += str(text) + "\n"
                    except Exception:
                        continue
        logging.debug(f"OCR 辨識結果: {extracted_text}")
        return extracted_text, results
    except Exception:
        logging.error("OCR 失敗: %s", traceback.format_exc())
        return "（OCR 失敗）", []

# ------------------ 去敏處理 ------------------
PATTERNS = {
    "電話": re.compile(r"0\d{1,4}-?\d{6,8}"),
    "身份證": re.compile(r"[A-Z]\d{9}"),
    "數字金額": re.compile(r"\d{3,}"),
}

def desensitize_image(image_path, ocr_results):
    """
    去敏：只遮掉敏感字，不會整行消失
    :param image_path: 原始圖片路徑
    :param ocr_results: PaddleOCR 回傳結果
    :return: (去敏後的 numpy 圖片, 去敏後的文字列表)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("無法讀取圖片：" + image_path)

    masked_texts = []
    # results may be structure like [ [ [box, (text, score)], ... ] ]
    blocks = ocr_results if isinstance(ocr_results, list) else []
    # If Paddle returns nested list, we use first-level lists
    # some implementations return results[0] as the list we want
    if len(blocks) == 1 and isinstance(blocks[0], list):
        lines_iter = blocks[0]
    else:
        # fallback: flatten
        lines_iter = []
        for b in blocks:
            if isinstance(b, list):
                lines_iter.extend(b)

    for line in lines_iter:
        # 防護：line 格式須為 [box, (text, score)]
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        box = line[0]
        info = line[1]
        if not info:
            continue
        # info 可能為 tuple/list 或直接字串
        if isinstance(info, (list, tuple)):
            original_text = str(info[0]).strip()
        else:
            original_text = str(info).strip()

        if not original_text:
            masked_texts.append("")
            continue

        # 計算座標
        try:
            pts = [(int(p[0]), int(p[1])) for p in box]
        except Exception:
            masked_texts.append(original_text)
            continue

        x_min = min(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        x_max = max(p[0] for p in pts)
        y_max = max(p[1] for p in pts)
        char_width = (x_max - x_min) / max(len(original_text), 1)

        masked_text = original_text
        sensitive_spans = []

        # Regex 偵測
        for key, pat in PATTERNS.items():
            for m in pat.finditer(original_text):
                sensitive_spans.append(m.group())

        # jieba 偵測人名 (nr)
        try:
            for w, flag in pseg.lcut(original_text):
                if flag == "nr" and len(w) >= 2:
                    sensitive_spans.append(w)
        except Exception:
            # 若 jieba 有問題不讓整個流程停掉
            logging.debug("jieba 解析失敗或無法使用")

        # 去除重複 span 並保持原順序
        seen = set()
        spans_ordered = []
        for s in sensitive_spans:
            if s not in seen:
                seen.add(s)
                spans_ordered.append(s)

        # 逐個遮罩敏感字串（只遮該字範圍）
        for span in spans_ordered:
            pos = masked_text.find(span)
            if pos == -1:
                # 若原始沒有，嘗試在 original_text 找位置
                pos = original_text.find(span)
            if pos != -1:
                start_x = int(x_min + pos * char_width)
                end_x = int(x_min + (pos + len(span)) * char_width)
                # safety clamp
                start_x = max(0, min(start_x, img.shape[1]-1))
                end_x = max(0, min(end_x, img.shape[1]-1))
                cv2.rectangle(img, (start_x, y_min), (end_x, y_max), (0, 0, 0), -1)
                masked_text = masked_text.replace(span, "*" * len(span))

        masked_texts.append(masked_text)
        if spans_ordered:
            logging.info("[去敏] 原文: %s", original_text)
            logging.info("[去敏] 結果: %s", masked_text)
            logging.info("[去敏] 被遮敏感字: %s", spans_ordered)

    return img, masked_texts

# ------------------ OpenAI Scam Judge ------------------
def analyze_with_openai(filename, timestamp, text):
    logging.debug(f"送 OpenAI 分析: {filename}, text={text[:50]}...")
    # 如果你沒有 openai key，這裡也可先回傳偵測規則（方便測試）
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("YOUR_"):
        # 簡單關鍵字規則
        kw = ["匯款", "中獎", "轉帳", "帳號", "密碼", "投資", "提款", "驗證碼"]
        hit = any(k in text for k in kw)
        return {
            "scam_status": "詐騙" if hit else "非詐騙",
            "reason": "離線關鍵字規則" if hit else "未命中關鍵字",
            "detail": ""
        }

    system_prompt = (
        "你是一個專門識別詐騙風險的專家。"
        "請嚴格按照以下格式返回 JSON："
        "{\"filename\": <圖片名>, \"timestamp\": <時間>, \"scam_status\": \"詐騙\"或\"非詐騙\", "
        "\"scam_type\": \"交友詐騙\"或\"投資詐騙\"或\"網購詐騙\"或\"N/A\", \"reason\": \"分析理由\"}"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=300,
            temperature=0.0
        )
        raw = resp.choices[0].message.content
        logging.debug(f"OpenAI 回覆原文: {raw}")
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {"scam_status": "未知", "reason": raw}
    except Exception:
        logging.error("OpenAI 分析失敗: %s", traceback.format_exc())
        return {"scam_status": "未知", "reason": "分析失敗"}

# ------------------ 推播警示 ------------------
def push_alert(to_id: str, name: str, phone: str, message_text: str, judge: dict, image_url: str = None):
    reason = judge.get("reason", "可能詐騙")
    payload = (
        f"⚠️ 你被設定為 {name} 的信任人\n\n"
        f"當事人姓名：{name}\n電話：{phone}\n\n"
        f"模型判斷：{judge.get('label', judge.get('scam_status','未知'))}\n理由：{reason}\n\n"
        "請立即提醒當事人不要匯款或提供個資。"
    )

    messages = [TextMessage(text=payload)]
    if image_url:
        # ImageMessage requires externally reachable URLs
        messages.append(ImageMessage(original_content_url=image_url, preview_image_url=image_url))

    try:
        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            api.push_message(PushMessageRequest(to=to_id, messages=messages))
        logging.info("✅ 推播成功 to=%s", to_id)
    except Exception:
        logging.error("推播失敗: %s", traceback.format_exc())

# ------------------ API: 上傳圖片 ------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    logging.info("📥 收到 /upload 請求")
    if 'file' not in request.files:
        logging.warning("No file in request")
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    # 安全檔名
    safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    try:
        file.save(save_path)
    except Exception:
        logging.error("檔案儲存失敗: %s", traceback.format_exc())
        return jsonify({"error": "file save failed"}), 500
    logging.info("✅ 原始圖片已保存: %s", save_path)

    # OCR
    ocr_text, ocr_results = process_ocr(save_path)

    # 去敏處理
    try:
        masked_img, masked_texts = desensitize_image(save_path, ocr_results)
    except Exception:
        logging.error("去敏處理失敗: %s", traceback.format_exc())
        return jsonify({"error": "desensitize failed"}), 500

    # 產生 masked 檔名（保持副檔名）
    name_root, ext = os.path.splitext(safe_name)
    masked_name = f"{name_root}_masked{ext}"
    masked_path = os.path.join(app.config['UPLOAD_FOLDER'], masked_name)

    # 寫檔並檢查
    try:
        ok = cv2.imwrite(masked_path, masked_img)
        if not ok:
            logging.error("cv2.imwrite 回傳 False，寫檔失敗")
            return jsonify({"error": "masked save failed"}), 500
    except Exception:
        logging.error("去敏圖片存檔失敗: %s", traceback.format_exc())
        return jsonify({"error": "masked save failed"}), 500

    logging.info("✅ 去敏圖片已保存: %s", masked_path)

    # 使用去敏後 OCR 文字（若沒辨識到，fallback 為原始 OCR 文字）
    text_for_analysis = "\n".join(masked_texts) if masked_texts else ocr_text

    # 分析 (openai)
    current_time = datetime.now().strftime("%Y/%m/%d %H:%M")
    analysis = analyze_with_openai(masked_name, current_time, text_for_analysis)

    # 若為詐騙就通知信任人（推播圖片 URL 要能被外界取得到）
    if analysis.get("scam_status") == "詐騙":
        # 確保 PUBLIC_BASE_URL 沒有尾端斜線問題
        base = PUBLIC_BASE_URL.rstrip('/')
        image_url = f"{base}/uploads/{masked_name}"
        logging.info("推播用 image_url=%s", image_url)
        for ok in owner_profiles.keys():
            name = owner_profiles.get(ok, {}).get("name", "當事人")
            phone = owner_profiles.get(ok, {}).get("phone", "未知")
            judge = {
                "score": 0.9,
                "label": "詐騙",
                "reason": analysis.get("reason", "")
            }
            for t in trust_db.get(ok, []):
                push_alert(t["id"], name, phone, text_for_analysis, judge, image_url)
            if owner_line_map.get(ok):
                push_alert(owner_line_map[ok], name, phone, text_for_analysis, judge, image_url)

    return jsonify({"analysis": analysis, "masked_file": masked_name}), 200

# ------------------ 提供上傳檔案 ------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # 直接從 uploads 資料夾讀取
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ------------------ LINE Webhook ------------------
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
    logging.info("收到 LINE 訊息: %s from %s", text, uid)

    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)

        # 綁定當事人
        m_owner = re.match(r"^我是當事人\s+(.+?)\s+電話\s+([0-9+\- ]+)$", text)
        if m_owner:
            name = m_owner.group(1).strip()
            phone = re.sub(r"[^\d+]", "", m_owner.group(2))
            ok = make_owner_key(name, phone)
            owner_profiles[ok] = {"name": name, "phone": phone}
            owner_line_map[ok] = uid
            logging.debug("綁定當事人: %s -> %s", ok, uid)
            api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=f"✅ 已建立當事人：{name} {phone}")]
                )
            )
            return

        # 綁定信任人
        m_trust = re.match(r"^我是信任人\s+(.+?)\s+負責\s+(.+?)\s+電話\s+([0-9+\- ]+)$", text)
        if m_trust:
            trust_name = m_trust.group(1).strip()
            owner_name = m_trust.group(2).strip()
            owner_phone = re.sub(r"[^\d+]", "", m_trust.group(3))
            ok = make_owner_key(owner_name, owner_phone)
            owner_profiles.setdefault(ok, {"name": owner_name, "phone": owner_phone})
            trust_db.setdefault(ok, []).append({"id": uid, "trust_name": trust_name})
            trust_reverse[uid] = ok
            logging.debug("綁定信任人: %s -> %s", trust_name, ok)
            api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=f"✅ 你是 {owner_name} 的信任人 {trust_name}")]
                )
            )
            return

        # 信任人回覆轉發當事人
        if uid in trust_reverse:
            ok = trust_reverse[uid]
            owner_line = owner_line_map.get(ok)
            name = owner_profiles.get(ok, {}).get("name", "當事人")
            phone = owner_profiles.get(ok, {}).get("phone", "未知")
            # forward text to owner if owner has line id
            if owner_line:
                try:
                    api.push_message(
                        PushMessageRequest(
                            to=owner_line,
                            messages=[TextMessage(text=f"📩 你的信任人回覆：{text}")]
                        )
                    )
                    api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text="已轉發給當事人。")]
                        )
                    )
                except Exception:
                    api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text="轉發失敗，請稍後再試或直接用電話聯絡。")]
                        )
                    )
            else:
                api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=f"當事人尚未加入 Bot。\n姓名：{name}\n電話：{phone}\n請改以電話或簡訊聯絡。")]
                    )
                )
            return

        # 其他訊息：簡短說明
        help_text = (
            "使用說明：\n"
            "1) 建立當事人：我是當事人 王小明 電話 0912345678\n"
            "2) 綁定信任人：我是信任人 小明媽媽 負責 王小明 電話 0912345678\n"
            "（註：App 偵測到可疑時會通知信任人）"
        )
        api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=help_text)]
            )
        )

# ------------------ 啟動 ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
