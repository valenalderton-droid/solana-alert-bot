import json
import math
import os
import re
import requests

STATE_FILE = "data/state.json"


def read_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state):
    os.makedirs("data", exist_ok=True)
    temp = STATE_FILE + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    os.replace(temp, STATE_FILE)


def poll_pay(state):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not chat_id:
        print("Telegram secrets missing; no command processed.")
        return False

    offset = int(state.get("telegram_update_offset", 0) or 0)
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params={"offset": offset + 1, "timeout": 0, "allowed_updates": json.dumps(["message"])},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            return False

        requested = False
        for update in payload.get("result", []):
            state["telegram_update_offset"] = max(
                int(state.get("telegram_update_offset", 0) or 0),
                int(update.get("update_id", 0)),
            )
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            if str(chat.get("id", "")) != chat_id:
                continue
            command = str(message.get("text", "")).strip()
            if command == "//Pay":
                requested = True
        save_state(state)
        return requested
    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"Telegram command polling failed: {exc}")
        return False


def concise_reply(messages):
    candidates = []
    for message in messages:
        project = re.search(r"^Project: (.+)$", message, re.MULTILINE)
        entry = re.search(r"^Entry reference: \$([0-9.eE+-]+)", message, re.MULTILINE)
        stop = re.search(r"^Stop reference: \$([0-9.eE+-]+)", message, re.MULTILINE)
        target = re.search(r"^Target reference: \$([0-9.eE+-]+)", message, re.MULTILINE)
        probability = re.search(r"^Combined model probability: ([0-9.]+)%?$", message, re.MULTILINE)
        mint = re.search(r"^Mint: (.+)$", message, re.MULTILINE)
        horizons = re.search(r"^Probability by horizon: (.+)$", message, re.MULTILINE)
        liquidity = re.search(r"^Liquidity: \$([0-9,]+)", message, re.MULTILINE)
        volume = re.search(r"^24h volume: \$([0-9,]+)", message, re.MULTILINE)
        if not all((project, entry, stop, target, probability, mint)):
            continue
        try:
            model_score = float(probability.group(1).rstrip("%"))
            horizon_values = {}
            if horizons:
                parsed = json.loads(horizons.group(1))
                if isinstance(parsed, dict):
                    horizon_values = {
                        str(key): float(value) for key, value in parsed.items()
                        if isinstance(value, (int, float)) and 0 <= float(value) <= 1
                    }
            # Prefer agreement across time horizons and better liquidity/volume,
            # rather than choosing by the model's raw score alone.
            consensus = (
                sum(value >= 0.50 for value in horizon_values.values()) / len(horizon_values)
                if horizon_values else 0.5
            )
            liq = float(liquidity.group(1).replace(",", "")) if liquidity else 100000.0
            vol = float(volume.group(1).replace(",", "")) if volume else 50000.0
            liquidity_bonus = min(4.0, max(0.0, 2.0 * math.log10(max(liq, 1) / 100000.0 + 1)))
            volume_bonus = min(3.0, max(0.0, 1.5 * math.log10(max(vol, 1) / 50000.0 + 1)))
            rank_score = model_score + 8.0 * consensus + liquidity_bonus + volume_bonus
            candidates.append((
                rank_score, model_score, project.group(1).strip(), entry.group(1),
                target.group(1), stop.group(1), mint.group(1).strip()
            ))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    if not candidates:
        return "NO_SETUP"
    _, _, project, entry, target, stop, mint = max(candidates, key=lambda row: row[0])
    return f"{project} | خرید: ${entry} | فروش: ${target} | حدضرر: ${stop} | {mint}"


def main():
    state = read_state()
    requested = poll_pay(state)
    # Keep checking exit conditions on scheduled runs while a paper position is
    # open. BUY messages are still suppressed unless the user explicitly sends //Pay.
    open_positions = state.get("open_positions", [])
    if not requested and not open_positions:
        print("No //Pay request and no open positions; scan skipped.")
        return

    import main as scanner

    messages = []
    original_send_telegram = scanner.send_telegram

    def capture_buy(message):
        if "SOLANA BUY WATCH" in message:
            messages.append(message)
            # BUY alerts are summarized into one compact reply only on //Pay.
            return False
        # SELL/EXIT alerts must reach Telegram even on scheduled monitoring runs.
        return original_send_telegram(message)

    scanner.send_telegram = capture_buy
    scanner.main()

    if not requested:
        print("Scheduled position-monitoring scan completed; BUY alerts suppressed.")
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    reply = concise_reply(messages)
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": reply, "disable_web_page_preview": True},
        timeout=20,
    )
    response.raise_for_status()


if __name__ == "__main__":
    main()
