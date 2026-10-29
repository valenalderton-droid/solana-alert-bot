import json
import math
import os
import re
from datetime import datetime, timezone
import requests
import time

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


def telegram_send(token, chat_id, text):
    """Send a Telegram message with bounded retries for transient failures."""
    token = str(token or "").strip()
    chat_id = str(chat_id or "").strip()
    if not token or not chat_id:
        raise RuntimeError(
            "Telegram credentials are missing: configure TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID in GitHub Actions secrets."
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": str(text), "disable_web_page_preview": True}
    last_error = None

    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=20)
            # Retry only transient server/rate-limit failures. Invalid credentials,
            # blocked bots, and invalid chat IDs need configuration fixes instead.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(
                    f"Telegram temporary HTTP error {response.status_code}: "
                    f"{getattr(response, 'text', '')[:300]}"
                )
                if attempt < 2:
                    retry_after = 1
                    try:
                        retry_after = int(response.json().get("parameters", {}).get("retry_after", 1))
                    except (ValueError, TypeError, AttributeError):
                        pass
                    time.sleep(min(max(retry_after, 1), 5))
                    continue
                raise last_error

            response.raise_for_status()
            result = response.json()
            if not result.get("ok"):
                error_code = int(result.get("error_code", 0) or 0)
                message = result.get("description", "unknown Telegram API error")
                if (error_code == 429 or error_code >= 500) and attempt < 2:
                    last_error = RuntimeError(f"Telegram API temporary error {error_code}: {message}")
                    time.sleep(1 + attempt)
                    continue
                raise RuntimeError(f"Telegram sendMessage failed ({error_code}): {message}")
            return True
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status != 429 and status < 500:
                raise RuntimeError(
                    f"Telegram rejected the message with HTTP {status}; "
                    "check the bot token, chat ID, and bot permissions."
                ) from exc
            last_error = exc
            if attempt >= 2:
                raise RuntimeError(f"Telegram HTTP request failed after 3 attempts: {exc}") from exc
            time.sleep(1 + attempt)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= 2:
                raise RuntimeError(f"Telegram network request failed after 3 attempts: {exc}") from exc
            time.sleep(1 + attempt)

    raise RuntimeError(f"Telegram sendMessage failed after retries: {last_error}")


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
            print(f"Telegram getUpdates rejected: {payload}")
            return False

        requested = False
        matched_chat_message = False
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
            if not command:
                continue
            # The documented trigger is case-sensitive and must be exactly //Pay.
            # Ignore ordinary text; explain other slash commands without scanning.
            if command == "//Pay":
                matched_chat_message = True
                requested = True
                try:
                    telegram_send(
                        token, chat_id,
                        "درخواستت دریافت شد؛ اسکن بازار سولانا شروع می‌شود. نتیجه را همین‌جا می‌فرستم."
                    )
                except Exception as send_exc:
                    # A failed acknowledgement must not discard a valid command.
                    print(f"Telegram acknowledgement failed; continuing scan: {send_exc}")
            elif command.startswith("/"):
                matched_chat_message = True
                try:
                    telegram_send(
                        token, chat_id,
                        "دستور پشتیبانی نمی‌شود. برای بررسی بازار دقیقاً این متن را بفرست: //Pay"
                    )
                except Exception as send_exc:
                    print(f"Telegram help reply failed: {send_exc}")
            else:
                continue

        save_state(state)
        if not matched_chat_message:
            print("Telegram polling succeeded; no text message for the configured chat was found.")
        elif requested:
            print("Telegram //Pay command received and acknowledged.")
        return requested
    except (requests.RequestException, ValueError, TypeError, RuntimeError) as exc:
        print(f"Telegram command polling failed: {exc}")
        return False

def refresh_news_if_due(state, interval_minutes=15):
    """Refresh source-linked news between trade scans without hammering market APIs."""
    last_value = state.get("last_news_refresh_utc")
    if last_value:
        try:
            last = datetime.fromisoformat(str(last_value).replace("Z", "+00:00"))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - last.astimezone(timezone.utc)).total_seconds()
            if 0 <= age_seconds < interval_minutes * 60:
                return
        except (ValueError, TypeError, OverflowError):
            pass

    try:
        from news_analyzer import fetch_recent_news, analyze_news
        seen = state.setdefault("seen_news_ids", [])
        seen_set = set(seen)
        observations = state.setdefault("news_observations", [])
        added = 0
        for item in fetch_recent_news()[:12]:
            if item.get("id") in seen_set:
                continue
            analyzed = analyze_news(item)
            observations.append({
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "published_at": analyzed.get("published_at"),
                "sentiment": analyzed.get("sentiment", "NEUTRAL / UNCLEAR"),
                "impact_score": int(analyzed.get("impact_score", 0)),
                "relevance": analyzed.get("relevance", "UNCERTAIN"),
                "confidence": analyzed.get("confidence", "UNKNOWN"),
                "feed_category": analyzed.get("feed_category", "Uncategorized"),
                "analysis": analyzed.get("analysis", "")[:900],
                "title": analyzed.get("title", "")[:250],
                "url": analyzed.get("url", "")
            })
            seen.append(item["id"])
            seen_set.add(item["id"])
            added += 1
        # Expand research beyond the fixed RSS feeds. The research engine rotates
        # public-source queries based on the latest paper outcomes and approved projects.
        # Its findings remain evidence, not executable trading rules.
        research_added = 0
        try:
            from research_engine import run_research_cycle
            def read_outcomes_for_research():
                rows = []
                try:
                    with open("data/outcomes.jsonl", "r", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                except OSError:
                    pass
                return rows
            try:
                with open("data/projects.json", "r", encoding="utf-8") as handle:
                    project_config = json.load(handle)
            except (OSError, json.JSONDecodeError):
                project_config = {}
            research_result = run_research_cycle(
                state,
                outcomes=read_outcomes_for_research(),
                configured_projects=project_config.get("projects", [])
            )
            print("Research cycle result:", json.dumps(research_result, ensure_ascii=False))
            # Bring newly found headlines into the same transparent, source-linked
            # context stream used by the bounded risk policy; never create a signal.
            from news_analyzer import analyze_news
            research_seen = set(state.setdefault("seen_research_news_ids", []))
            for item in state.get("research_observations", [])[-40:]:
                if item.get("kind") == "metric" or item.get("id") in research_seen:
                    continue
                analyzed = analyze_news(item)
                observations.append({
                    "observed_at": item.get("observed_at") or datetime.now(timezone.utc).isoformat(),
                    "published_at": analyzed.get("published_at"),
                    "sentiment": analyzed.get("sentiment", "NEUTRAL / UNCLEAR"),
                    "impact_score": int(analyzed.get("impact_score", 0)),
                    "relevance": analyzed.get("relevance", "UNCERTAIN"),
                    "confidence": analyzed.get("confidence", "UNKNOWN"),
                    "feed_category": analyzed.get("feed_category", "Uncategorized"),
                    "analysis": analyzed.get("analysis", "")[:900],
                    "title": analyzed.get("title", "")[:250],
                    "url": analyzed.get("url", ""),
                    "research_topic": item.get("research_topic", "")
                })
                research_seen.add(item.get("id"))
                research_added += 1
            state["seen_research_news_ids"] = list(research_seen)[-1000:]
            state["news_observations"] = observations[-500:]
        except Exception as research_exc:
            # A research outage must never stop the bot or erase market state.
            print(f"Autonomous research failed safely: {research_exc}")

        state["news_observations"] = observations[-500:]
        state["seen_news_ids"] = seen[-500:]
        state["last_news_refresh_utc"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print(f"News-only refresh completed; new headlines analyzed: {added}; research headlines added: {research_added}")
    except Exception as exc:
        # News is supplemental; a temporary feed failure must not stop command polling.
        print(f"News-only refresh failed: {exc}")


def run_self_improvement_if_due(state):
    """Run the bounded learning review on scheduled cycles, even without open positions."""
    def read_jsonl(path):
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return rows

    try:
        from self_improver import update_self_improvement
        result = update_self_improvement(
            predictions=read_jsonl("data/predictions.jsonl"),
            outcomes=read_jsonl("data/outcomes.jsonl"),
            news_observations=state.get("news_observations", []),
            current_threshold=float(os.getenv("MIN_PROBABILITY", "0.60"))
        )
        print("Scheduled self-improvement review:", json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(f"Scheduled self-improvement review failed safely: {exc}")


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


def is_buy_alert(message):
    """Recognize both the legacy and ranked BUY alert titles."""
    text = str(message or "").upper()
    return "BUY WATCH" in text and "NOT AN AUTOMATIC TRADE" in text


def is_sell_alert(message):
    """Recognize explicit exit alerts, not watchlist/status messages."""
    text = str(message or "").upper()
    return (
        ("SELL" in text or "EXIT" in text)
        and ("ALERT" in text or "EXIT REASON" in text)
        and ("NO AUTOMATIC TRADE" in text or "EXIT ALERT ONLY" in text)
    )


def is_trade_alert(message):
    """Only explicit BUY and SELL/EXIT alerts may be delivered to Telegram."""
    return is_buy_alert(message) or is_sell_alert(message)


def main():
    state = read_state()
    requested = poll_pay(state)
    # Research and evidence-gated learning run on every scheduled cycle,
    # including when positions are open or the user requests a scan. Each has
    # its own cooldown; neither is allowed to bypass trading safety filters.
    refresh_news_if_due(state)
    run_self_improvement_if_due(state)

    # Keep checking exit conditions on scheduled runs while a paper position is
    # open. BUY messages are still suppressed unless the user explicitly sends //Pay.
    open_positions = state.get("open_positions", [])
    if not requested and not open_positions:
        print("No //Pay request and no open positions; market scan skipped.")
        return

    import main as scanner

    messages = []
    delivered_buys = []
    delivered_messages = []
    original_send_telegram = scanner.send_telegram

    def capture_buy(message):
        if is_buy_alert(message):
            if not requested:
                # Scheduled runs only monitor exits; never open new paper positions
                # without an explicit //Pay request.
                return False
            # Never send a placeholder as if it were a valid trade alert.
            compact = concise_reply([message])
            if compact == "NO_SETUP":
                print("BUY alert rejected: required fields could not be parsed; no Telegram message sent.")
                return False
            # Record the signal only after Telegram confirms delivery.
            sent = original_send_telegram(compact)
            if sent:
                messages.append(message)
                delivered_buys.append(message)
                delivered_messages.append(message)
            return sent

        if is_sell_alert(message):
            # Exit alerts are allowed on both scheduled and requested scans.
            sent = original_send_telegram(message)
            if sent:
                delivered_messages.append(message)
            return sent

        # Suppress watchlist, no-setup, news, and routine status messages.
        # The user requested Telegram alerts only for actionable BUY/SELL events.
        print("Non-trade Telegram message suppressed.")
        return False

    scanner.send_telegram = capture_buy
    scan_error = None
    try:
        scanner.main()
    except Exception as exc:
        scan_error = exc
        print(f"Market scan failed: {type(exc).__name__}: {exc}")
    finally:
        # Restore the module function even if the scan raises; otherwise a second
        # invocation in the same process can recursively wrap the Telegram sender.
        scanner.send_telegram = original_send_telegram

    if not requested:
        print("Scheduled position-monitoring scan completed; BUY alerts suppressed.")
        return

    # Always provide a final response to an explicit trigger, including when
    # the scanner crashes. Do not report a normal no-setup result after an error.
    if not delivered_messages:
        if scan_error is not None:
            final_reply = (
                "در اجرای اسکن بازار خطا رخ داد و نتیجه قابل اتکا نیست. "
                "گزارش خطا ثبت شد؛ لطفاً چند دقیقه دیگر دوباره //Pay را بفرست."
            )
        else:
            final_reply = (
                "اسکن بازار سولانا تمام شد، اما هیچ ستاپ خرید معتبری از فیلترهای فعلی عبور نکرد. "
                "فعلاً خریدی پیشنهاد نمی‌شود. برای بررسی دوباره، //Pay را بفرست."
            )
        try:
            telegram_send(
                os.getenv("TELEGRAM_BOT_TOKEN"),
                str(os.getenv("TELEGRAM_CHAT_ID", "")).strip(),
                final_reply
            )
            print("Final scan response sent for trigger.")
        except Exception as reply_exc:
            print(f"Final scan response failed: {reply_exc}")



if __name__ == "__main__":
    main()
