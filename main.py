
import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import requests

from market import (
    get_best_solana_pair, get_hourly_candles, utc_now,
    discover_solana_projects, is_allowed_non_meme_project
)
from adaptive_policy import derive_policy
from self_improver import update_self_improvement, evaluate_experience_rules
from signal_quality import score_signal
from strategy_ensemble import evaluate_strategies, rank_candidate
from learner import train_model, latest_features, price_action_ok
from paper_portfolio import open_paper_trade, update_paper_portfolio, portfolio_snapshot
from news_analyzer import fetch_recent_news, analyze_news, format_news_alert

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROJECTS_FILE = DATA / "projects.json"
STATE_FILE = DATA / "state.json"
CANDLES_FILE = DATA / "candles.jsonl"
PREDICTIONS_FILE = DATA / "predictions.jsonl"
OUTCOMES_FILE = DATA / "outcomes.jsonl"
PAPER_PREDICTIONS_FILE = DATA / "paper_predictions.jsonl"
PAPER_OUTCOMES_FILE = DATA / "paper_outcomes.jsonl"

MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY_USD", "100000"))
MIN_PROBABILITY = float(os.getenv("MIN_PROBABILITY", "0.60"))
MAX_STOP_PCT = float(os.getenv("MAX_STOP_PCT", "0.05"))
TARGET_PCT = float(os.getenv("TARGET_PCT", "0.10"))
MAX_CANDLE_HISTORY = 1500


def find_resistance_target(candles, entry, minimum_target_pct, max_target_pct=0.30):
    """Return a conservative target just below the nearest confirmed swing high.

    Uses only completed candles supplied by the caller. Returns None when no
    resistance can support the required reward/risk or the target is too far.
    """
    try:
        entry = float(entry)
        minimum_target_pct = float(minimum_target_pct)
        max_target_pct = float(max_target_pct)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(entry) or entry <= 0 or minimum_target_pct < 0:
        return None
    if len(candles or []) < 5:
        return None

    minimum_price = entry * (1.0 + minimum_target_pct)
    maximum_price = entry * (1.0 + max_target_pct)
    highs = []
    for candle in (candles or [])[-100:]:
        try:
            high = float(candle.get("high") or 0)
        except (TypeError, ValueError, OverflowError):
            high = 0.0
        highs.append(high if math.isfinite(high) and high > 0 else 0.0)

    # A confirmed local swing high must have two candles on either side.
    targets = []
    for i in range(2, len(highs) - 2):
        high = highs[i]
        if high <= 0 or high <= entry:
            continue
        if high >= max(highs[i - 2:i] + highs[i + 1:i + 3]):
            target = high * 0.995  # place target just before resistance
            if minimum_price <= target <= maximum_price:
                targets.append(target)
    return min(targets) if targets else None


def build_comprehensive_analysis(candles, price, ensemble=None, quality=None,
                                horizon_probabilities=None, global_context=None,
                                news_context=None, liquidity=0, volume24h=0,
                                entry=None, stop=None, target=None):
    """Build a compact, evidence-based technical/fundamental-context report.

    Candles are hourly. The report explicitly labels unavailable data and does
    not convert strategy scores into win probabilities.
    """
    ensemble = ensemble or {}
    quality = quality or {}
    horizon_probabilities = horizon_probabilities or {}
    global_context = global_context or {}
    news_context = news_context or []
    try:
        candles = sorted(
            [c for c in (candles or []) if isinstance(c, dict)],
            key=lambda c: int(c.get("timestamp", 0))
        )
    except (TypeError, ValueError, OverflowError):
        candles = []

    def num(value, default=None):
        try:
            value = float(value)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError, OverflowError):
            return default

    def pct_change(old, new):
        return (new / old - 1.0) * 100 if old and new and old > 0 else None

    def fmt_pct(value):
        return "ناموجود" if value is None else f"{value:+.2f}%"

    lines = ["\n\n━━ تحلیل جامع بازار و پرایس اکشن ━━"]
    price = num(price, 0) or 0
    lines.append(f"قیمت مرجع: ${price:.8g}" if price > 0 else "قیمت مرجع: نامعتبر")
    lines.append(f"نقدینگی استخر: ${num(liquidity, 0):,.0f} | حجم ۲۴ساعته: ${num(volume24h, 0):,.0f}")
    if liquidity and num(liquidity, 0) > 0:
        lines.append(f"نسبت حجم ۲۴ساعته به نقدینگی: {num(volume24h, 0) / num(liquidity, 1) * 100:.1f}%")

    closes, highs, lows, volumes = [], [], [], []
    for c in candles:
        close, high, low = num(c.get("close")), num(c.get("high")), num(c.get("low"))
        if close and high and low and close > 0 and high >= low > 0:
            closes.append(close)
            highs.append(high)
            lows.append(low)
            volumes.append(max(0.0, num(c.get("volume"), 0) or 0))
    if len(closes) >= 2:
        lines.append(f"داده نمودار: {len(closes)} کندل ساعتی معتبر")
        for label, span in (("۱ساعت", 1), ("۴ساعت", 4), ("۲۴ساعت", 24), ("۷۲ساعت", 72), ("۷روز", 168)):
            if len(closes) > span:
                lines.append(f"بازده {label}: {fmt_pct(pct_change(closes[-span-1], closes[-1]))}")
        last = closes[-1]
        for span, label in ((24, "۲۴ساعت"), (72, "۷۲ساعت"), (168, "۷روز")):
            if len(lows) >= span:
                subset_lows, subset_highs = lows[-span:], highs[-span:]
                support, resistance = min(subset_lows), max(subset_highs)
                lines.append(
                    f"سطوح {label}: حمایت تقریبی ${support:.8g} ({pct_change(last, support):+.2f}%)؛ "
                    f"مقاومت تقریبی ${resistance:.8g} ({pct_change(last, resistance):+.2f}%)"
                )
            else:
                lines.append(f"سطوح {label}: تاریخچه کامل این بازه موجود نیست")
        if len(closes) >= 50:
            ema20 = sum(closes[-20:]) / 20
            ema50 = sum(closes[-50:]) / 50
            if len(closes) >= 70:
                # Approximate EMA using a stable seed and recursive smoothing.
                def ema(values, period):
                    value = sum(values[:period]) / period
                    alpha = 2 / (period + 1)
                    for item in values[period:]:
                        value = alpha * item + (1-alpha) * value
                    return value
                ema20, ema50 = ema(closes[-100:], 20), ema(closes[-100:], 50)
            trend = "صعودی" if last > ema20 > ema50 else "نزولی" if last < ema20 < ema50 else "مختلط/خنثی"
            lines.append(f"رژیم روند با EMA20/50: {trend}؛ EMA20 ${ema20:.8g}؛ EMA50 ${ema50:.8g}")
        else:
            lines.append("رژیم روند EMA20/50: تاریخچه کافی نیست")
        if len(closes) >= 15:
            tr = []
            for i in range(max(1, len(closes)-14), len(closes)):
                tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) / closes[i])
            atr_pct = sum(tr) / len(tr) * 100 if tr else None
            gains, losses = [], []
            for i in range(max(1, len(closes)-14), len(closes)):
                delta = closes[i] - closes[i-1]
                gains.append(max(0, delta) / closes[i-1])
                losses.append(max(0, -delta) / closes[i-1])
            avg_gain = sum(gains) / max(1, len(gains))
            avg_loss = sum(losses) / max(1, len(losses))
            rsi = 100 if avg_loss == 0 and avg_gain > 0 else 50 if avg_gain == avg_loss == 0 else 100 - 100/(1+avg_gain/avg_loss)
            lines.append(f"نوسان ATR تقریبی ۱۴ کندل: {fmt_pct(atr_pct)}؛ RSI14 تقریبی: {rsi:.1f}/100")
        if len(volumes) >= 30:
            recent_avg = sum(volumes[-6:]) / 6
            baseline = sum(volumes[-30:-6]) / 24
            ratio = recent_avg / baseline if baseline > 0 else None
            lines.append(f"حجم ۶ کندل اخیر نسبت به میانگین ۲۴ کندل قبل: {ratio:.2f}x" if ratio is not None else "مقایسه حجم: حجم پایه صفر/ناموجود")
        else:
            lines.append("مقایسه حجم: تاریخچه حجمی کافی نیست")
    else:
        lines.append("تحلیل تکنیکال چندبازه‌ای: کندل معتبر کافی نیست")

    strategies = ensemble.get("strategies") or {}
    if strategies:
        lines.append("امتیاز استراتژی‌ها (از ۱۰۰، نه احتمال برد): " + " | ".join(
            f"{key}={num(value, 0):.0f}" for key, value in strategies.items()
        ))
    if ensemble.get("market_regime"):
        lines.append(f"رژیم بازار: {ensemble.get('market_regime')}؛ اجماع جهت‌دار: {num(ensemble.get('consensus_strength'), 0) * 100:.0f}%؛ اختلاف استراتژی‌ها: {num(ensemble.get('strategy_dispersion'), 0):.1f}")
    lines.append(
        f"اجماع استراتژی: {num(ensemble.get('score'), 0):.1f}/100؛ "
        f"استراتژی‌های صعودی: {ensemble.get('bullish_strategies', 'ناموجود')}؛ "
        f"پرچم‌های ریسک: {', '.join(ensemble.get('risk_flags') or []) or 'مورد مشخصی شناسایی نشد'}"
    )
    lines.append(
        f"کیفیت سیگنال: {quality.get('score', 'ناموجود')}/100 ({quality.get('grade', 'نامشخص')})؛ "
        f"قبولی فیلتر: {'بله' if quality.get('passed') else 'خیر/نامشخص'}"
    )
    if horizon_probabilities:
        lines.append("برآورد مدل برحسب افق: " + " | ".join(
            f"{k}={num(v, 0)*100:.1f}%" for k, v in horizon_probabilities.items()
            if num(v) is not None
        ))
        lines.append("توجه: برآورد مدل تضمین یا احتمال کالیبره‌شده سود نیست.")
    if entry and stop and target:
        risk_pct = (entry-stop)/entry*100 if entry > stop > 0 else None
        reward_pct = (target-entry)/entry*100 if target > entry > 0 else None
        rr = reward_pct/risk_pct if reward_pct is not None and risk_pct and risk_pct > 0 else None
        lines.append(
            f"طرح معامله: ورود ${entry:.8g}؛ ابطال/حدضرر ${stop:.8g}؛ هدف ${target:.8g}؛ "
            f"ریسک {fmt_pct(-risk_pct if risk_pct is not None else None)}؛ پاداش {fmt_pct(reward_pct)}؛ "
            f"نسبت پاداش/ریسک {f'{rr:.2f}:1' if rr is not None else 'نامشخص'}"
        )
    btc, sol = global_context.get("btc_24h_pct"), global_context.get("sol_24h_pct")
    lines.append(f"زمینه بازار: BTC ۲۴ساعت {fmt_pct(num(btc))}؛ SOL ۲۴ساعت {fmt_pct(num(sol))}؛ ریسک‌گریزی بازار: {'فعال' if global_context.get('risk_off') else 'فعال تشخیص داده نشد/نامعلوم'}")
    if news_context:
        lines.append("اخبار مرتبط اخیر (تیترمحور؛ نیازمند تأیید منبع اصلی):")
        for item in news_context[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "بدون عنوان")[:130]
            sentiment = str(item.get("sentiment") or "نامشخص")
            relevance = str(item.get("relevance") or "نامشخص")
            url = str(item.get("url") or "")
            lines.append(f"• [{sentiment}; {relevance}] {title}" + (f" — {url}" if url else ""))
    else:
        lines.append("اخبار مرتبط: مورد تازه‌ای در داده‌های نگهداری‌شده موجود نیست؛ این به معنی نبود خبر منفی نیست.")
    lines.append("محدودیت‌ها: سطوح از داده‌های تاریخی محاسبه شده‌اند؛ عمق دفتر سفارش، لغزش، کارمزد و اجرای واقعی ممکن است متفاوت باشد.")
    return "\n".join(lines)


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json_atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    temp.replace(path)


def read_jsonl(path):
    result = []
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


def write_jsonl_atomic(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def translate_telegram(message):
    """Translate fixed English Telegram labels and status text into Persian."""
    replacements = {
        "SOLANA BOT — LAST 24H PERFORMANCE": "گزارش عملکرد ربات سولانا — ۲۴ ساعت گذشته",
        "BUY-watch alerts recorded in last 24h:": "تعداد هشدارهای بررسی خرید در ۲۴ ساعت گذشته:",
        "Target/stop outcomes in last 24h:": "نتایج بررسی حد سود/ضرر در ۲۴ ساعت گذشته:",
        "wins:": "موفق:", "losses:": "ناموفق:", "timeouts:": "پایان زمان:", "data gaps:": "کمبود داده:",
        "Target-hit rate among target/stop outcomes:": "درصد رسیدن به حد سود:",
        "Average return of alerts resolved in last 24h:": "میانگین بازده نتایج بررسی‌شده در ۲۴ ساعت گذشته:",
        "Projects with usable market data this run:": "تعداد پروژه‌های دارای داده معتبر در این اجرا:",
        "Projects with usable market data:": "پروژه‌های دارای داده معتبر:",
        "Time (UTC):": "زمان (UTC):",
        "Win rate uses target/stop outcomes only; average return includes numeric resolved outcomes. Data gaps are excluded from average return. ": "درصد موفقیت فقط بر اساس برخورد به حد سود یا ضرر است؛ میانگین بازده شامل نتایج عددی است و کمبود داده محاسبه نمی‌شود. ",
        "This is not a guarantee of future performance.": "عملکرد گذشته تضمین‌کننده نتیجه آینده نیست.",
        "SOLANA BOT STATUS — RUNNING": "وضعیت ربات سولانا — فعال است",
        "The bot ran successfully, but the model is not ready yet, so no BUY alert was generated.": "ربات با موفقیت اجرا شد، اما مدل هنوز آماده نیست؛ بنابراین هشدار خریدی صادر نشد.",
        "Model details:": "جزئیات مدل:",
        "This is a status message, not a trading signal.": "این پیام فقط گزارش وضعیت است و سیگنال معاملاتی نیست.",
        "SOLANA BUY WATCH — NOT AN AUTOMATIC TRADE": "هشدار بررسی خرید سولانا — خرید خودکار انجام نمی‌شود",
        "SOLANA TOP-RANKED BUY WATCH — NOT AN AUTOMATIC TRADE": "هشدار جامع ستاپ برتر خرید سولانا — خرید خودکار انجام نمی‌شود",
        "SOLANA SELL / EXIT ALERT — NO AUTOMATIC TRADE": "هشدار فروش / خروج سولانا — معامله خودکار انجام نمی‌شود",
        "Exit reference:": "قیمت مرجع خروج:",
        "Data warning: this exit check uses cached hourly candles; latest candle age:": "هشدار داده: بررسی خروج بر اساس کندل‌های ساعتی ذخیره‌شده است؛ سن آخرین کندل:",
        "hours. Verify the current quote before acting.": " ساعت. پیش از اقدام، قیمت لحظه‌ای را بررسی کن.",
        "Estimated return before fees/slippage:": "بازده تخمینی پیش از کارمزد و لغزش قیمت:",
        "Reason:": "دلیل خروج:",
        "STOP LOSS reached": "حد ضرر لمس شد",
        "TAKE PROFIT reached": "حد سود لمس شد",
        "Trend breakdown below EMA21": "شکست روند و افت زیر EMA21",
        "14-day time limit reached": "رسیدن به سقف زمانی ۱۴ روز",
        "STOP (both levels touched; conservative)": "حد ضرر (هر دو سطح در یک کندل لمس شدند؛ فرض محافظه‌کارانه)",
        "This is an exit alert only; the bot does not place trades.": "این فقط هشدار خروج است و ربات معامله انجام نمی‌دهد.",
        "Global market context:": "وضعیت کلی بازار:",
        "Recent news and possible SOL impact:": "اخبار تازه و تأثیر احتمالی بر سولانا:",
        "BTC 24h change:": "تغییر ۲۴ ساعته بیت‌کوین:",
        "SOL 24h change:": "تغییر ۲۴ ساعته سولانا:",
        "BUY generation paused by global market risk filter:": "تولید هشدار خرید به‌دلیل ریسک کلی بازار متوقف شد:",
        "BTC 24h change": "تغییر ۲۴ ساعته بیت‌کوین",
        "SOL 24h change": "تغییر ۲۴ ساعته سولانا",
        "BTC 24h:": "تغییر ۲۴ ساعته بیت‌کوین:",
        "SOL 24h:": "تغییر ۲۴ ساعته سولانا:",
        "New BUY alerts paused:": "هشدارهای خرید جدید متوقف شده‌اند:",
        "Project:": "پروژه:", "Entry reference:": "قیمت مرجع ورود:",
        "Stop reference:": "حد ضرر مرجع:", "Target reference:": "حد سود مرجع:",
        "Model probability:": "احتمال پیش‌بینی مدل:",
        "Combined model probability:": "احتمال ترکیبی مدل:",
        "Probability by horizon:": "احتمال بر اساس افق زمانی:",
        "Model horizons available:": "افق‌های زمانی مدل:",
        "Price-action check:": "تأیید پرایس اکشن:",
        "Signal quality:": "کیفیت سیگنال:",
        "Quality checks:": "بررسی‌های کیفیت:",
        "Spot vs signal-candle deviation:": "فاصله قیمت لحظه‌ای از کندل سیگنال:",
        "Entry skipped for": "ورود رد شد برای", "Liquidity:": "نقدینگی:",
        "24h volume:": "حجم معاملات ۲۴ ساعته:", "Evaluation:": "ارزیابی مدل:",
        "Mint:": "آدرس توکن:",
        "This is a research alert, not financial advice. ": "این هشدار صرفاً برای بررسی است و توصیه مالی نیست. ",
        "Slippage, fees and execution can change results.": "لغزش قیمت، کارمزد و نحوه اجرا می‌توانند نتیجه را تغییر دهند.",
        "SOLANA NEWS IMPACT — RESEARCH, NOT A TRADE SIGNAL": "تحلیل خبر سولانا — برای بررسی، نه سیگنال معامله",
        "Headline:": "عنوان خبر:", "Source:": "منبع:", "Published (UTC):": "زمان انتشار (UTC):",
        "Meaning:": "مفهوم خبر:", "Possible SOL impact:": "تأثیر احتمالی بر سولانا:",
        "Direction:": "جهت احتمالی:", "Relevance:": "میزان ارتباط:",
        "Expected horizon:": "بازه اثر احتمالی:", "Confidence:": "میزان اطمینان:",
        "Why classified this way:": "دلیل این دسته‌بندی:", "Original link:": "پیوند منبع اصلی:",
        "Do not trade on a headline alone. Confirm the original report and check SOL price, volume, BTC direction and whether the news is already priced in.": "فقط بر اساس تیتر خبر معامله نکن. خبر اصلی را بررسی کن و قیمت و حجم سولانا، جهت بیت‌کوین و احتمال پیش‌خور شدن خبر را بسنج.",
        "A serious security, availability, legal or solvency risk may undermine confidence.": "ریسک جدی امنیتی، اختلال شبکه، حقوقی یا ناتوانی در پرداخت بدهی می‌تواند اعتماد بازار را کاهش دهد.",
        "The headline suggests a risk, delay, selling pressure or a negative catalyst.": "تیتر خبر به ریسک، تأخیر، فشار فروش یا عامل منفی اشاره دارد.",
        "If confirmed and material, this could improve access, demand or confidence.": "اگر خبر تأیید شود و اهمیت واقعی داشته باشد، ممکن است دسترسی، تقاضا یا اعتماد را افزایش دهد.",
        "This may support adoption, liquidity or sentiment, but implementation and scale matter.": "این خبر ممکن است به پذیرش، نقدینگی یا احساسات بازار کمک کند؛ اجرای واقعی و مقیاس آن مهم است.",
        "The headline alone does not provide a clear directional catalyst.": "تیتر به‌تنهایی عامل روشن و قابل اتکایی برای جهت قیمت نشان نمی‌دهد.",
        "Potentially relevant to SOL itself; verify official status and actual network impact.": "ممکن است مستقیماً به خود SOL مربوط باشد؛ وضعیت رسمی و اثر واقعی بر شبکه را بررسی کن.",
        "Could affect SOL demand, access or investor expectations if the report is accurate.": "اگر گزارش دقیق باشد، ممکن است بر تقاضا برای SOL، دسترسی یا انتظارات سرمایه‌گذاران اثر بگذارد.",
        "The first-order impact may be limited to that project; broader SOL impact depends on contagion, TVL and liquidity.": "اثر اولیه ممکن است محدود به همان پروژه باشد؛ اثر گسترده‌تر بر SOL به سرایت اثر، ارزش کل قفل‌شده و نقدینگی بستگی دارد.",
        "Do not assume this moves SOL; compare BTC, market-wide sentiment and SOL's own price/volume.": "فرض نکن این خبر حتماً قیمت SOL را جابه‌جا می‌کند؛ بیت‌کوین، احساسات کلی بازار و قیمت/حجم خود SOL را مقایسه کن.",
        "LOW — unconfirmed wording": "کم — خبر با عبارت‌های تأییدنشده بیان شده",
        "MEDIUM — aggregator headline; open original source": "متوسط — تیتر از گردآورنده خبر است؛ منبع اصلی را باز کن",
        "MEDIUM — headline classification only": "متوسط — دسته‌بندی فقط بر اساس تیتر خبر است",
        "hours to days; often priced in quickly": "چند ساعت تا چند روز؛ اثر خبر گاهی سریع در قیمت لحاظ می‌شود",
        "immediate to days; risk can persist if confirmed": "از فوری تا چند روز؛ در صورت تأیید، ریسک ممکن است ادامه یابد",
        "no reliable horizon until the facts and price reaction are checked": "تا بررسی واقعیت‌ها و واکنش قیمت، بازه قابل اتکایی مشخص نیست",
        "NEUTRAL / UNCLEAR": "خنثی / نامشخص", "BULLISH": "مثبت", "BEARISH": "منفی",
        "DIRECT — Solana network / protocol": "مستقیم — شبکه یا پروتکل سولانا",
        "DIRECT — SOL market / regulation": "مستقیم — بازار SOL یا مقررات",
        "INDIRECT — one Solana ecosystem project": "غیرمستقیم — یک پروژه از اکوسیستم سولانا",
        "UNCERTAIN — broader crypto or weak Solana link": "نامشخص — بازار کلی رمزارز یا ارتباط ضعیف با سولانا",
        "This is a headline-based heuristic, not full-article verification.": "این ارزیابی بر اساس تیتر و قواعد ساده است، نه بررسی کامل متن خبر.",
    }
    for english, persian in replacements.items():
        message = message.replace(english, persian)
    return message

def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets missing; message was not sent.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram sendMessage has a 4096-character limit. Split at line boundaries
    # so comprehensive research alerts are delivered rather than silently rejected.
    translated = translate_telegram(message)
    limit = 3600
    chunks = []
    remaining = translated
    while len(remaining) > limit:
        split_at = remaining.rfind("\\n", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    try:
        for index, chunk in enumerate(chunks, start=1):
            text = chunk if len(chunks) == 1 else f"[بخش {index}/{len(chunks)}]\\n{chunk}"
            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True
                },
                timeout=20
            )
            response.raise_for_status()
            if not response.json().get("ok"):
                return False
        return True
    except requests.RequestException as exc:
        print(f"Telegram error: {exc}")
        return False



def fetch_global_market_context():
    """Check BTC/SOL market-wide risk before allowing a new BUY watch."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    try:
        response = requests.get(
            url,
            params={
                "ids": "bitcoin,solana",
                "vs_currencies": "usd",
                "include_24hr_change": "true"
            },
            headers={"User-Agent": "SolanaAlertBot/1.0"},
            timeout=(4, 8)
        )
        response.raise_for_status()
        payload = response.json()
        btc = payload.get("bitcoin", {})
        sol = payload.get("solana", {})
        btc_change = float(btc["usd_24h_change"])
        sol_change = float(sol["usd_24h_change"])
        # Avoid new entries during sharp market-wide selloffs; existing positions
        # are still monitored and can generate EXIT alerts.
        risk_off = btc_change <= -3.0 or sol_change <= -5.0 or (
            btc_change <= -1.5 and sol_change <= -2.5
        )
        return {
            "available": True,
            "btc_24h_pct": round(btc_change, 2),
            "sol_24h_pct": round(sol_change, 2),
            "risk_off": bool(risk_off),
            "reason": (
                "Market-wide risk-off: pause new BUY alerts"
                if risk_off else "No extreme BTC/SOL 24h selloff detected"
            )
        }
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        # Do not falsely claim that global conditions were checked.
        print(f"Global market context unavailable: {exc}")
        return {
            "available": False, "btc_24h_pct": None,
            "sol_24h_pct": None, "risk_off": True,
            "reason": "Global market context unavailable; fail closed for new BUY alerts"
        }


def monitor_open_positions(state, market_info):
    """Send EXIT/SELL alerts at stop/target or clear trend failure; never trades."""
    positions = state.setdefault("open_positions", [])
    remaining = []
    for position in positions:
        address = position.get("token_address")
        info = market_info.get(address)
        if not info:
            remaining.append(position)
            continue
        candles = sorted(info.get("candles", []), key=lambda c: int(c["timestamp"]))
        entry = float(position.get("entry", 0) or 0)
        stop = float(position.get("stop", 0) or 0)
        target = float(position.get("target", 0) or 0)
        entry_ts = int(position.get("timestamp", 0) or 0)
        if entry <= 0 or stop <= 0 or target <= 0:
            continue

        exit_reason = None
        exit_price = None
        exit_ts = None
        # Re-scan every post-entry candle on every run. Providers may update the
        # current hourly candle during the hour; skipping last_checked_ts could
        # permanently miss a stop/target touched later in that same candle.
        for candle in candles:
            ts = int(candle["timestamp"])
            if ts <= entry_ts:
                continue
            low = float(candle["low"])
            high = float(candle["high"])
            # Conservative when OHLC cannot reveal which level was reached first.
            if low <= stop and high >= target:
                exit_reason, exit_price = "STOP (both levels touched; conservative)", stop
            elif low <= stop:
                exit_reason, exit_price = "STOP LOSS reached", stop
            elif high >= target:
                exit_reason, exit_price = "TAKE PROFIT reached", target
            if exit_reason:
                exit_ts = ts
                break

        latest = candles[-1] if candles else None
        if not exit_reason and latest:
            latest_ts = int(latest["timestamp"])
            current_close = float(latest["close"])
            # Apply the time exit even if indicator history is too short or fails.
            if latest_ts - entry_ts >= 14 * 24 * 3600:
                exit_reason, exit_price, exit_ts = "14-day time limit reached", current_close, latest_ts
            else:
                # If structure breaks after entry, signal exit instead of waiting blindly.
                try:
                    X, row = latest_features(candles)
                    if row is not None and latest_ts > entry_ts:
                        if float(row.get("price_vs_ema21", 0)) < 0 and float(row.get("ema_gap", 0)) < 0:
                            exit_reason, exit_price, exit_ts = "Trend breakdown below EMA21", current_close, latest_ts
                except Exception as exc:
                    print(f"Position trend check failed for {position.get('name')}: {exc}")
            position["last_checked_ts"] = latest_ts

        if exit_reason:
            pnl_pct = (float(exit_price) / entry - 1) * 100
            data_warning = ""
            if info.get("data_is_cached"):
                data_warning = (
                    "Data warning: this exit check uses cached hourly candles; "
                    f"latest candle age: {info.get('candle_age_hours', 'unknown')} "
                    "hours. Verify the current quote before acting.\n"
                )
            message = (
                "SOLANA SELL / EXIT ALERT — NO AUTOMATIC TRADE\n\n"
                f"Project: {position.get('name', 'Unknown')} ({position.get('symbol', '')})\n"
                f"Entry reference: ${entry:.8g}\n"
                f"Exit reference: ${float(exit_price):.8g}\n"
                f"Estimated return before fees/slippage: {pnl_pct:+.2f}%\n"
                f"Reason: {exit_reason}\n"
                f"Mint: {address}\n"
                f"{data_warning}"
                f"Time (UTC): {utc_now()}\n\n"
                "This is an exit alert only; the bot does not place trades."
            )
            sent = send_telegram(message)
            print(f"Exit alert {position.get('signal_id')}; telegram_sent={sent}; reason={exit_reason}")
            if not sent:
                remaining.append(position)
            else:
                append_jsonl(OUTCOMES_FILE, {
                    "signal_id": position.get("signal_id"),
                    "token_address": address,
                    "name": position.get("name", "Unknown"),
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "exit": float(exit_price),
                    "return_pct": round(pnl_pct, 4),
                    "result": "win" if pnl_pct > 0 else "loss",
                    "resolved_at": utc_now(),
                    "method": "position_exit_alert",
                    "exit_reason": exit_reason
                })
        else:
            remaining.append(position)
    state["open_positions"] = remaining


def load_candle_store():
    # Address -> timestamp -> candle
    store = {}
    for row in read_jsonl(CANDLES_FILE):
        address = row.get("token_address")
        candle = row.get("candle")
        if address and candle and "timestamp" in candle:
            store.setdefault(address, {})[int(candle["timestamp"])] = candle
    return store


def save_candle_store(store):
    rows = []
    for address, candles in store.items():
        ordered = sorted(candles.values(), key=lambda c: c["timestamp"])
        ordered = ordered[-MAX_CANDLE_HISTORY:]
        store[address] = {int(c["timestamp"]): c for c in ordered}
        for candle in ordered:
            rows.append({
                "token_address": address,
                "candle": candle
            })
    write_jsonl_atomic(CANDLES_FILE, rows)


def get_closed_candles(raw_candles):
    # Do not use the currently forming hourly candle.
    current_hour = int(datetime.now(timezone.utc).timestamp() // 3600) * 3600
    return [
        c for c in raw_candles
        if int(c["timestamp"]) < current_hour
    ]


def recent_cached_position_market_info(address, store, now=None, max_age_hours=3.0):
    """Return recent cached candles for exit monitoring only, never for fresh entries."""
    now = now or datetime.now(timezone.utc)
    current_hour = int(now.timestamp() // 3600) * 3600
    cached = sorted(
        (store.get(address) or {}).values(),
        key=lambda candle: int(candle.get("timestamp", 0))
    )
    if len(cached) < 35:
        return None
    latest_ts = int(cached[-1].get("timestamp", 0) or 0)
    if latest_ts <= 0:
        return None
    age_hours = (current_hour - latest_ts) / 3600
    if age_hours < 0 or age_hours > max_age_hours:
        return None
    try:
        price = float(cached[-1]["close"])
        if not math.isfinite(price) or price <= 0:
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return {
        "pair": None,
        "candles": cached,
        "price": price,
        "liquidity": 0.0,
        "volume24h": 0.0,
        "data_is_cached": True,
        "candle_age_hours": round(age_hours, 2),
    }


def select_projects_for_scan(projects, cursor=0, max_projects=5, always_include_addresses=None):
    """Rotate through the project universe while always monitoring open positions."""
    projects = list(projects or [])
    if not projects:
        return [], 0
    limit = max(1, int(max_projects))
    start = int(cursor or 0) % len(projects)
    count = min(limit, len(projects))
    selected = [projects[(start + offset) % len(projects)] for offset in range(count)]
    selected_addresses = {
        str(project.get("token_address", "")).strip()
        for project in selected if isinstance(project, dict)
    }
    always_include_addresses = set(always_include_addresses or [])
    for project in projects:
        address = str(project.get("token_address", "")).strip()
        if address in always_include_addresses and address not in selected_addresses:
            selected.append(project)
            selected_addresses.add(address)
    return selected, (start + count) % len(projects)


def resolve_old_signals(state, store, signal_key="open_signals", outcomes_path=OUTCOMES_FILE, count_live_stats=True):
    """Resolve four-hour paper observations, keeping live and shadow ledgers separate."""
    pending = state.setdefault(signal_key, [])
    still_pending = []
    known_outcomes = {
        row.get("signal_id") for row in read_jsonl(outcomes_path)
    }

    for signal in pending:
        signal_id = signal["signal_id"]
        if signal_id in known_outcomes:
            continue

        address = signal["token_address"]
        candles = sorted(
            store.get(address, {}).values(),
            key=lambda c: c["timestamp"]
        )
        signal_time = int(signal["timestamp"])
        future = [
            candle for candle in candles
            if signal_time < int(candle["timestamp"]) <= signal_time + 4 * 3600
        ]
        # Require four consecutive hourly candles. A missing candle must not
        # make the evaluator silently treat a shorter/irregular window as 4h.
        expected_times = [signal_time + hour * 3600 for hour in range(1, 5)]
        future_by_time = {int(candle["timestamp"]): candle for candle in future}
        complete_horizon = all(ts in future_by_time for ts in expected_times)
        if not complete_horizon:
            latest_ts = int(candles[-1]["timestamp"]) if candles else signal_time
            # Keep waiting while data may still arrive; after 8h, close the
            # paper observation as a data gap rather than leave it pending forever.
            if latest_ts < signal_time + 8 * 3600:
                still_pending.append(signal)
                continue
            append_jsonl(outcomes_path, {
                "signal_id": signal_id,
                "token_address": address,
                "name": signal.get("name", "Unknown"),
                "entry": float(signal.get("entry", 0) or 0),
                "result": "data_gap",
                "return_pct": None,
                "resolved_at": utc_now(),
                "method": "missing_hourly_candles",
                "source": signal.get("source", "live_alert" if count_live_stats else "paper_watchlist")
            })
            if count_live_stats:
                state["stats"]["resolved"] += 1
            continue
        future = [future_by_time[ts] for ts in expected_times]

        entry_price = float(signal["entry"])
        stop_price = float(signal.get("stop", entry_price * 0.98))
        target_price = float(signal.get("target", entry_price * 1.10))
        if entry_price <= 0:
            continue

        exit_price = None
        exit_candle = None
        exit_method = "close_after_4h"
        for candle in future[:4]:
            hit_stop = float(candle["low"]) <= stop_price
            hit_target = float(candle["high"]) >= target_price
            # Conservative assumption when OHLC data cannot reveal which level hit first.
            if hit_stop and hit_target:
                exit_price = stop_price
                exit_candle = candle
                exit_method = "both_hit_same_candle_assume_stop"
                break
            if hit_stop:
                exit_price = stop_price
                exit_candle = candle
                exit_method = "stop_hit"
                break
            if hit_target:
                exit_price = target_price
                exit_candle = candle
                exit_method = "target_hit"
                break

        if exit_price is None:
            exit_candle = future[3]
            exit_price = float(exit_candle["close"])

        ret = exit_price / entry_price - 1
        outcome = {
            "signal_id": signal_id,
            "token_address": address,
            "name": signal["name"],
            "entry": entry_price,
            "stop": stop_price,
            "target": target_price,
            "exit": exit_price,
            "return_pct": round(ret * 100, 4),
            # A win means the defined target was hit before the stop. If neither
            # level was hit within four candles, keep it as a timeout rather
            # than relabelling a small positive close as a target-hit win.
            "result": (
                "win" if exit_method == "target_hit"
                else "loss" if exit_method in ("stop_hit", "both_hit_same_candle_assume_stop")
                else "timeout"
            ),
            "exit_timestamp": int(exit_candle["timestamp"]),
            "resolved_at": utc_now(),
            "method": exit_method,
            "source": signal.get("source", "live_alert" if count_live_stats else "paper_watchlist")
        }
        append_jsonl(outcomes_path, outcome)

        if count_live_stats:
            state["stats"]["resolved"] += 1
            if outcome["result"] == "win":
                state["stats"]["wins"] += 1
            elif outcome["result"] == "loss":
                state["stats"]["losses"] += 1

    state[signal_key] = still_pending


def send_watchlist_fallback(active_projects, market_info, state, reason, configured_projects=None):
    """Send one top-ranked watchlist report per scan, even when no BUY setup qualifies."""
    now = datetime.now(timezone.utc)
    projects = list(active_projects or [])
    if not projects:
        projects = list(configured_projects or [])

    ranked = []
    for project in projects:
        address = str(project.get("token_address", "")).strip()
        info = market_info.get(address) if address else None
        if info and info.get("candles"):
            candles = sorted(info["candles"], key=lambda c: int(c.get("timestamp", 0)))
            ensemble = evaluate_strategies(candles)
            liquidity = float(info.get("liquidity") or 0)
            volume = float(info.get("volume24h") or 0)
            score = float(ensemble.get("score") or 0)
            score += min(8, math.log10(1 + liquidity / 10000) * 2)
            score += min(5, math.log10(1 + volume / 10000))
            score -= min(15, len(ensemble.get("risk_flags", [])) * 2)
            ranked.append((score, project, info, ensemble, candles))
        else:
            ranked.append((0.0, project, info, None, []))

    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        score, project, info, ensemble, candles = ranked[0]
        address = str(project.get("token_address", "")).strip()
        if info and candles:
            price_is_valid = bool(info.get("price_reference_valid", False))
            price = float(info.get("price") or 0) if price_is_valid else 0.0
            recent_lows = [
                float(c.get("low") or 0) for c in candles[-20:]
                if float(c.get("low") or 0) > 0
            ]
            stop = min(recent_lows) if recent_lows else 0.0
            if price > 0:
                has_valid_stop = 0 < stop < price
                risk_pct = (price - stop) / price * 100 if has_valid_stop else 0.0
                # Always show the valid current quote even if stop/target cannot be found.
                target = find_resistance_target(candles, price, 0.10)
                target_reward_pct = ((target / price) - 1.0) * 100 if target else 0.0
                reward_risk = target_reward_pct / risk_pct if target and risk_pct > 0 else 0.0
                stop_line = (
                    f"حد خروج/ابطال زیر کف ۲۰ کندل اخیر: ${stop:.8g} ({risk_pct:.2f}% پایین‌تر)\\n"
                    if has_valid_stop else
                    "حد خروج/ابطال: تعیین نشد؛ کف معتبر زیر قیمت فعلی در داده‌ها مشخص نیست.\\n"
                )
                target_line = (
                    f"هدف مرجع تکنیکال: ${target:.8g} ({target_reward_pct:.2f}% بالاتر"
                    + (f"؛ نسبت سود به ریسک تقریبی {reward_risk:.2f}:1" if risk_pct > 0 else "")
                    + ")\\n"
                    if target is not None else
                    "هدف مرجع: مقاومت قابل‌تشخیص در داده‌های فعلی پیدا نشد؛ قیمت مرجع همچنان معتبر است.\\n"
                )
                exit_lines = f"قیمت مرجع فعلی: ${price:.8g}\\n" + stop_line + target_line
                exit_warning = (
                    "این گزارش پروژهٔ برتر سیگنال خرید نیست؛ قیمت مرجع، قیمت نمایش‌داده‌شده توسط استخر است "
                    "و قیمت اجرای معامله را تضمین نمی‌کند.\\n"
                )
            else:
                deviation = info.get("price_deviation_pct")
                deviation_note = (
                    f" اختلاف قیمت لحظه‌ای و آخرین کندل: {deviation:.2f}٪."
                    if isinstance(deviation, (int, float)) and math.isfinite(deviation)
                    else ""
                )
                exit_lines = (
                    "قیمت مرجع فعلی در پاسخ API معتبر نیست؛ دادهٔ بازار ناکافی یا نامعتبر است."
                    + deviation_note + "\\n"
                )
                exit_warning = "تا دریافت قیمت مرجع معتبر، ورود نکن.\\n"
            message = (
                "SOLANA TOP-RANKED PROJECT — NOT A BUY SIGNAL\n\n"
                "بهترین پروژه قابل رتبه‌بندی این نوبت (نه الزاماً برنده آینده): "
                f"{project.get('name', 'Unknown')} ({project.get('symbol', '')})\n"
                f"رتبه امتیازی: {score:.1f}؛ امتیاز استراتژی: {ensemble.get('score', 0)}/100 (احتمال برد نیست)\n"
                f"نقدینگی: ${float(info.get('liquidity') or 0):,.0f}\n"
                f"حجم ۲۴ ساعته: ${float(info.get('volume24h') or 0):,.0f}\n"
                + exit_lines
                + build_comprehensive_analysis(
                    candles=candles, price=price, ensemble=ensemble,
                    global_context={}, liquidity=info.get("liquidity", 0),
                    volume24h=info.get("volume24h", 0)
                ) + "\n"
                + f"وضعیت ورود: ستاپ خرید از تمام فیلترها عبور نکرد؛ علت: {reason}\n"
                + f"ریسک‌ها: {', '.join(ensemble.get('risk_flags', [])) or 'ریسک مشخصی در این بررسی شناسایی نشد'}\n"
                + f"آدرس توکن: {address}\nزمان (UTC): {utc_now()}\n\n"
                + exit_warning
                + "رتبه‌بندی اجباری به معنی توصیه خرید نیست؛ در نبود ستاپ معتبر، اقدام نکردن هم نتیجه درست تحلیل است."
            )
        else:
            message = (
                "SOLANA TOP-RANKED PROJECT — DATA UNAVAILABLE\n\n"
                "یک پروژه از فهرست فعلی انتخاب شد، اما داده معتبر بازار برای تعیین قیمت/حد خروج در دسترس نیست.\n"
                f"پروژه: {project.get('name', 'Unknown')} ({project.get('symbol', '')})\n"
                f"آدرس توکن: {address or 'در دسترس نیست'}\n"
                f"علت: {reason}\nزمان (UTC): {utc_now()}\n\n"
                "قیمت و حد خروج ساخته نمی‌شوند؛ تا برگشت داده معتبر، معامله نکن."
            )
    else:
        message = (
            "SOLANA SCAN STATUS — NO PROJECT CAN BE RANKED\n\n"
            "در این اجرا پروژه تأییدشده‌ای برای رتبه‌بندی پیدا نشد؛ برای جلوگیری از ساختن نام یا قیمت جعلی، "
            "برنده‌ای اعلام نمی‌شود.\n"
            f"علت: {reason}\nزمان (UTC): {utc_now()}\n"
            "پس از برگشت داده/فهرست معتبر، اسکن بعدی دوباره تلاش می‌کند."
        )

    # Shadow-track the selected project independently of Telegram delivery.
    # This is a paper observation, not a BUY signal and never opens a real position.
    if ranked:
        paper_score, paper_project, paper_info, paper_ensemble, paper_candles = ranked[0]
        paper_address = str(paper_project.get("token_address", "")).strip()
        if paper_address and paper_info and paper_candles:
            ordered = sorted(paper_candles, key=lambda candle: int(candle.get("timestamp", 0)))
            last = ordered[-1]
            try:
                paper_ts = int(last["timestamp"])
                paper_entry = float(last.get("close") or 0)
                recent_lows = [float(item.get("low") or 0) for item in ordered[-20:]
                               if float(item.get("low") or 0) > 0]
                paper_stop = min(recent_lows) if recent_lows else 0.0
            except (TypeError, ValueError, KeyError, OverflowError):
                paper_ts, paper_entry, paper_stop = 0, 0.0, 0.0
            paper_signal_id = f"{paper_address}:{paper_ts}:PAPER_TOP"
            existing_ids = {str(item.get("signal_id")) for item in state.get("paper_open_signals", [])}
            # Keep completed or data-gap observations from being re-added on every 15m scan.
            existing_ids.update(
                str(item.get("signal_id")) for item in read_jsonl(PAPER_PREDICTIONS_FILE)
                if item.get("signal_id")
            )
            if paper_ts > 0 and paper_entry > 0 and 0 < paper_stop < paper_entry and paper_signal_id not in existing_ids:
                # One sample per token per completed hourly candle; repeated 15m scans do not duplicate it.
                paper_target = paper_entry + 2 * (paper_entry - paper_stop)
                paper_prediction = {
                    "signal_id": paper_signal_id,
                    "source": "paper_watchlist",
                    "timestamp": paper_ts,
                    "created_at": now.isoformat(),
                    "token_address": paper_address,
                    "name": paper_project.get("name", "Unknown"),
                    "symbol": paper_project.get("symbol", ""),
                    "entry": paper_entry,
                    "stop": paper_stop,
                    "target": paper_target,
                    "rank_score": round(float(paper_score), 3),
                    "strategy_score": float((paper_ensemble or {}).get("score") or 0),
                    "liquidity_usd": float(paper_info.get("liquidity") or 0),
                    "volume_24h_usd": float(paper_info.get("volume24h") or 0),
                    "risk_flags": list((paper_ensemble or {}).get("risk_flags", [])),
                    "model_probability": None,
                    "note": "Shadow observation only; no calibrated model probability and no real trade."
                }
                append_jsonl(PAPER_PREDICTIONS_FILE, paper_prediction)
                state.setdefault("paper_open_signals", []).append({
                    "signal_id": paper_signal_id,
                    "source": "paper_watchlist",
                    "token_address": paper_address,
                    "name": paper_project.get("name", "Unknown"),
                    "timestamp": paper_ts,
                    "entry": paper_entry,
                    "stop": paper_stop,
                    "target": paper_target
                })
                print(f"Paper observation recorded independently of Telegram: {paper_signal_id}")

    sent = send_telegram(message)
    print(f"Per-scan top-ranked project report sent={sent}; no cooldown applied.")
    if sent:
        state["last_watchlist_ping_utc"] = now.isoformat()
    return sent

def main():
    DATA.mkdir(parents=True, exist_ok=True)

    config = read_json(PROJECTS_FILE, {"projects": []})

    # Combine any manually configured projects with automatic discovery.
    # Manual entries take precedence for the same mint address.
    configured_projects = config.get("projects", [])
    discovered_projects = discover_solana_projects(
        min_liquidity=0,
        min_volume_24h=0,
        min_age_days=0,
        max_projects=int(os.getenv("MAX_DISCOVERED_PROJECTS", "500"))
    )
    combined = {}
    for project in discovered_projects:
        address = str(project.get("token_address", "")).strip()
        if address:
            combined[address] = project
    for project in configured_projects:
        address = str(project.get("token_address", "")).strip()
        if address:
            combined[address] = project
    # Hard safety boundary: configured projects cannot bypass the non-meme allowlist.
    candidates = list(combined.values())
    config["projects"] = [
        project for project in candidates
        if is_allowed_non_meme_project(project)
    ]
    rejected = len(candidates) - len(config["projects"])
    print(
        f"Project list: {len(configured_projects)} configured + "
        f"{len(discovered_projects)} discovered; "
        f"{len(config['projects'])} approved Solana utility/protocol candidates; "
        f"{rejected} rejected by non-meme allowlist."
    )

    state = read_json(STATE_FILE, {
        "last_alerts": {},
        "open_signals": [],
        "open_positions": [],
        "stats": {
            "signals": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0
        },
        "last_run_utc": None
    })

    state.setdefault("last_alerts", {})
    state.setdefault("open_signals", [])
    state.setdefault("paper_open_signals", [])
    state.setdefault("open_positions", [])
    state.setdefault("stats", {
        "signals": 0, "resolved": 0, "wins": 0, "losses": 0
    })

    # Keep monitoring existing positions even if discovery drops a token from
    # its top-ranked universe. The same strict non-meme allowlist still applies.
    known_addresses = {
        str(project.get("token_address", "")).strip()
        for project in config.get("projects", [])
    }
    for position in state.get("open_positions", []):
        address = str(position.get("token_address", "")).strip()
        candidate = {
            "name": position.get("name", ""),
            "symbol": position.get("symbol", ""),
            "token_address": address,
            "approved_non_meme": True
        }
        if address and address not in known_addresses and is_allowed_non_meme_project(candidate):
            config["projects"].append(candidate)
            known_addresses.add(address)

    open_position_addresses = {
        str(position.get("token_address", "")).strip()
        for position in state.get("open_positions", [])
    }

    # Analyze news internally; Telegram is reserved for BUY/SELL alerts.
    seen_news = state.setdefault("seen_news_ids", [])
    seen_news_set = set(seen_news)
    for item in fetch_recent_news()[:12]:
        if item["id"] in seen_news_set:
            continue
        analyzed = analyze_news(item)
        observations = state.setdefault("news_observations", [])
        observations.append({
            "observed_at": utc_now(),
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
        state["news_observations"] = observations[-500:]
        seen_news.append(item["id"])
        seen_news_set.add(item["id"])
        print(
            "News analysis:",
            json.dumps({
                "title": analyzed["title"],
                "source": analyzed.get("source", ""),
                "feed_category": analyzed.get("feed_category", ""),
                "sentiment": analyzed["sentiment"],
                "impact_score": analyzed["impact_score"],
                "relevance": analyzed["relevance"]
            }, ensure_ascii=False)
        )
    state["seen_news_ids"] = seen_news[-500:]
    state["last_news_refresh_utc"] = utc_now()

    # Adaptive policy learns cautiously from resolved paper outcomes and recent news.
    policy = derive_policy(
        outcomes=read_jsonl(OUTCOMES_FILE),
        news_observations=state.get("news_observations", []),
        base_probability=MIN_PROBABILITY,
        base_cooldown_hours=max(1, int(os.getenv("ALERT_COOLDOWN_HOURS", "6"))),
        research_observations=state.get("research_observations", [])
    )
    # Self-improvement can tune only an allowlisted probability floor after
    # chronological holdout validation; the user-configured floor and hard
    # safety gates remain authoritative.
    try:
        self_improvement = update_self_improvement(
            predictions=read_jsonl(PREDICTIONS_FILE),
            outcomes=read_jsonl(OUTCOMES_FILE),
            news_observations=state.get("news_observations", []),
            current_threshold=MIN_PROBABILITY
        )
        policy["min_probability"] = max(
            float(policy["min_probability"]),
            float(self_improvement["min_probability"])
        )
        policy["self_improvement"] = self_improvement
        print("Self-improvement:", json.dumps(self_improvement, ensure_ascii=False))
    except Exception as exc:
        # The bot must fall back to its existing conservative policy if learning fails.
        print(f"Self-improvement unavailable; using existing policy: {exc}")

    print("Adaptive policy:", json.dumps(policy, ensure_ascii=False))

    global_context = fetch_global_market_context()
    global_bearish = int(policy.get("global_bearish_news_24h", 0))
    global_bullish = int(policy.get("global_bullish_news_24h", 0))
    direct_bearish = int(policy.get("direct_bearish_news_24h", 0))
    if (global_bearish >= 3 and global_bearish > global_bullish) or direct_bearish >= 3:
        global_context["risk_off"] = True
        global_context["reason"] = (
            "Multiple bearish global or direct Solana news observations; pause new BUY alerts"
        )
    global_context["global_bearish_news_24h"] = global_bearish
    global_context["global_bullish_news_24h"] = global_bullish
    global_context["direct_bearish_news_24h"] = direct_bearish
    # Attach a short, recent headline rationale to BUY alerts, without sending
    # standalone news notifications to Telegram.
    recent_news_context = []
    news_now = datetime.now(timezone.utc)
    for item in reversed(state.get("news_observations", [])):
        published = item.get("published_at")
        try:
            published_dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        age_hours = (news_now - published_dt.astimezone(timezone.utc)).total_seconds() / 3600
        category = str(item.get("feed_category", ""))
        relevance = str(item.get("relevance", ""))
        sentiment = str(item.get("sentiment", ""))
        confidence = str(item.get("confidence", "")).upper()
        if confidence.startswith("LOW"):
            continue
        is_market_news = (
            category in ("Macro / global markets", "Geopolitics / global risk")
            or relevance.startswith("DIRECT")
        )
        if 0 <= age_hours <= 24 and is_market_news and sentiment in ("BULLISH", "BEARISH"):
            analysis_lines = str(item.get("analysis", "")).splitlines()
            impact_summary = ""
            if len(analysis_lines) > 1:
                impact_summary = analysis_lines[1].replace("Possible SOL impact:", "").strip()
            recent_news_context.append({
                "sentiment": sentiment,
                "impact_score": int(item.get("impact_score", 0)),
                "title": str(item.get("title", ""))[:130],
                "possible_sol_impact": impact_summary[:200],
                "url": str(item.get("url", ""))[:300]
            })
            if len(recent_news_context) >= 2:
                break
    recent_news_context.reverse()

    store = load_candle_store()
    active_projects = []
    market_info = {}
    scan_diagnostics = {
        "projects_seen": 0,
        "not_approved_or_missing_address": 0,
        "no_eligible_pair": 0,
        "insufficient_candles": 0,
        "stale_market_data": 0,
        "market_errors": 0,
        "usable_projects": 0,
        "price_action_rejected": 0,
        "invalid_features": 0,
        "below_probability_threshold": 0,
        "invalid_atr": 0,
        "alerts_created": 0
    }

    configured_universe = config.get("projects", [])
    scan_limit = max(1, int(os.getenv("MAX_PROJECTS_PER_SCAN", "5")))
    projects_to_scan, state["project_scan_cursor"] = select_projects_for_scan(
        configured_universe,
        cursor=state.get("project_scan_cursor", 0),
        max_projects=scan_limit,
        always_include_addresses=open_position_addresses
    )
    print(
        f"Rotating scan: checking {len(projects_to_scan)} of "
        f"{len(configured_universe)} approved projects this run; "
        f"configured limit={scan_limit}, open positions always included."
    )
    scan_diagnostics["projects_scheduled"] = len(projects_to_scan)
    scan_diagnostics["cached_fallbacks"] = 0

    for project in projects_to_scan:
        scan_diagnostics["projects_seen"] += 1
        if not is_allowed_non_meme_project(project):
            scan_diagnostics["not_approved_or_missing_address"] += 1
            continue

        address = str(project.get("token_address", "")).strip()
        if not address or address.startswith("REPLACE_"):
            scan_diagnostics["not_approved_or_missing_address"] += 1
            continue

        pair = None
        try:
            # Existing positions use a lower monitoring threshold so a
            # liquidity drop does not silently disable their exit checks.
            required_liquidity = (
                max(10000.0, MIN_LIQUIDITY * 0.10)
                if address in open_position_addresses else MIN_LIQUIDITY
            )
            pair = get_best_solana_pair(address, required_liquidity)
            if not pair:
                scan_diagnostics["no_eligible_pair"] += 1
                print(f"No eligible liquid Solana pair: {project.get('name')}")
                # Keep monitoring exits from a recent cached candle history if a
                # position is already open. Cached data is never used for new entries.
                if address in open_position_addresses:
                    cached_info = recent_cached_position_market_info(address, store)
                    if cached_info:
                        market_info[address] = cached_info
                        scan_diagnostics["cached_fallbacks"] += 1
                        print(
                            f"Exit-only cached fallback for {project.get('name')}; "
                            f"latest candle age={cached_info['candle_age_hours']:.2f}h."
                        )
                continue

            pool_address = pair.get("pairAddress")
            raw = get_hourly_candles(pool_address)
            candles = get_closed_candles(raw)
            if len(candles) < 35:
                scan_diagnostics["insufficient_candles"] += 1
                print(f"Not enough candles: {project.get('name')}")
                continue

            latest_candle_ts = int(candles[-1]["timestamp"])
            current_hour_ts = int(datetime.now(timezone.utc).timestamp() // 3600) * 3600
            candle_age_hours = (current_hour_ts - latest_candle_ts) / 3600
            if candle_age_hours > 3:
                scan_diagnostics["stale_market_data"] += 1
                print(
                    f"Stale hourly data for {project.get('name')}: "
                    f"latest candle is {candle_age_hours:.1f} hours old"
                )
                continue

            address_store = store.setdefault(address, {})
            for candle in candles:
                address_store[int(candle["timestamp"])] = candle

            merged = sorted(
                address_store.values(),
                key=lambda c: c["timestamp"]
            )[-MAX_CANDLE_HISTORY:]
            store[address] = {
                int(c["timestamp"]): c for c in merged
            }

            active_projects.append(project)
            scan_diagnostics["usable_projects"] += 1
            spot_price = float(pair.get("priceUsd") or 0)
            signal_close = float(candles[-1].get("close") or 0)
            price_deviation_pct = (
                abs(spot_price / signal_close - 1.0) * 100
                if spot_price > 0 and signal_close > 0 else float("inf")
            )
            # A current quote that disagrees materially with the latest closed
            # hourly candle is not safe to print as a verified reference price.
            # DexScreener's priceUsd is the current pool quote. A gap from the
            # last completed hourly close is market movement, not proof the quote
            # is stale. Keep the valid positive quote as the reference and use
            # the gap only to block a late BUY entry below.
            price_reference_valid = math.isfinite(spot_price) and spot_price > 0
            if math.isfinite(price_deviation_pct) and price_deviation_pct > 3.0:
                print(
                    f"Large spot/hourly-close move for {project.get('name')}: "
                    f"{price_deviation_pct:.2f}%; quote remains displayable, "
                    "but entry confirmation may reject it."
                )
            market_info[address] = {
                "pair": pair,
                "candles": merged,
                "price": spot_price if price_reference_valid else None,
                "raw_spot_price": spot_price,
                "price_reference_valid": price_reference_valid,
                "price_deviation_pct": round(price_deviation_pct, 3) if math.isfinite(price_deviation_pct) else None,
                "liquidity": float(
                    (pair.get("liquidity") or {}).get("usd") or 0
                ),
                "volume24h": float(
                    (pair.get("volume") or {}).get("h24") or 0
                )
            }

        except Exception as exc:
            scan_diagnostics["market_errors"] += 1
            # If GeckoTerminal is rate-limiting, reuse only a recent completed
            # candle history and a freshly fetched DexScreener quote. Never
            # fabricate a quote or use cached candles older than three hours.
            cached = sorted(
                store.get(address, {}).values(),
                key=lambda candle: int(candle.get("timestamp", 0))
            )
            cached_latest = int(cached[-1].get("timestamp", 0)) if cached else 0
            current_hour = int(datetime.now(timezone.utc).timestamp() // 3600) * 3600
            cached_age_hours = (current_hour - cached_latest) / 3600 if cached_latest else float("inf")
            if address in open_position_addresses:
                cached_position_info = recent_cached_position_market_info(address, store)
                if cached_position_info:
                    market_info[address] = cached_position_info
                    scan_diagnostics["cached_fallbacks"] += 1
                    print(
                        f"Exit-only cached fallback for {project.get('name')} after market-data "
                        f"error; latest candle age={cached_position_info['candle_age_hours']:.2f}h."
                    )
            if (
                pair is not None and len(cached) >= 65
                and 0 <= cached_age_hours <= 3
            ):
                spot_price = float(pair.get("priceUsd") or 0)
                cached_close = float(cached[-1].get("close") or 0)
                price_deviation_pct = (
                    abs(spot_price / cached_close - 1.0) * 100
                    if spot_price > 0 and cached_close > 0 else float("inf")
                )
                price_reference_valid = math.isfinite(spot_price) and spot_price > 0
                market_info[address] = {
                    "pair": pair,
                    "candles": cached,
                    "price": spot_price if price_reference_valid else None,
                    "raw_spot_price": spot_price,
                    "price_reference_valid": price_reference_valid,
                    "price_deviation_pct": round(price_deviation_pct, 3) if math.isfinite(price_deviation_pct) else None,
                    "liquidity": float((pair.get("liquidity") or {}).get("usd") or 0),
                    "volume24h": float((pair.get("volume") or {}).get("h24") or 0)
                }
                active_projects.append(project)
                scan_diagnostics["usable_projects"] += 1
                scan_diagnostics["cached_fallbacks"] += 1
                print(
                    f"Using recent cached candles for {project.get('name')} "
                    f"after market-data error; candle age={cached_age_hours:.1f}h."
                )
            elif address not in open_position_addresses or not market_info.get(address):
                print(f"Market data error for {project.get('name')}: {exc}")

    save_candle_store(store)

    # Resolve paper-performance labels, then monitor real-time exit conditions
    # for every BUY alert that was successfully delivered.
    resolve_old_signals(state, store)
    resolve_old_signals(
        state, store, signal_key="paper_open_signals",
        outcomes_path=PAPER_OUTCOMES_FILE, count_live_stats=False
    )
    monitor_open_positions(state, market_info)
    paper_status = update_paper_portfolio(state, market_info)
    print("Virtual portfolio:", json.dumps(paper_status, ensure_ascii=False))

    # Send a compact performance digest at most once every 24 hours.
    now = datetime.now(timezone.utc)
    last_report = state.get("last_stats_report_utc")
    report_due = not last_report
    if last_report:
        try:
            previous_report = datetime.fromisoformat(last_report.replace("Z", "+00:00"))
            report_due = (now - previous_report).total_seconds() >= 24 * 3600
        except (ValueError, TypeError):
            report_due = True

    if report_due:
        outcomes = read_jsonl(OUTCOMES_FILE)
        predictions = read_jsonl(PREDICTIONS_FILE)
        window_start = now.timestamp() - 24 * 3600

        def timestamp_in_window(value):
            if not value:
                return False
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp() >= window_start
            except (ValueError, TypeError, OverflowError):
                return False

        recent_predictions = [
            item for item in predictions
            if timestamp_in_window(item.get("created_at"))
        ]
        recent_outcomes = [
            item for item in outcomes
            if timestamp_in_window(item.get("resolved_at"))
        ]
        wins = sum(1 for item in recent_outcomes if item.get("result") == "win")
        losses = sum(1 for item in recent_outcomes if item.get("result") == "loss")
        timeouts = sum(1 for item in recent_outcomes if item.get("result") == "timeout")
        data_gaps = sum(1 for item in recent_outcomes if item.get("result") == "data_gap")
        resolved = wins + losses
        win_rate = (wins / resolved * 100) if resolved else 0.0
        returns = [
            float(item["return_pct"]) for item in recent_outcomes
            if isinstance(item.get("return_pct"), (int, float))
        ]
        avg_return = (sum(returns) / len(returns)) if returns else 0.0
        digest = (
            "SOLANA BOT — LAST 24H PERFORMANCE\n\n"
            f"BUY-watch alerts recorded in last 24h: {len(recent_predictions)}\n"
            f"Target/stop outcomes in last 24h: {resolved} (wins: {wins}, losses: {losses}); timeouts: {timeouts}; data gaps: {data_gaps}\n"
            f"Target-hit rate among target/stop outcomes: {win_rate:.1f}%\n"
            f"Average return of alerts resolved in last 24h: {avg_return:.2f}%\n"
            f"Projects with usable market data this run: {len(active_projects)}\n"
            f"Time (UTC): {utc_now()}\n\n"
            "Win rate uses target/stop outcomes only; average return includes numeric resolved outcomes. Data gaps are excluded from average return. "
            "This is not a guarantee of future performance."
        )
        print("Performance digest (log only; Telegram reserved for BUY/SELL):", digest)
        state["last_stats_report_utc"] = utc_now()

    # Train on all approved projects with persisted historical candles,
    # not just the small rotating subset fetched in this run.
    candle_sets = [
        sorted(store[address].values(), key=lambda candle: int(candle["timestamp"]))
        for address in {
            str(project.get("token_address", "")).strip()
            for project in config.get("projects", [])
        }
        if address in store and len(store[address]) >= 65
    ]

    model, metrics = train_model(candle_sets)
    print("Model evaluation:", json.dumps(metrics))
    print("Scan diagnostics before signals:", json.dumps(scan_diagnostics))

    if model is None:
        print("Model is not ready; no BUY alerts will be generated.")
        send_watchlist_fallback(
            active_projects, market_info, state,
            reason="ML model is not ready or lacks enough historical training data",
            configured_projects=config.get("projects", [])
        )
        print(
            "Model not ready. "
            f"Metrics={json.dumps(metrics)}; usable_projects={len(active_projects)}"
        )
        state["last_run_utc"] = utc_now()
        write_json_atomic(STATE_FILE, state)
        return

    if global_context.get("risk_off", True):
        print("BUY generation paused by global market risk filter:", global_context.get("reason"))
        scan_diagnostics["below_probability_threshold"] += len(active_projects)
    else:
      candidate_alerts = []
      for project in active_projects:
          address = project["token_address"]
          info = market_info.get(address)
          if not info:
              continue
          # Do not stack duplicate BUY recommendations while an exit-monitored
          # position for this mint is still open.
          if any(p.get("token_address") == address for p in state["open_positions"]):
              continue

          candles = info["candles"]
          X, row = latest_features(candles)
          if X is None or row is None:
              scan_diagnostics["invalid_features"] += 1
              continue

          # Use the last completed candle as the signal timestamp.
          candle_ts = int(row["timestamp"])
          signal_id = f"{address}:{candle_ts}:BUY_WATCH"

          last_alert_ts = int(state["last_alerts"].get(address, 0) or 0)
          cooldown_hours = policy["cooldown_hours"]
          if last_alert_ts and candle_ts - last_alert_ts < cooldown_hours * 3600:
              continue

          # Compare multiple strategy families; price action remains one input, not the only one.
          ensemble = evaluate_strategies(candles)
          if not ensemble.get("ready") or ensemble.get("score", 0) < 58 or ensemble.get("bullish_strategies", 0) < 2:
              scan_diagnostics["price_action_rejected"] += 1
              print(f"Strategy ensemble rejection {project.get('name')}: {json.dumps(ensemble)}")
              continue
          pa_ok, pa_reason = price_action_ok(candles, row)
          if not pa_ok:
              pa_reason = "Legacy PA disagrees; ensemble passed. " + str(pa_reason)

          probability = float(model.predict_proba(X)[0][1])
          horizon_probabilities = (
              model.predict_horizon_proba(X)
              if hasattr(model, "predict_horizon_proba") else {}
          )
          if probability < policy["min_probability"]:
              scan_diagnostics["below_probability_threshold"] += 1
              print(
                  f"Probability rejection {project.get('name')}: "
                  f"{probability:.3f} < adaptive {policy['min_probability']:.3f}"
              )
              continue

          # Demand agreement across independently trained horizons. A strong
          # ensemble average can otherwise hide a sharply bearish short horizon.
          horizon_values = {
              str(key): float(value)
              for key, value in horizon_probabilities.items()
              if isinstance(value, (int, float)) and math.isfinite(float(value))
          }
          short_horizon = horizon_values.get("4h")
          if short_horizon is not None and short_horizon < 0.50:
              scan_diagnostics["below_probability_threshold"] += 1
              print(
                  f"Horizon disagreement rejection {project.get('name')}: "
                  f"4h score={short_horizon:.3f} is below neutral"
              )
              continue
          if len(horizon_values) >= 2:
              horizon_consensus = sum(value >= 0.50 for value in horizon_values.values()) / len(horizon_values)
              if horizon_consensus < 0.60:
                  scan_diagnostics["below_probability_threshold"] += 1
                  print(
                      f"Horizon consensus rejection {project.get('name')}: "
                      f"{horizon_consensus:.0%} horizons are neutral/bullish"
                  )
                  continue

          signal_close = float(row["close"])
          atr_pct = float(row["atr_pct"])
          if not math.isfinite(atr_pct) or atr_pct <= 0:
              scan_diagnostics["invalid_atr"] += 1
              continue

          # Never substitute a candle close for a missing/discordant current quote.
          # A missing or rejected spot reference means no new BUY alert.
          if not info.get("price_reference_valid", False) or not info.get("price"):
              scan_diagnostics["invalid_price_reference"] += 1
              print(f"Entry skipped for {project.get('name')}: current price reference failed validation")
              continue
          entry = float(info["price"])
          entry_deviation_pct = abs(entry / signal_close - 1.0) * 100
          max_entry_deviation_pct = max(1.5, min(3.0, 1.5 * atr_pct * 100))
          if entry <= 0 or entry_deviation_pct > max_entry_deviation_pct:
              print(
                  f"Entry skipped for {project.get('name')}: spot moved "
                  f"{entry_deviation_pct:.2f}% from signal close "
                  f"(limit {max_entry_deviation_pct:.2f}%)"
              )
              continue

          stop_pct = min(MAX_STOP_PCT, max(0.02, 1.5 * atr_pct))
          # Require a stop distance no greater than the configured cap.
          if stop_pct > MAX_STOP_PCT:
              continue

          stop = entry * (1 - stop_pct)
          minimum_target_pct = max(0.10, TARGET_PCT, 3 * atr_pct, 2.0 * stop_pct)
          target = find_resistance_target(candles, entry, minimum_target_pct)
          if target is None:
              scan_diagnostics.setdefault("no_valid_resistance_target", 0)
              scan_diagnostics["no_valid_resistance_target"] += 1
              print(
                  f"Entry skipped for {project.get('name')}: no confirmed resistance "
                  f"supports required target >= {minimum_target_pct * 100:.2f}% "
                  "within 30% and reward/risk constraints"
              )
              continue
          target_pct = target / entry - 1.0

          quality = score_signal(
              row=row,
              candles=candles,
              liquidity_usd=info["liquidity"],
              volume_24h_usd=info["volume24h"],
              global_context=global_context,
              stop_pct=stop_pct,
              target_pct=target_pct
          )
          if not quality["passed"]:
              scan_diagnostics.setdefault("quality_rejected", 0)
              scan_diagnostics["quality_rejected"] += 1
              print(
                  f"Signal-quality rejection {project.get('name')}: "
                  f"{quality['score']}/100 < {quality['minimum_required']}"
              )
              continue

          # Experience-derived strategy rules are learned from resolved outcomes.
          # They can reject a weak setup, but cannot alter network or data-integrity gates.
          learned_rule_result = evaluate_experience_rules({
              "entry_deviation_pct": entry_deviation_pct,
              "liquidity_usd": info["liquidity"],
              "volume_24h_usd": info["volume24h"],
              "ensemble_score": float(ensemble.get("score", 0) or 0),
              "probability": probability,
              "quality_score": float(quality.get("score", 0) or 0)
          }, (policy.get("self_improvement") or {}).get("experience_rules", []))
          if learned_rule_result.get("blocked"):
              scan_diagnostics.setdefault("experience_rule_rejected", 0)
              scan_diagnostics["experience_rule_rejected"] += 1
              print(
                  f"Experience rule rejected {project.get('name')}: "
                  f"{learned_rule_result.get('rule_id')} — "
                  f"{learned_rule_result.get('reason')}; "
                  f"evidence={json.dumps(learned_rule_result.get('evidence', {}))}"
              )
              continue

          compact_metrics = {
              "active_horizons": metrics.get("active_horizons", []),
              "horizons": {
                  horizon_name: {
                      key: horizon_data.get(key)
                      for key in ("ready", "labelled_examples", "balanced_accuracy", "brier_score")
                  }
                  for horizon_name, horizon_data in metrics.get("horizons", {}).items()
              }
          }

          prediction = {
              "signal_id": signal_id,
              "timestamp": candle_ts,
              "created_at": utc_now(),
              "token_address": address,
              "name": project.get("name", "Unknown"),
              "symbol": project.get("symbol", ""),
              "entry": entry,
              "stop": stop,
              "target": target,
              "stop_pct": round(stop_pct * 100, 3),
              "target_pct": round(target_pct * 100, 3),
              "probability": round(probability, 4),
              "horizon_probabilities": horizon_probabilities,
              "recent_news_context": recent_news_context,
              "entry_deviation_pct": round(entry_deviation_pct, 3),
              "liquidity_usd": info["liquidity"],
              "volume_24h_usd": info["volume24h"],
              "price_action": pa_reason,
              "strategy_ensemble": ensemble,
              "rank_score": rank_candidate(
                  ensemble, probability, quality["score"],
                  liquidity_usd=info["liquidity"],
                  volume_24h_usd=info["volume24h"],
                  entry_deviation_pct=entry_deviation_pct
              ),
              "signal_quality": quality,
              "model_metrics": metrics
          }

          candidate_alerts.append({
              "prediction": prediction,
              "info": info,
              "ensemble": ensemble,
              "quality": quality,
              "probability": probability,
              "entry": entry,
              "stop": stop,
              "target": target,
              "stop_pct": stop_pct,
              "target_pct": target_pct,
              "entry_deviation_pct": entry_deviation_pct,
              "pa_reason": pa_reason,
              "horizon_probabilities": horizon_probabilities,
              "compact_metrics": compact_metrics,
          })


      # Alert only the highest-ranked qualified candidate; never force a trade.
      if candidate_alerts:
          candidate_alerts.sort(key=lambda item: item["prediction"].get("rank_score", 0), reverse=True)
          chosen = candidate_alerts[0]
          prediction = chosen["prediction"]
          paper_open_result = open_paper_trade(state, prediction)
          print("Virtual trade decision:", json.dumps(paper_open_result, ensure_ascii=False))
          info = chosen["info"]
          ensemble = chosen["ensemble"]
          quality = chosen["quality"]
          probability = chosen["probability"]
          entry = chosen["entry"]
          stop = chosen["stop"]
          target = chosen["target"]
          stop_pct = chosen["stop_pct"]
          target_pct = chosen["target_pct"]
          entry_deviation_pct = chosen["entry_deviation_pct"]
          pa_reason = chosen["pa_reason"]
          horizon_probabilities = chosen["horizon_probabilities"]
          compact_metrics = chosen["compact_metrics"]
          message = (
              "SOLANA TOP-RANKED BUY WATCH — NOT AN AUTOMATIC TRADE\n\n"
              f"Ranked #1 of {len(candidate_alerts)} qualified candidates\n"
              f"Project: {prediction['name']} ({prediction['symbol']})\n"
              f"Entry reference: ${entry:.8g}\n"
              f"Stop reference: ${stop:.8g} ({stop_pct * 100:.2f}%)\n"
              f"Target reference: ${target:.8g} ({target_pct * 100:.2f}%)\n"
              f"Combined model probability: {probability:.1%}\n"
              f"Ensemble rank score: {prediction['rank_score']:.1f}/100\n"
              f"Strategy ensemble: {json.dumps(ensemble, ensure_ascii=False)}\n"
              f"Risk flags: {', '.join(ensemble.get('risk_flags', [])) or 'none detected'}\n"
              f"Probability by horizon: {json.dumps(horizon_probabilities)}\n"
              f"Price-action note: {pa_reason}\n"
              f"Signal quality: {quality['score']}/100 (grade {quality['grade']})\n"
              f"Quality checks: {', '.join(reason for reason in quality['reasons'] if reason)}\n"
              f"Spot vs signal-candle deviation: {entry_deviation_pct:.2f}%\n"
              f"Liquidity: ${info['liquidity']:,.0f}\n"
              f"24h volume: ${info['volume24h']:,.0f}\n"
              f"BTC 24h change: {global_context['btc_24h_pct']}%; SOL 24h change: {global_context['sol_24h_pct']}%\n"
              f"Recent news and possible SOL impact: {json.dumps(recent_news_context, ensure_ascii=False)}\n"
              f"Evaluation: {json.dumps(compact_metrics)}\n"
              + build_comprehensive_analysis(
                  candles=info.get("candles", []), price=entry,
                  ensemble=ensemble, quality=quality,
                  horizon_probabilities=horizon_probabilities,
                  global_context=global_context, news_context=recent_news_context,
                  liquidity=info.get("liquidity", 0), volume24h=info.get("volume24h", 0),
                  entry=entry, stop=stop, target=target
              ) + "\n"
              f"Mint: {prediction['token_address']}\n"
              f"Time (UTC): {utc_now()}\n\n"
              "Research alert only. Rank is not a guarantee; fees, slippage and losses can change results."
          )
          sent = send_telegram(message)
          print(f"Top-ranked candidate {prediction['signal_id']}; rank={prediction['rank_score']:.1f}; telegram_sent={sent}")
          if sent:
              scan_diagnostics["alerts_created"] += 1
              append_jsonl(PREDICTIONS_FILE, prediction)
              state["open_signals"].append({
                  "signal_id": prediction["signal_id"], "token_address": prediction["token_address"],
                  "name": prediction["name"], "timestamp": prediction["timestamp"],
                  "entry": entry, "stop": stop, "target": target
              })
              if not any(p.get("token_address") == prediction["token_address"] for p in state["open_positions"]):
                  state["open_positions"].append({
                      "signal_id": prediction["signal_id"], "token_address": prediction["token_address"],
                      "name": prediction["name"], "symbol": prediction["symbol"],
                      "timestamp": prediction["timestamp"], "last_checked_ts": prediction["timestamp"],
                      "entry": entry, "stop": stop, "target": target
                  })
              state["last_alerts"][prediction["token_address"]] = prediction["timestamp"]
              state["stats"]["signals"] += 1
          else:
              print(f"Telegram delivery failed; top candidate will be retried: {prediction['signal_id']}")
    if scan_diagnostics["alerts_created"] == 0:
        reason = (
            "global market/news risk filter paused entries"
            if global_context.get("risk_off", True)
            else "no candidate passed every entry filter this run"
        )
        send_watchlist_fallback(
            active_projects, market_info, state,
            reason=reason,
            configured_projects=config.get("projects", [])
        )
    print("Final scan diagnostics:", json.dumps(scan_diagnostics))
    if scan_diagnostics["alerts_created"] == 0:
        print(
            "No BUY-watch alert this run. This can be normal; inspect the "
            "diagnostic counters above to identify the strictest filter."
        )

    state["last_run_utc"] = utc_now()
    write_json_atomic(STATE_FILE, state)


if __name__ == "__main__":
    main()
