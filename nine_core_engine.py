"""Nine diverse, independently scored paper-trading cores.

These are deterministic rule-based voters, not nine independent AI models.
Each core has its own $10,000 virtual ledger and can choose a different
qualified candidate. A Telegram BUY-WATCH candidate requires at least 5/9 votes.
"""
from datetime import datetime, timezone
import math

from paper_portfolio import get_portfolio, open_paper_trade, update_paper_portfolio

CORE_SPECS = [
    {"id": "trend", "name": "Trend follower", "threshold": 58},
    {"id": "breakout", "name": "Breakout hunter", "threshold": 58},
    {"id": "momentum", "name": "Momentum analyst", "threshold": 60},
    {"id": "volume", "name": "Volume/flow analyst", "threshold": 58},
    {"id": "structure", "name": "Market-structure analyst", "threshold": 58},
    {"id": "risk", "name": "Capital-preservation analyst", "threshold": 65},
    {"id": "model", "name": "Model-confidence analyst", "threshold": 60},
    {"id": "quality", "name": "Signal-quality analyst", "threshold": 60},
    {"id": "context", "name": "News/regime analyst", "threshold": 60},
]

SHARED_DATA_SOURCES = [
    {"name": "DexScreener", "purpose": "Solana token discovery, pair price, liquidity and volume"},
    {"name": "GeckoTerminal", "purpose": "Hourly OHLCV candles"},
    {"name": "Google News RSS", "purpose": "Ecosystem, security, macro and geopolitical headlines"},
    {"name": "Strategy ensemble", "purpose": "Trend, breakout, momentum, volume-flow, structure and risk scores"},
    {"name": "Learner and signal-quality modules", "purpose": "Model estimates, validation metrics and signal quality"},
]
CORE_TOOLSETS = {
    "trend": ["OHLCV", "trend score", "structure score", "momentum score", "model probability"],
    "breakout": ["OHLCV", "breakout score", "volume-flow score", "structure score"],
    "momentum": ["OHLCV", "momentum score", "trend score", "volume-flow score"],
    "volume": ["pair volume/liquidity", "volume-flow score", "breakout score", "momentum score"],
    "structure": ["OHLCV", "structure score", "trend score", "candle-rejection score"],
    "risk": ["liquidity/volume", "risk-control score", "signal quality", "stop/target", "risk flags"],
    "model": ["learner probability", "rank score", "model validation metrics when available"],
    "quality": ["signal quality", "risk-control score", "probability", "stop/target"],
    "context": ["recent news when available", "market regime", "trend/risk scores", "risk flags"],
}

STARTING_CASH_PER_CORE = 10_000.0
MAJORITY_VOTES = 5

MIN_SUCCESS_TRADES = 8
MIN_ELIMINATION_TRADES = 12
MIN_WIN_RATE_FOR_SUCCESS = 0.55
MAX_WIN_RATE_FOR_ELIMINATION = 0.45


def _closed_trade_metrics(core, genome):
    trades = (core.get("portfolio") or {}).get("closed_trades") or []
    start = int(genome.get("generation_start_closed_count", 0) or 0)
    rows = [t for t in trades[start:] if isinstance(t, dict) and isinstance(t.get("pnl_usd"), (int, float))]
    pnl = sum(float(t["pnl_usd"]) for t in rows)
    wins = sum(float(t["pnl_usd"]) > 0 for t in rows)
    count = len(rows)
    return {"trades": count, "pnl_usd": round(pnl, 4), "win_rate": round(wins / count, 4) if count else 0.0,
            "return_pct": round(pnl / STARTING_CASH_PER_CORE * 100, 4)}


def _initial_genome(spec):
    return {"profile_id": spec["id"], "threshold": spec["threshold"], "risk_multiplier": 1.0,
            "generation": 0, "parent_id": None, "generation_start_closed_count": 0,
            "last_evaluated_closed_count": 0, "bad_streak": 0, "status": "founder"}


def evolve_core_population(system):
    """Replace persistently weak strategy genomes with mutated offspring of paper winners."""
    cores = system.get("cores", {})
    for spec in CORE_SPECS:
        core = cores.get(spec["id"])
        if isinstance(core, dict) and not isinstance(core.get("genome"), dict):
            core["genome"] = _initial_genome(spec)
    successful, weak, metrics_by_id = [], [], {}
    for spec in CORE_SPECS:
        cid = spec["id"]
        core = cores.get(cid) or {}
        genome = core.get("genome") or _initial_genome(spec)
        metrics = _closed_trade_metrics(core, genome)
        metrics_by_id[cid] = metrics
        total_closed = len((core.get("portfolio") or {}).get("closed_trades") or [])
        has_new_close = total_closed > int(genome.get("last_evaluated_closed_count", 0) or 0)
        if metrics["trades"] >= MIN_SUCCESS_TRADES and metrics["pnl_usd"] > 0 and metrics["win_rate"] >= MIN_WIN_RATE_FOR_SUCCESS:
            successful.append((metrics["return_pct"], cid))
        if not has_new_close:
            continue
        genome["last_evaluated_closed_count"] = total_closed
        if metrics["trades"] >= MIN_ELIMINATION_TRADES:
            if metrics["pnl_usd"] <= 0 and metrics["win_rate"] < MAX_WIN_RATE_FOR_ELIMINATION:
                genome["bad_streak"] = int(genome.get("bad_streak", 0)) + 1
            else:
                genome["bad_streak"] = 0
            if genome["bad_streak"] >= 2:
                weak.append((metrics["return_pct"], cid))
    events = system.setdefault("evolution_events", [])
    successful.sort(reverse=True)
    weak.sort()
    replacements, used_parents = [], set()
    for _, weak_id in weak:
        parents = [item for item in successful if item[1] != weak_id and item[1] not in used_parents]
        if not parents:
            break
        _, parent_id = parents[0]
        target = cores[weak_id]
        old_genome = dict(target["genome"])
        parent_genome = cores[parent_id]["genome"]
        generation = int(old_genome.get("generation", 0)) + 1
        role_ids = [item["id"] for item in CORE_SPECS]
        parent_profile = parent_genome.get("profile_id", parent_id)
        parent_index = role_ids.index(parent_profile) if parent_profile in role_ids else role_ids.index(parent_id)
        child_profile = role_ids[(parent_index + generation + 1) % len(role_ids)]
        child_threshold = max(52, min(72, _num(parent_genome.get("threshold"), 60) + (-2 if generation % 2 else 2)))
        child_risk = max(0.8, min(1.25, _num(parent_genome.get("risk_multiplier"), 1.0) + (-0.05 if generation % 2 else 0.05)))
        closed_count = len((target.get("portfolio") or {}).get("closed_trades") or [])
        target["genome"] = {"profile_id": child_profile, "threshold": child_threshold,
            "risk_multiplier": round(child_risk, 3), "generation": generation, "parent_id": parent_id,
            "generation_start_closed_count": closed_count, "last_evaluated_closed_count": closed_count,
            "bad_streak": 0, "status": "offspring"}
        event = {"type": "offspring_replaced_weak_genome", "slot": weak_id, "parent_id": parent_id,
            "generation": generation, "old_profile": old_genome.get("profile_id", weak_id),
            "new_profile": child_profile, "threshold": child_threshold, "risk_multiplier": round(child_risk, 3),
            "parent_metrics": metrics_by_id.get(parent_id, {})}
        events.append(event)
        replacements.append(event)
        used_parents.add(parent_id)
        successful = [item for item in successful if item[1] != parent_id]
    system["evolution_events"] = events[-100:]
    system["last_evolution"] = {"checked_at": datetime.now(timezone.utc).isoformat(),
        "replacements": replacements, "active_offspring": sum(1 for c in cores.values()
        if isinstance(c, dict) and (c.get("genome") or {}).get("status") == "offspring")}
    return system["last_evolution"]


def _num(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _clamp(x):
    return max(0.0, min(100.0, x))


def _strategy(ensemble, key):
    return _num((ensemble.get("strategies") or {}).get(key), 50.0)


def _risk_penalty(candidate):
    pred = candidate.get("prediction") or {}
    ensemble = pred.get("strategy_ensemble") or candidate.get("ensemble") or {}
    flags = ensemble.get("risk_flags") or []
    return min(35.0, len(flags) * 7.0)


def score_for_core(core_id, candidate, genome=None):
    """Score a candidate using the active genome's inherited analytical profile."""
    genome = genome or {}
    profile_id = genome.get("profile_id", core_id)
    pred = candidate.get("prediction") or {}
    ensemble = pred.get("strategy_ensemble") or candidate.get("ensemble") or {}
    quality = pred.get("signal_quality") or candidate.get("quality") or {}
    strategies = {
        "trend": _strategy(ensemble, "trend_following"),
        "breakout": _strategy(ensemble, "breakout"),
        "momentum": _strategy(ensemble, "momentum"),
        "volume": _strategy(ensemble, "volume_flow"),
        "structure": _strategy(ensemble, "market_structure"),
        "risk": _strategy(ensemble, "risk_control"),
        "reversion": _strategy(ensemble, "mean_reversion"),
        "rejection": _strategy(ensemble, "candle_rejection"),
    }
    probability = _clamp(_num(pred.get("probability"), 0.0) * 100)
    quality_score = _clamp(_num(quality.get("score"), 0.0))
    liquidity = max(0.0, _num(pred.get("liquidity_usd"), _num(candidate.get("info", {}).get("liquidity"))))
    volume = max(0.0, _num(pred.get("volume_24h_usd"), _num(candidate.get("info", {}).get("volume24h"))))
    entry_deviation = max(0.0, _num(pred.get("entry_deviation_pct"), _num(candidate.get("entry_deviation_pct"))))
    stop_pct = max(0.0, _num(pred.get("stop_pct"), _num(candidate.get("stop_pct")) * 100))
    target_pct = max(0.0, _num(pred.get("target_pct"), _num(candidate.get("target_pct")) * 100))
    reward_risk = target_pct / stop_pct if stop_pct > 0 else 0.0
    flags_penalty = _risk_penalty(candidate) * max(0.5, min(1.5, _num(genome.get("risk_multiplier"), 1.0)))

    if profile_id == "trend":
        score = .42*strategies["trend"] + .20*strategies["structure"] + .18*strategies["momentum"] + .20*probability
    elif profile_id == "breakout":
        score = .42*strategies["breakout"] + .23*strategies["volume"] + .20*strategies["structure"] + .15*probability
    elif profile_id == "momentum":
        score = .42*strategies["momentum"] + .25*strategies["trend"] + .18*strategies["volume"] + .15*probability
    elif profile_id == "volume":
        score = .48*strategies["volume"] + .22*strategies["breakout"] + .15*strategies["momentum"] + .15*probability
    elif profile_id == "structure":
        score = .42*strategies["structure"] + .23*strategies["trend"] + .20*strategies["rejection"] + .15*probability
    elif profile_id == "risk":
        score = .45*strategies["risk"] + .25*quality_score + .15*min(100, reward_risk*35) + .15*probability - flags_penalty
        if stop_pct > 5 or entry_deviation > 2.5:
            score -= 15
    elif profile_id == "model":
        score = .58*probability + .22*quality_score + .20*_num(pred.get("rank_score"))
        metrics = pred.get("model_metrics") or {}
        horizons = metrics.get("horizons") or {}
        ready = [v for v in horizons.values() if isinstance(v, dict) and v.get("ready")]
        if ready and all(_num(v.get("balanced_accuracy"), 0.5) < 0.52 for v in ready):
            score -= 12
    elif profile_id == "quality":
        score = .58*quality_score + .22*strategies["risk"] + .20*probability
        if reward_risk < 1.5:
            score -= 12
    else:  # news/regime
        score = .30*probability + .25*strategies["trend"] + .20*strategies["risk"] + .25*quality_score
        news = pred.get("recent_news_context") or []
        negative = sum(1 for n in news if isinstance(n, dict) and
                       str(n.get("sentiment", "")).lower() in {"negative", "risk", "bearish"})
        positive = sum(1 for n in news if isinstance(n, dict) and
                       str(n.get("sentiment", "")).lower() in {"positive", "bullish"})
        score += min(8, positive*2) - min(18, negative*6)
        if ensemble.get("market_regime") in {"high_volatility", "downtrend"}:
            score -= 8

    if liquidity < 100_000:
        score -= 25
    if volume < 100_000:
        score -= 12
    if entry_deviation > 3:
        score -= 15
    return round(_clamp(score), 2)


def deliberate_between_cores(candidate, scores):
    """Structured peer discussion using computed evidence; no external LLM is implied."""
    pred = candidate.get("prediction") or {}
    ensemble = pred.get("strategy_ensemble") or candidate.get("ensemble") or {}
    quality = pred.get("signal_quality") or candidate.get("quality") or {}
    strategies = ensemble.get("strategies") or {}
    flags = ensemble.get("risk_flags") or []
    stop_pct = max(0.0, _num(pred.get("stop_pct"), _num(candidate.get("stop_pct")) * 100))
    entry_deviation = max(0.0, _num(pred.get("entry_deviation_pct"), _num(candidate.get("entry_deviation_pct"))))
    liquidity = max(0.0, _num(pred.get("liquidity_usd"), _num(candidate.get("info", {}).get("liquidity"))))
    volume = max(0.0, _num(pred.get("volume_24h_usd"), _num(candidate.get("info", {}).get("volume24h"))))
    memos = {}
    for spec in CORE_SPECS:
        cid = spec["id"]
        own = scores.get(cid, 0.0)
        if cid == "trend":
            evidence = "trend=" + str(_num(strategies.get("trend_following"), 0)) + ", structure=" + str(_num(strategies.get("market_structure"), 0))
            question = "آیا روند و ساختار قیمت هم‌جهت‌اند؟"
        elif cid == "breakout":
            evidence = "breakout=" + str(_num(strategies.get("breakout"), 0)) + ", volume-flow=" + str(_num(strategies.get("volume_flow"), 0))
            question = "آیا شکست با حجم تأیید شده است؟"
        elif cid == "momentum":
            evidence = "momentum=" + str(_num(strategies.get("momentum"), 0)) + ", trend=" + str(_num(strategies.get("trend_following"), 0))
            question = "آیا حرکت قیمت قدرت و تداوم دارد؟"
        elif cid == "volume":
            evidence = "volume24h=" + str(round(volume)) + ", liquidity=" + str(round(liquidity))
            question = "آیا حجم و نقدینگی برای ورود کافی‌اند؟"
        elif cid == "structure":
            evidence = "structure=" + str(_num(strategies.get("market_structure"), 0)) + ", rejection=" + str(_num(strategies.get("candle_rejection"), 0))
            question = "سطوح ساختاری قیمت چه می‌گویند؟"
        elif cid == "risk":
            evidence = "risk-flags=" + str(len(flags)) + ", stop=" + str(stop_pct) + "%, entry-deviation=" + str(entry_deviation) + "%"
            question = "آیا ریسک معامله با کیفیت فرصت متناسب است؟"
        elif cid == "model":
            evidence = "probability=" + str(_num(pred.get("probability"), 0)) + ", rank=" + str(_num(pred.get("rank_score"), 0))
            question = "آیا خروجی مدل و اعتبارسنجی آن قابل اتکاست؟"
        elif cid == "quality":
            evidence = "quality=" + str(_num(quality.get("score"), 0)) + ", stop=" + str(stop_pct) + "%"
            question = "آیا سیگنال و نسبت سود به زیان قابل قبول‌اند؟"
        else:
            evidence = "regime=" + str(ensemble.get("market_regime", "unknown")) + ", news-items=" + str(len(pred.get("recent_news_context") or []))
            question = "آیا خبر یا رژیم بازار با سناریو تعارض دارد؟"
        memos[cid] = {"core": spec["name"], "initial_score": own, "evidence": evidence,
                      "question_to_peers": question, "peer_feedback": [], "revised_score": own}

    for spec in CORE_SPECS:
        cid = spec["id"]
        peers = [other for other in CORE_SPECS if other["id"] != cid]
        peer_scores = [scores.get(other["id"], 0.0) for other in peers]
        peer_average = sum(peer_scores) / len(peer_scores) if peer_scores else scores.get(cid, 0.0)
        memos[cid]["peer_feedback"] = [
            {"from_core": other["id"], "score": scores.get(other["id"], 0.0),
             "stance": "support" if scores.get(other["id"], 0.0) >= 58 else
                       ("objection" if scores.get(other["id"], 0.0) < 45 else "question")}
            for other in peers
        ]
        revised = 0.85 * scores.get(cid, 0.0) + 0.15 * peer_average
        if cid != "risk" and len(flags) >= 3:
            revised -= 5
        if cid != "risk" and (stop_pct > 5 or entry_deviation > 3):
            revised -= 4
        memos[cid]["peer_average"] = round(peer_average, 2)
        memos[cid]["supporting_peers"] = sum(1 for value in peer_scores if value >= 58)
        memos[cid]["objecting_peers"] = sum(1 for value in peer_scores if value < 45)
        memos[cid]["revised_score"] = round(_clamp(revised), 2)

    veto_reasons = []
    if len(flags) >= 4:
        veto_reasons.append(str(len(flags)) + " risk flags")
    if stop_pct > 8:
        veto_reasons.append("stop exceeds 8%")
    if entry_deviation > 6:
        veto_reasons.append("entry deviation exceeds 6%")
    if liquidity > 0 and liquidity < 25000:
        veto_reasons.append("liquidity below 25000")
    return {"memos": memos, "risk_veto": bool(veto_reasons), "risk_veto_reasons": veto_reasons}


def _core_state(system, core_id):
    core = system["cores"].get(core_id)
    if not isinstance(core, dict):
        spec = next(x for x in CORE_SPECS if x["id"] == core_id)
        core = {"id": core_id, "name": spec["name"], "cash_start_usd": STARTING_CASH_PER_CORE,
                "portfolio": None, "last_score": None, "last_vote": "HOLD",
                "last_candidate": None, "votes": 0, "decisions": [], "updated_at": None}
        core["genome"] = _initial_genome(spec)
        system["cores"][core_id] = core
    if not isinstance(core.get("genome"), dict):
        spec = next(x for x in CORE_SPECS if x["id"] == core_id)
        core["genome"] = _initial_genome(spec)
    if not isinstance(core.get("portfolio"), dict):
        substate = {}
        get_portfolio(substate, STARTING_CASH_PER_CORE)
        core["portfolio"] = substate["paper_portfolio"]
    core.setdefault("decisions", [])
    return core


def evaluate_nine_cores(candidates, state, market_info=None, now=None):
    """Maintain nine $10k ledgers, let each core choose one candidate, then vote."""
    system = state.get("nine_core_system")
    if not isinstance(system, dict):
        system = {"version": 1, "starting_cash_per_core_usd": STARTING_CASH_PER_CORE,
                  "majority_required": MAJORITY_VOTES, "cores": {}, "last_result": {},
                  "updated_at": None}
        state["nine_core_system"] = system
    system.setdefault("cores", {})
    timestamp = now or datetime.now(timezone.utc).isoformat()

    # Update every virtual account every scan, including when there are no candidates.
    for spec in CORE_SPECS:
        core = _core_state(system, spec["id"])
        substate = {"paper_portfolio": core["portfolio"]}
        update_paper_portfolio(substate, market_info or {}, now=now)
        core["portfolio"] = substate["paper_portfolio"]

    evolution = evolve_core_population(system)

    candidate_scores = []
    for candidate in candidates or []:
        pred = candidate.get("prediction") or {}
        if not pred.get("token_address") or not pred.get("signal_id"):
            continue
        votes = []
        scores = {}
        for spec in CORE_SPECS:
            genome = _core_state(system, spec["id"]).get("genome") or {}
            score = score_for_core(spec["id"], candidate, genome)
            scores[spec["id"]] = score
            if score >= _num(genome.get("threshold"), spec["threshold"]):
                votes.append(spec["id"])
        discussion = deliberate_between_cores(candidate, scores)
        adjusted_scores = {cid: memo["revised_score"] for cid, memo in discussion["memos"].items()}
        candidate_scores.append({"candidate": candidate, "scores": scores,
                                 "adjusted_scores": adjusted_scores, "discussion": discussion,
                                 "votes": votes, "vote_count": len(votes)})

    # Each core independently picks only its highest-scoring qualified candidate.
    core_choices = {}
    for spec in CORE_SPECS:
        eligible = [row for row in candidate_scores
                    if not row["discussion"]["risk_veto"]
                    and row["adjusted_scores"].get(spec["id"], 0) >= _num(
                        (_core_state(system, spec["id"]).get("genome") or {}).get("threshold"), spec["threshold"])]
        eligible.sort(key=lambda row: row["adjusted_scores"][spec["id"]], reverse=True)
        chosen = eligible[0] if eligible else None
        core = _core_state(system, spec["id"])
        if chosen:
            pred = chosen["candidate"]["prediction"]
            core["last_score"] = chosen["adjusted_scores"][spec["id"]]
            core["last_vote"] = "BUY" 
            core["last_candidate"] = pred.get("token_address")
            core["votes"] = 1
            substate = {"paper_portfolio": core["portfolio"]}
            trade = open_paper_trade(substate, pred, now=now)
            core["portfolio"] = substate["paper_portfolio"]
            core_choices[spec["id"]] = {"token_address": pred["token_address"],
                                        "signal_id": pred["signal_id"],
                                        "score": core["last_score"], "trade": trade,
                                        "peer_feedback_count": len(chosen["discussion"]["memos"][spec["id"]]["peer_feedback"])}
            decision = {"at": timestamp, "vote": "BUY", "token_address": pred["token_address"],
                        "signal_id": pred["signal_id"], "score": core["last_score"],
                        "paper_trade_opened": bool(trade.get("opened"))}
        else:
            core["last_score"] = None
            core["last_vote"] = "PASS"
            core["last_candidate"] = None
            core["votes"] = 0
            decision = {"at": timestamp, "vote": "PASS", "reason": "no candidate passed this core's threshold"}
            core_choices[spec["id"]] = {"token_address": None, "score": None, "trade": {"opened": False, "reason": "no_candidate"}}
        core["decisions"].append(decision)
        core["decisions"] = core["decisions"][-200:]
        core["updated_at"] = timestamp

    # Count one vote per core per candidate. Majority means at least 5 of 9.
    by_address = {}
    for row in candidate_scores:
        address = row["candidate"]["prediction"]["token_address"]
        row["votes"] = [core_id for core_id, choice in core_choices.items()
                        if choice.get("token_address") == address]
        row["vote_count"] = len(row["votes"])
        by_address[address] = row
    ranked = sorted(candidate_scores, key=lambda row: (row["vote_count"],
                      sum(row["scores"].values()) / max(1, len(row["scores"]))), reverse=True)
    winner = ranked[0] if ranked and ranked[0]["vote_count"] >= MAJORITY_VOTES else None
    result = {
        "timestamp": timestamp,
        "majority_required": MAJORITY_VOTES,
        "winner": (winner["candidate"]["prediction"]["token_address"] if winner else None),
        "winner_name": (winner["candidate"]["prediction"].get("name") if winner else None),
        "winner_votes": (winner["vote_count"] if winner else 0),
        "candidates": [{"token_address": row["candidate"]["prediction"]["token_address"],
                        "name": row["candidate"]["prediction"].get("name"),
                        "symbol": row["candidate"]["prediction"].get("symbol"),
                        "votes": row["vote_count"], "voting_cores": row["votes"],
                        "core_scores": row["scores"],
                        "adjusted_scores": row["adjusted_scores"],
                        "discussion": row["discussion"]} for row in ranked[:10]],
        "core_choices": core_choices,
        "evolution": evolution,
        "core_lineage": {spec["id"]: {
            "profile_id": (system["cores"][spec["id"]].get("genome") or {}).get("profile_id", spec["id"]),
            "generation": (system["cores"][spec["id"]].get("genome") or {}).get("generation", 0),
            "parent_id": (system["cores"][spec["id"]].get("genome") or {}).get("parent_id"),
            "status": (system["cores"][spec["id"]].get("genome") or {}).get("status", "founder"),
            "threshold": (system["cores"][spec["id"]].get("genome") or {}).get("threshold", spec["threshold"]),
            "generation_metrics": _closed_trade_metrics(system["cores"][spec["id"]],
                system["cores"][spec["id"]].get("genome") or {}),
        } for spec in CORE_SPECS},
        "shared_data_sources": SHARED_DATA_SOURCES,
        "core_access": {spec["id"]: {"name": spec["name"], "tools": CORE_TOOLSETS[spec["id"]], "uses_shared_scan_data": True, "independent_external_fetch": False} for spec in CORE_SPECS},
        "core_accounts": {spec["id"]: {
            "name": system["cores"][spec["id"]]["name"],
            "initial_cash_usd": system["cores"][spec["id"]]["portfolio"]["initial_cash_usd"],
            "cash_usd": system["cores"][spec["id"]]["portfolio"]["cash_usd"],
            "open_positions": len(system["cores"][spec["id"]]["portfolio"]["open_positions"]),
            "closed_trades": len(system["cores"][spec["id"]]["portfolio"]["closed_trades"]),
            "return_pct": round((sum(float(t.get("pnl_usd", 0)) for t in system["cores"][spec["id"]]["portfolio"]["closed_trades"]) /
                                STARTING_CASH_PER_CORE) * 100, 3),
            "last_vote": system["cores"][spec["id"]]["last_vote"],
            "last_score": system["cores"][spec["id"]]["last_score"],
        } for spec in CORE_SPECS},
        "virtual_capital_total_usd": STARTING_CASH_PER_CORE * len(CORE_SPECS),
        "updated_at": timestamp,
    }
    system["last_result"] = result
    system["updated_at"] = timestamp
    return result


def format_core_result(result):
    lines = ["\n━━ رأی ۹ هستهٔ آزمایشی ━━",
             f"اکثریت لازم: {result.get('majority_required', MAJORITY_VOTES)}/9",
             f"سرمایهٔ مجازی کل: ${result.get('virtual_capital_total_usd', 90000):,.0f}"]
    if result.get("winner"):
        lines.append(f"برندهٔ رأی‌گیری: {result.get('winner_name')} — {result.get('winner_votes')}/9 رأی")
    else:
        lines.append("هیچ توکنی اکثریت ۵ از ۹ را نگرفت؛ هشدار خرید صادر نشود.")
    lines.append(f"ابزارهای اختصاصی تعریف‌شده: {len(result.get('core_access', {}))}/9 هسته؛ منابع مشترک: {len(result.get('shared_data_sources', []))}")
    lines.append("تعامل: هر هسته به ۸ هستهٔ دیگر بازخورد می‌دهد؛ امتیازها پس از گفت‌وگو بازبینی می‌شوند.")
    evolution = result.get("evolution") or {}
    lines.append(f"تکامل ژنتیکی: فرزند جدید {len(evolution.get('replacements', []))}؛ فرزندان فعال {evolution.get('active_offspring', 0)}")
    for event in evolution.get("replacements", []):
        lines.append(f"  {event.get('parent_id')} → فرزند جایگزین {event.get('slot')}؛ نسل {event.get('generation')}؛ پروفایل {event.get('new_profile')}")
    winner_row = next((item for item in result.get("candidates", [])
                       if item.get("token_address") == result.get("winner")), None)
    if winner_row:
        lines.append("خلاصهٔ جلسهٔ برنده:")
        discussion = winner_row.get("discussion") or {}
        for memo in discussion.get("memos", {}).values():
            lines.append(f"  {memo.get('core')}: {memo.get('initial_score', 0):.0f}→{memo.get('revised_score', 0):.0f}؛ حمایت {memo.get('supporting_peers', 0)}، اعتراض {memo.get('objecting_peers', 0)}")
    for candidate in result.get("candidates", [])[:5]:
        lines.append(f"{candidate.get('name')} ({candidate.get('symbol')}): {candidate.get('votes')}/9")
        discussion = candidate.get("discussion") or {}
        if discussion.get("risk_veto"):
            lines.append("  وتوی ریسک: " + ", ".join(discussion.get("risk_veto_reasons", [])))
    return "\n".join(lines)
