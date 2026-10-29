"""A deterministic civic simulation for the nine paper-trading cores.

This is governance logic for software agents, not real-world policing or autonomous AI.
It audits invariants, runs periodic elections, keeps a small legislature/court,
tracks a virtual treasury and market-diversity ecology, and records a public bulletin.
"""
from datetime import datetime, timezone

ELECTION_EVERY_SCANS = 48
TERM_LENGTH_SCANS = 336
SUSPENSION_SCANS = 6
POLICE_SUSPEND_AFTER = 2
COUNCIL_SIZE = 3

CONSTITUTION = [
    {"id": "one_core_one_vote", "text": "Each eligible core has one vote."},
    {"id": "paper_only", "text": "All core portfolios are simulated; no real orders are allowed."},
    {"id": "risk_veto_is_final", "text": "A risk veto cannot be overridden by elections or legislation."},
    {"id": "due_process", "text": "A core is suspended only after repeated critical integrity findings and may appeal."},
    {"id": "separation_of_powers", "text": "Police may flag evidence but cannot edit scores, votes, or ledgers."},
    {"id": "evidence_first", "text": "Changes to a core's status must be logged with an auditable reason."},
]


def _num(value, default=0.0):
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else default
    except (TypeError, ValueError, OverflowError):
        return default


def _ensure_civilization(system):
    civ = system.setdefault("civilization", {})
    civ.setdefault("scan_count", 0)
    civ.setdefault("last_election_scan", -ELECTION_EVERY_SCANS)
    civ.setdefault("election_number", 0)
    civ.setdefault("president", None)
    civ.setdefault("council", [])
    civ.setdefault("constitution", [dict(x) for x in CONSTITUTION])
    civ.setdefault("police", {"incidents": [], "strikes": {}, "suspended_until": {}, "audit_count": 0})
    civ.setdefault("court", {"appeals": [], "decisions": []})
    civ.setdefault("legislature", {"proposals": [], "enacted": []})
    civ.setdefault("economy", {"virtual_treasury_usd": 0.0, "last_total_pnl_usd": 0.0})
    civ.setdefault("ecology", {"token_counts": {}, "recent_winners": [], "diversity_ratio": 1.0, "warning": None})
    civ.setdefault("academy", {"case_studies": [], "lessons": []})
    civ.setdefault("bulletin", [])
    return civ


def _core_performance(core):
    portfolio = core.get("portfolio") or {}
    trades = [t for t in portfolio.get("closed_trades", []) if isinstance(t, dict)]
    pnl = sum(_num(t.get("pnl_usd")) for t in trades)
    wins = sum(_num(t.get("pnl_usd")) > 0 for t in trades)
    win_rate = wins / len(trades) if trades else 0.5
    initial = _num(portfolio.get("initial_cash_usd"), 10000.0) or 10000.0
    return {"closed_trades": len(trades), "pnl_usd": round(pnl, 4),
            "win_rate": round(win_rate, 4), "return_pct": round(pnl / initial * 100, 4)}


def _audit_cores(system, core_specs, civ, timestamp):
    police = civ["police"]
    police["audit_count"] = int(police.get("audit_count", 0)) + 1
    active_ids = {s["id"] for s in core_specs}
    for core_id in sorted(active_ids):
        core = system.get("cores", {}).get(core_id) or {}
        portfolio = core.get("portfolio") or {}
        findings = []
        if _num(portfolio.get("initial_cash_usd"), 10000.0) < 0:
            findings.append("negative_initial_virtual_cash")
        if _num(portfolio.get("cash_usd"), 0.0) < -0.000001:
            findings.append("negative_virtual_cash")
        genome = core.get("genome") or {}
        threshold = _num(genome.get("threshold"), 60)
        if not 40 <= threshold <= 80:
            findings.append("strategy_threshold_out_of_bounds")
        if len(core.get("decisions") or []) > 200:
            findings.append("decision_log_limit_exceeded")
        for finding in findings:
            incident = {"at": timestamp, "scan": civ["scan_count"], "core_id": core_id, "finding": finding, "severity": "critical"}
            police.setdefault("incidents", []).append(incident)
            strikes = police.setdefault("strikes", {})
            strikes[core_id] = int(strikes.get(core_id, 0)) + 1
            if strikes[core_id] >= POLICE_SUSPEND_AFTER:
                until = civ["scan_count"] + SUSPENSION_SCANS
                police.setdefault("suspended_until", {})[core_id] = max(
                    int(police["suspended_until"].get(core_id, 0)), until)
                incident["action"] = "temporary_suspension"
                incident["suspended_until_scan"] = until
                civ["court"].setdefault("appeals", []).append({
                    "core_id": core_id, "filed_scan": civ["scan_count"],
                    "reason": "automatic appeal against temporary suspension",
                    "status": "pending", "evidence_finding": finding,
                })
    police["incidents"] = police.get("incidents", [])[-200:]


def is_core_eligible(system, core_id):
    civ = system.get("civilization") or {}
    until = (civ.get("police") or {}).get("suspended_until", {}).get(core_id, 0)
    return int((civ.get("scan_count") or 0)) >= int(until or 0)


def _review_court(system, core_specs, civ, timestamp):
    police, court = civ["police"], civ["court"]
    core_ids = {s["id"] for s in core_specs}
    for core_id, until in list((police.get("suspended_until") or {}).items()):
        if core_id not in core_ids or int(civ["scan_count"]) < int(until):
            continue
        recent = [i for i in police.get("incidents", [])
                  if i.get("core_id") == core_id and
                  civ["scan_count"] - int(i.get("scan", civ["scan_count"])) < SUSPENSION_SCANS]
        # Due process: expiry triggers review; unresolved recent critical incidents extend suspension.
        if recent:
            police["suspended_until"][core_id] = civ["scan_count"] + SUSPENSION_SCANS
            decision = "suspension_extended_pending_clean_audit"
        else:
            police["suspended_until"].pop(core_id, None)
            police.setdefault("strikes", {})[core_id] = 0
            decision = "reinstated_after_review"
        court.setdefault("decisions", []).append(
            {"at": timestamp, "core_id": core_id, "decision": decision})
        pending = next((a for a in reversed(court.get("appeals", []))
                        if a.get("core_id") == core_id and a.get("status") == "pending"), None)
        if pending:
            pending["status"] = "accepted" if decision == "reinstated_after_review" else "deferred"
            pending["decision_at"] = timestamp
            pending["decision"] = decision
    court["decisions"] = court.get("decisions", [])[-100:]


def _elect(system, core_specs, civ, timestamp):
    cores = system.get("cores", {})
    eligible = [s for s in core_specs if is_core_eligible(system, s["id"])]
    if not eligible:
        return
    profiles = {s["id"]: _core_performance(cores.get(s["id"]) or {}) for s in eligible}
    # Votes are cast by all eligible cores; voters have distinct issue priorities.
    tallies = {s["id"]: 0 for s in eligible}
    ballots = {}
    for voter in eligible:
        voter_id = voter["id"]
        risk_priority = voter_id == "risk"
        quality_priority = voter_id in {"quality", "model"}
        best_id, best_score = None, float("-inf")
        for candidate in eligible:
            cid = candidate["id"]
            p = profiles[cid]
            core = cores.get(cid) or {}
            genome = core.get("genome") or {}
            police_strikes = int(civ["police"].get("strikes", {}).get(cid, 0))
            law_score = max(0, 100 - 20 * police_strikes)
            if risk_priority:
                score = 0.30 * max(0, 100 - abs(p["return_pct"])) + 0.25 * law_score + 0.25 * (100 * p["win_rate"]) + 0.20 * _num(genome.get("threshold"), 60)
            elif quality_priority:
                score = 0.35 * (100 * p["win_rate"]) + 0.30 * law_score + 0.20 * min(100, p["closed_trades"] * 5) + 0.15 * p["return_pct"]
            else:
                score = 0.45 * p["return_pct"] + 0.25 * (100 * p["win_rate"]) + 0.20 * min(100, p["closed_trades"] * 5) + 0.10 * law_score
            # Tiny stable role affinity breaks ties without random outcomes.
            if voter_id == cid:
                score += 0.01
            if score > best_score:
                best_id, best_score = cid, score
        if best_id:
            tallies[best_id] += 1
            ballots[voter_id] = best_id
    ordered = sorted(tallies, key=lambda cid: (-tallies[cid], -profiles[cid]["return_pct"], cid))
    civ["election_number"] = int(civ.get("election_number", 0)) + 1
    civ["president"] = ordered[0] if ordered else None
    civ["council"] = ordered[:COUNCIL_SIZE]
    civ["last_election_scan"] = civ["scan_count"]
    record = {"at": timestamp, "election": civ["election_number"], "scan": civ["scan_count"],
              "president": civ["president"], "council": list(civ["council"]),
              "votes": dict(tallies), "ballots": ballots}
    civ.setdefault("elections", []).append(record)
    civ["elections"] = civ["elections"][-20:]
    civ["bulletin"].append({"at": timestamp, "type": "election",
                            "message": "Core election completed; president and council recorded."})


def _legislate(civ, timestamp):
    # Laws are policy records, not executable code. Safety-critical constitutional rules cannot be repealed.
    legislature = civ["legislature"]
    if not legislature.get("enacted"):
        legislature["enacted"] = [
            {"id": "signal_quorum", "rule": "BUY-WATCH requires at least five eligible core votes.", "status": "active"},
            {"id": "liquidity_guard", "rule": "Existing liquidity and shared risk veto checks remain mandatory.", "status": "active"},
            {"id": "paper_trading_only", "rule": "Civilization and evolution systems may not submit real trades.", "status": "constitutional"},
            {"id": "audit_trail", "rule": "Elections, suspensions, appeals and evolution are logged.", "status": "active"},
        ]
        civ["bulletin"].append({"at": timestamp, "type": "law", "message": "Initial core constitution and laws published."})


def _update_economy_and_academy(system, civ, timestamp):
    """Account for new closed paper trades once; the treasury is virtual, never real money."""
    economy = civ["economy"]
    seen = economy.setdefault("seen_closed_counts", {})
    academy = civ["academy"]
    stats = academy.setdefault("resolved_trade_stats", {
        "trades": 0, "wins": 0, "losses": 0, "net_pnl_usd": 0.0,
    })
    for core_id, core in (system.get("cores") or {}).items():
        trades = ((core or {}).get("portfolio") or {}).get("closed_trades") or []
        start = min(int(seen.get(core_id, 0) or 0), len(trades))
        for trade in trades[start:]:
            if not isinstance(trade, dict) or not isinstance(trade.get("pnl_usd"), (int, float)):
                continue
            pnl = _num(trade.get("pnl_usd"))
            tax = max(0.0, pnl) * 0.01
            economy["virtual_treasury_usd"] = round(_num(economy.get("virtual_treasury_usd")) + tax, 4)
            economy["tax_revenue_usd"] = round(_num(economy.get("tax_revenue_usd")) + tax, 4)
            stats["trades"] += 1
            stats["wins" if pnl > 0 else "losses"] += 1
            stats["net_pnl_usd"] = round(_num(stats.get("net_pnl_usd")) + pnl, 4)
            academy.setdefault("lessons", []).append({
                "at": timestamp, "core_id": core_id,
                "outcome": "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven",
                "pnl_usd": round(pnl, 4), "token_address": trade.get("token_address"),
                "exit_reason": trade.get("exit_reason"),
                "note": "Resolved paper-trade evidence; not a standalone prediction rule.",
            })
        seen[core_id] = len(trades)
    academy["lessons"] = academy.get("lessons", [])[-200:]


def run_civilization_cycle(system, core_specs, timestamp=None):
    """Advance civic institutions once per bot scan and return a concise report."""
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    civ = _ensure_civilization(system)
    civ["scan_count"] = int(civ.get("scan_count", 0)) + 1
    _audit_cores(system, core_specs, civ, timestamp)
    _review_court(system, core_specs, civ, timestamp)
    _legislate(civ, timestamp)
    _update_economy_and_academy(system, civ, timestamp)
    if (civ["president"] is None or
            civ["scan_count"] - int(civ.get("last_election_scan", -ELECTION_EVERY_SCANS)) >= ELECTION_EVERY_SCANS or
            civ["scan_count"] - int(civ.get("last_election_scan", 0)) >= TERM_LENGTH_SCANS):
        _elect(system, core_specs, civ, timestamp)
    return {"scan_count": civ["scan_count"], "president": civ.get("president"),
            "council": list(civ.get("council") or []), "election_number": civ.get("election_number", 0),
            "laws": list(civ["legislature"].get("enacted") or []),
            "police_incidents": len(civ["police"].get("incidents") or []),
            "suspended_cores": [cid for cid in (civ["police"].get("suspended_until") or {})
                                if not is_core_eligible(system, cid)],
            "court_decisions": len(civ["court"].get("decisions") or []),
            "virtual_treasury_usd": round(_num(civ["economy"].get("virtual_treasury_usd")), 2),
            "virtual_tax_revenue_usd": round(_num(civ["economy"].get("tax_revenue_usd")), 2),
            "research_lessons": len(civ["academy"].get("lessons") or []),
            "ecology": dict(civ["ecology"]), "bulletin": list(civ["bulletin"][-5:])}


def record_civilization_outcome(system, winner_row, timestamp=None):
    """Update civic economy, ecology and research ledger from observed scan outcomes."""
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    civ = _ensure_civilization(system)
    if not winner_row:
        civ["bulletin"].append({"at": timestamp, "type": "market",
                                "message": "No candidate reached the required eligible-core quorum."})
        civ["bulletin"] = civ["bulletin"][-50:]
        return
    pred = (winner_row.get("candidate") or {}).get("prediction") or {}
    address = str(pred.get("token_address") or "")
    if address:
        ecology = civ["ecology"]
        counts = ecology.setdefault("token_counts", {})
        counts[address] = int(counts.get(address, 0)) + 1
        recent = (ecology.setdefault("recent_winners", []) + [address])[-20:]
        ecology["recent_winners"] = recent
        unique = len(set(recent))
        ecology["diversity_ratio"] = round(unique / len(recent), 3) if recent else 1.0
        ecology["warning"] = ("low_project_diversity" if len(recent) >= 10 and unique / len(recent) < 0.4 else None)
    civ["academy"].setdefault("case_studies", []).append({
        "at": timestamp, "token_address": address, "vote_count": winner_row.get("vote_count", 0),
        "mean_score": round(sum((winner_row.get("scores") or {}).values()) /
                            max(1, len(winner_row.get("scores") or {})), 2),
        "risk_veto": bool((winner_row.get("discussion") or {}).get("risk_veto")),
        "evidence_note": "Decision snapshot only; no outcome/profit claim until the paper trade closes."
    })
    civ["academy"]["case_studies"] = civ["academy"]["case_studies"][-100:]
    civ["bulletin"].append({"at": timestamp, "type": "market",
                            "message": "Candidate decision recorded for later paper-outcome review."})
    civ["bulletin"] = civ["bulletin"][-50:]


def format_civilization_report(report):
    lines = ["\n━━ جامعهٔ ۹ هسته ━━",
             f"انتخابات: {report.get('election_number', 0)} | رئیس: {report.get('president') or 'تعیین‌نشده'}",
             "شورای منتخب: " + (", ".join(report.get("council") or []) or "تعیین‌نشده"),
             f"پلیس/بازرسی: {report.get('police_incidents', 0)} رخداد ثبت‌شده | تعلیق فعال: {', '.join(report.get('suspended_cores') or []) or 'ندارد'}",
             f"دادگاه: {report.get('court_decisions', 0)} تصمیم ثبت‌شده | قوانین: {len(report.get('laws') or [])}",
             f"خزانهٔ مجازی: ${_num(report.get('virtual_treasury_usd')):,.2f} (مالیات شبیه‌سازی‌شده: ${_num(report.get('virtual_tax_revenue_usd')):,.2f})",
             f"آکادمی: {report.get('research_lessons', 0)} نتیجهٔ معاملهٔ مجازی برای مطالعه ثبت شده",
             f"تنوع پروژه‌های منتخب: {report.get('ecology', {}).get('diversity_ratio', 1.0):.0%}"]
    if report.get("ecology", {}).get("warning"):
        lines.append("هشدار محیط: تمرکز بیش از حد روی پروژه‌های تکراری؛ تنوع انتخاب‌ها پایین است.")
    lines.append("نهادها: انتخابات، شورا/قانون‌گذاری، پلیسِ حسابرس، دادگاه، خزانه، محیط‌زیست و آکادمی پژوهش.")
    return "\n".join(lines)
