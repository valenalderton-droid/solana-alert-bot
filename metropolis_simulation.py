"""A deterministic, persistent metropolis simulation for the nine market-analysis cores.

All residents, institutions, budgets and projects are virtual. No real trades are
placed. The city evolves slowly by scan and runs a larger civic/economic tick
every simulated day (96 scans at a 15-minute schedule).
"""
from datetime import datetime, timezone

SCANS_PER_SIM_DAY = 96
DISTRICTS = {
    "quant_district": {"name": "Quant & Statistics Quarter", "service": "research"},
    "exchange_district": {"name": "Exchange and Market Data Quarter", "service": "market_data"},
    "university_district": {"name": "University Campus", "service": "education"},
    "risk_district": {"name": "Risk, Audit & Resilience Quarter", "service": "integrity"},
    "news_district": {"name": "News and Macro Observatory", "service": "news"},
    "infrastructure_district": {"name": "Transport, Energy & Compute", "service": "infrastructure"},
    "residential_district": {"name": "Residential Commons", "service": "wellbeing"},
}
SERVICES = {
    "research": {"budget_share": 0.22, "capacity": 65.0},
    "market_data": {"budget_share": 0.18, "capacity": 70.0},
    "education": {"budget_share": 0.16, "capacity": 60.0},
    "integrity": {"budget_share": 0.16, "capacity": 75.0},
    "news": {"budget_share": 0.10, "capacity": 55.0},
    "infrastructure": {"budget_share": 0.12, "capacity": 60.0},
    "wellbeing": {"budget_share": 0.06, "capacity": 65.0},
}
JOBS = {
    "quant_researcher": "quant_district",
    "price_action_analyst": "exchange_district",
    "risk_auditor": "risk_district",
    "news_verifier": "news_district",
    "data_engineer": "infrastructure_district",
    "professor": "university_district",
    "portfolio_researcher": "quant_district",
    "execution_engineer": "exchange_district",
    "civic_mediator": "residential_district",
}
ROLE_BY_CORE = {
    "trend": "price_action_analyst", "breakout": "execution_engineer",
    "momentum": "quant_researcher", "volume": "data_engineer",
    "structure": "portfolio_researcher", "risk": "risk_auditor",
    "model": "quant_researcher", "quality": "professor", "context": "news_verifier",
}


def _num(value, default=0.0):
    try:
        v = float(value)
        return v if v == v and abs(v) != float("inf") else default
    except (TypeError, ValueError, OverflowError):
        return default


def ensure_metropolis(civ, core_specs):
    city = civ.setdefault("metropolis", {})
    city.setdefault("name", "Solana Financial Research Metropolis")
    city.setdefault("version", 1)
    city.setdefault("scan_count", 0)
    city.setdefault("simulated_day", 0)
    city.setdefault("districts", {
        key: {"name": value["name"], "population": 100 + i * 17,
              "service_level": 60.0 + i, "jobs": 0, "infrastructure_health": 75.0}
        for i, (key, value) in enumerate(DISTRICTS.items())
    })
    city.setdefault("services", {key: dict(value, budget_usd=0.0, satisfaction=60.0, backlog=0)
                                 for key, value in SERVICES.items()})
    city.setdefault("residents", {})
    city.setdefault("economy", {
        "treasury_usd": 5000.0, "daily_income_usd": 0.0, "daily_expenses_usd": 0.0,
        "cumulative_income_usd": 0.0, "cumulative_expenses_usd": 0.0,
        "employment_rate": 1.0, "housing_pressure": 0.15, "energy_reliability": 0.8,
        "compute_capacity": 100.0, "research_productivity": 50.0,
    })
    city.setdefault("research", {
        "proposals": [], "experiments": [], "findings": [],
        "next_experiment_id": 1, "champion": None, "incubator": [],
    })
    city.setdefault("supply_chains", {
        "data": {"reliability": 0.80, "backlog": 0},
        "compute": {"reliability": 0.85, "backlog": 0},
        "education": {"reliability": 0.80, "backlog": 0},
        "news": {"reliability": 0.75, "backlog": 0},
        "risk_controls": {"reliability": 0.90, "backlog": 0},
    })
    city.setdefault("events", [])
    city.setdefault("history", [])
    for spec in core_specs:
        cid = spec["id"]
        city["residents"].setdefault(cid, {
            "role": ROLE_BY_CORE.get(cid, "quant_researcher"),
            "district": JOBS.get(ROLE_BY_CORE.get(cid, "quant_researcher"), "quant_district"),
            "employment": "employed", "wellbeing": 60.0, "reputation": 50.0,
            "collaborations": 0, "disputes": 0, "research_contributions": 0,
            "last_reviewed_trade_count": 0,
        })
    return city


def _portfolio_stats(system, core_id):
    core = (system.get("cores") or {}).get(core_id) or {}
    portfolio = core.get("portfolio") or {}
    trades = [t for t in portfolio.get("closed_trades", []) if isinstance(t, dict)]
    pnl = sum(_num(t.get("pnl_usd")) for t in trades)
    wins = sum(_num(t.get("pnl_usd")) > 0 for t in trades)
    return {"trades": len(trades), "pnl": pnl, "win_rate": wins / len(trades) if trades else 0.5}


def _update_residents(system, city, core_specs):
    for spec in core_specs:
        cid = spec["id"]
        resident = city["residents"][cid]
        stats = _portfolio_stats(system, cid)
        seen = min(int(resident.get("last_reviewed_trade_count", 0)), stats["trades"])
        newly_closed = stats["trades"] - seen
        if newly_closed:
            # Wellbeing/reputation reflect work quality and review, not profit alone.
            resident["wellbeing"] = round(max(0.0, min(100.0,
                _num(resident.get("wellbeing"), 60) + (0.5 if stats["win_rate"] >= 0.5 else -0.25))), 2)
            resident["reputation"] = round(max(0.0, min(100.0,
                _num(resident.get("reputation"), 50) + (0.25 if stats["win_rate"] >= 0.5 else -0.2))), 2)
            resident["research_contributions"] = int(resident.get("research_contributions", 0)) + newly_closed
        resident["last_reviewed_trade_count"] = stats["trades"]
    # Cross-core collaboration is scheduled, deterministic and auditable.
    if city["scan_count"] % 8 == 0 and len(core_specs) > 1:
        ids = [s["id"] for s in core_specs]
        for i, cid in enumerate(ids):
            partner = ids[(i + 1) % len(ids)]
            city["residents"][cid]["collaborations"] += 1
            city["residents"][partner]["collaborations"] += 0


def _research_round(city, core_specs, timestamp):
    research = city["research"]
    residents = city["residents"]
    ids = [s["id"] for s in core_specs]
    if not ids:
        return
    proposals = [
        {"topic": "out_of_sample_validation", "owner": "model", "cost": 80, "priority": 5},
        {"topic": "liquidity_and_slippage", "owner": "risk", "cost": 65, "priority": 5},
        {"topic": "false_breakout_study", "owner": "breakout", "cost": 55, "priority": 4},
        {"topic": "news_regime_event_study", "owner": "context", "cost": 60, "priority": 4},
        {"topic": "cross_project_correlation", "owner": "structure", "cost": 50, "priority": 3},
        {"topic": "volume_quality_audit", "owner": "volume", "cost": 45, "priority": 3},
    ]
    known = {p.get("topic") for p in research.get("proposals", [])}
    for proposal in proposals:
        if proposal["topic"] not in known:
            research["proposals"].append(dict(proposal, status="proposed", submitted_at=timestamp))
    research["proposals"] = research["proposals"][-50:]
    treasury = _num(city["economy"].get("treasury_usd"))
    completed_topics = {e.get("topic") for e in research.get("experiments", [])}
    # One affordable experiment per simulated day, prioritizing risk and validation.
    available = [p for p in research["proposals"] if p["status"] == "proposed"
                 and p["topic"] not in completed_topics and p["cost"] <= treasury]
    available.sort(key=lambda p: (-p["priority"], p["cost"], p["topic"]))
    if available:
        proposal = available[0]
        exp_id = int(research.get("next_experiment_id", 1))
        research["next_experiment_id"] = exp_id + 1
        city["economy"]["treasury_usd"] = round(treasury - proposal["cost"], 2)
        proposal["status"] = "running"
        research["experiments"].append({
            "id": exp_id, "topic": proposal["topic"], "owner": proposal["owner"],
            "started_at": timestamp, "status": "queued_for_evidence",
            "budget_usd": proposal["cost"], "method": "paper-trading and historical-data protocol",
            "result": None, "promoted_to_champion": False,
            "guardrail": "No promotion without sufficient out-of-sample evidence.",
        })
        research["experiments"] = research["experiments"][-100:]
        residents.setdefault(proposal["owner"], {}).setdefault("research_contributions", 0)
        residents[proposal["owner"]]["research_contributions"] += 1


def _review_experiments(system, city, timestamp):
    """Use a chronological holdout only as descriptive evidence; never auto-promote."""
    research = city["research"]
    for experiment in research.get("experiments", []):
        if experiment.get("status") != "queued_for_evidence":
            continue
        owner = experiment.get("owner")
        core = (system.get("cores") or {}).get(owner) or {}
        trades = [t for t in ((core.get("portfolio") or {}).get("closed_trades") or [])
                  if isinstance(t, dict) and isinstance(t.get("pnl_usd"), (int, float))]
        # 30 observations total, with the latest 20 reserved as chronological holdout.
        if len(trades) < 30:
            experiment["observations_available"] = len(trades)
            continue
        holdout = trades[-20:]
        pnl = [_num(t.get("pnl_usd")) for t in holdout]
        wins = sum(x > 0 for x in pnl)
        running = peak = 0.0
        max_drawdown = 0.0
        for value in pnl:
            running += value
            peak = max(peak, running)
            max_drawdown = max(max_drawdown, peak - running)
        experiment["status"] = "evidence_ready_for_review"
        experiment["completed_at"] = timestamp
        experiment["result"] = {
            "total_observations": len(trades),
            "holdout_observations": len(holdout),
            "holdout_net_pnl_usd": round(sum(pnl), 4),
            "holdout_win_rate": round(wins / len(holdout), 4),
            "holdout_max_drawdown_usd": round(max_drawdown, 4),
            "method": "last 20 chronologically ordered closed paper trades",
            "interpretation": "descriptive evidence only; not proof that the named topic caused performance",
        }
        research.setdefault("findings", []).append({
            "at": timestamp, "experiment_id": experiment["id"], "topic": experiment["topic"],
            "status": "evidence_ready_for_review",
            "holdout_net_pnl_usd": experiment["result"]["holdout_net_pnl_usd"],
            "holdout_win_rate": experiment["result"]["holdout_win_rate"],
            "decision": "manual review required; no automatic strategy promotion",
        })
    research["findings"] = research.get("findings", [])[-100:]


def _update_city_dynamics(city):
    """Slow, bounded city dynamics; avoid random or explosive population changes."""
    economy = city["economy"]
    districts = city["districts"]
    avg_service = sum(_num(s.get("satisfaction"), 60) for s in city["services"].values()) / max(1, len(city["services"]))
    for district in districts.values():
        service = _num(district.get("service_level"), 60)
        infra = _num(district.get("infrastructure_health"), 75)
        district["service_level"] = round(max(0.0, min(100.0, service + (0.15 if avg_service >= 55 else -0.2))), 2)
        district["infrastructure_health"] = round(max(0.0, min(100.0, infra + (0.1 if _num(economy.get("treasury_usd")) > 100 else -0.15))), 2)
        population = int(district.get("population", 0))
        # Population changes by at most one resident per simulated day and district.
        if district["service_level"] >= 70 and population < 500:
            district["population"] = population + 1
        elif district["service_level"] < 35 and population > 50:
            district["population"] = population - 1
        district["jobs"] = min(int(district["population"]), max(0, int(district["population"] * district["service_level"] / 100)))
    economy["housing_pressure"] = round(max(0.0, min(1.0,
        sum(int(d.get("population", 0)) for d in districts.values()) / 5000.0
        + (100.0 - avg_service) / 500.0)), 3)


def _daily_tick(system, city, core_specs, timestamp):
    economy = city["economy"]
    services = city["services"]
    total_population = sum(int(d.get("population", 0)) for d in city["districts"].values())
    treasury = _num(economy.get("treasury_usd"))
    # Income is virtual and mostly generated by simulated city services, never real trading capital.
    income = round(20.0 + total_population * 0.025 + min(30.0, len(core_specs) * 2.0), 2)
    expenses = round(sum(_num(s.get("budget_usd")) for s in services.values()) + 15.0, 2)
    economy["daily_income_usd"] = income
    economy["daily_expenses_usd"] = expenses
    economy["cumulative_income_usd"] = round(_num(economy.get("cumulative_income_usd")) + income, 2)
    economy["cumulative_expenses_usd"] = round(_num(economy.get("cumulative_expenses_usd")) + expenses, 2)
    economy["treasury_usd"] = round(max(0.0, treasury + income - expenses), 2)
    economy["employment_rate"] = round(sum(r.get("employment") == "employed" for r in city["residents"].values()) / max(1, len(city["residents"])), 3)
    service_avg = sum(_num(s.get("satisfaction"), 60) for s in services.values()) / max(1, len(services))
    economy["housing_pressure"] = round(max(0.0, min(1.0, total_population / 5000.0 + (100 - service_avg) / 500.0)), 3)
    infra = services.get("infrastructure", {})
    economy["energy_reliability"] = round(max(0.1, min(1.0, _num(infra.get("satisfaction"), 60) / 100.0)), 3)
    economy["compute_capacity"] = round(max(10.0, min(200.0,
        _num(economy.get("compute_capacity"), 100) + (1.0 if economy["treasury_usd"] > 1000 else -0.5))), 2)
    economy["research_productivity"] = round(max(0.0, min(100.0,
        0.6 * _num(economy.get("compute_capacity"), 100) + 0.4 * service_avg)), 2)
    for key, service in services.items():
        share = _num(service.get("budget_share"))
        budget = round(min(economy["treasury_usd"], max(5.0, share * max(0.0, economy["treasury_usd"]) * 0.08)), 2)
        service["budget_usd"] = budget
        pressure = 1.0 if key in {"research", "market_data", "integrity"} else 0.4
        service["satisfaction"] = round(max(0.0, min(100.0,
            _num(service.get("satisfaction"), 60) + (0.8 if budget >= 5 else -1.0) - pressure * _num(service.get("backlog")) * 0.02)), 2)
        service["backlog"] = max(0, int(service.get("backlog", 0)) + (1 if budget < 5 else -1))
    # Reliability follows service quality, with bounded slow changes.
    for name, chain in city["supply_chains"].items():
        service_key = {"data": "market_data", "compute": "infrastructure", "education": "education",
                       "news": "news", "risk_controls": "integrity"}[name]
        target = _num(services[service_key].get("satisfaction"), 60) / 100.0
        chain["reliability"] = round(max(0.1, min(0.99,
            _num(chain.get("reliability"), 0.8) * 0.8 + target * 0.2)), 3)
        chain["backlog"] = services[service_key]["backlog"]
    _research_round(city, core_specs, timestamp)
    _review_experiments(system, city, timestamp)
    _update_city_dynamics(city)
    city["simulated_day"] += 1
    city["events"].append({"at": timestamp, "type": "daily_economy",
                           "message": "Daily city budget, services, supply chains and research queue updated.",
                           "treasury_usd": economy["treasury_usd"]})
    city["events"] = city["events"][-100:]
    city["history"].append({
        "day": city["simulated_day"], "treasury_usd": economy["treasury_usd"],
        "employment_rate": economy["employment_rate"], "energy_reliability": economy["energy_reliability"],
        "compute_capacity": economy["compute_capacity"], "research_productivity": economy["research_productivity"],
    })
    city["history"] = city["history"][-180:]


def update_metropolis(system, civ, core_specs, timestamp=None):
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    city = ensure_metropolis(civ, core_specs)
    city["scan_count"] = int(city.get("scan_count", 0)) + 1
    _update_residents(system, city, core_specs)
    if city["scan_count"] % SCANS_PER_SIM_DAY == 0:
        _daily_tick(system, city, core_specs, timestamp)
    return metropolis_summary(civ)


def metropolis_summary(civ):
    city = (civ or {}).get("metropolis") or {}
    districts = city.get("districts") or {}
    services = city.get("services") or {}
    economy = city.get("economy") or {}
    research = city.get("research") or {}
    residents = city.get("residents") or {}
    experiments = research.get("experiments") or []
    return {
        "name": city.get("name", "Solana Financial Research Metropolis"),
        "districts": len(districts), "residents": len(residents),
        "simulated_day": int(city.get("simulated_day", 0)),
        "employment_rate": _num(economy.get("employment_rate"), 1.0),
        "treasury_usd": _num(economy.get("treasury_usd")),
        "energy_reliability": _num(economy.get("energy_reliability"), 0.8),
        "compute_capacity": _num(economy.get("compute_capacity"), 100),
        "research_productivity": _num(economy.get("research_productivity"), 50),
        "service_average": round(sum(_num(s.get("satisfaction"), 60) for s in services.values()) / max(1, len(services)), 2),
        "supply_reliability": round(sum(_num(s.get("reliability"), 0.8) for s in (city.get("supply_chains") or {}).values()) / max(1, len(city.get("supply_chains") or {})), 3),
        "experiments": len(experiments),
        "queued_experiments": sum(e.get("status") == "queued_for_evidence" for e in experiments),
        "proposals": len(research.get("proposals") or []),
        "events": len(city.get("events") or []),
        "collaborations": sum(int(r.get("collaborations", 0)) for r in residents.values()),
    }


def format_metropolis_report(summary):
    return (
        "\n━━ کلان‌شهر پژوهش مالی ━━\n"
        f"مناطق: {summary.get('districts', 0)} | ساکنان هسته‌ای: {summary.get('residents', 0)} | روز شبیه‌سازی: {summary.get('simulated_day', 0)}\n"
        f"خزانهٔ شهری مجازی: ${_num(summary.get('treasury_usd')):,.2f} | اشتغال: {_num(summary.get('employment_rate')):.0%} | میانگین خدمات: {_num(summary.get('service_average')):.1f}/100\n"
        f"پایداری انرژی: {_num(summary.get('energy_reliability')):.0%} | ظرفیت محاسباتی: {_num(summary.get('compute_capacity')):.1f} | بهره‌وری پژوهش: {_num(summary.get('research_productivity')):.1f}/100\n"
        f"پایداری زنجیره‌های تأمین: {_num(summary.get('supply_reliability')):.0%} | همکاری‌های ثبت‌شده: {summary.get('collaborations', 0)}\n"
        f"پروژه‌های پژوهشی: {summary.get('proposals', 0)} | آزمایش‌های ثبت‌شده: {summary.get('experiments', 0)} | منتظر شواهد: {summary.get('queued_experiments', 0)}"
    )
