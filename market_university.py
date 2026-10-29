"""University, faculty and research simulation for the nine paper-trading cores.

All grades and grants are simulated. The university improves ranking modestly;
it never overrides market-data, risk-veto, or majority-vote gates.
"""
from datetime import datetime, timezone

DEPARTMENTS = {
    "quant_lab": {"name": "Quantitative Finance & Econometrics", "skills": ["statistics", "calibration"], "courses": ["walk_forward_validation", "probability_calibration", "sample_size_and_bias"]},
    "price_action": {"name": "Price Action & Market Structure", "skills": ["structure", "trend", "breakout"], "courses": ["market_structure", "support_resistance", "false_breakout_detection"]},
    "risk_school": {"name": "Risk Management & Survival", "skills": ["risk", "position_sizing", "drawdown"], "courses": ["risk_of_ruin", "drawdown_control", "liquidity_slippage"]},
    "onchain_lab": {"name": "On-chain Data & Liquidity", "skills": ["onchain", "liquidity", "volume"], "courses": ["pool_liquidity", "volume_quality", "wallet_concentration"]},
    "news_school": {"name": "News, Macro & Event Studies", "skills": ["news", "macro", "event_risk"], "courses": ["source_verification", "event_studies", "macro_regimes"]},
    "ml_institute": {"name": "Machine Learning & Model Validation", "skills": ["model_validation", "features", "drift"], "courses": ["chronological_splits", "overfit_detection", "drift_monitoring"]},
    "portfolio_school": {"name": "Portfolio Theory & Correlation", "skills": ["portfolio", "correlation", "diversification"], "courses": ["correlation_regimes", "portfolio_drawdown", "capital_allocation"]},
    "market_microstructure": {"name": "Market Microstructure & Execution", "skills": ["execution", "spread", "market_impact"], "courses": ["spread_and_slippage", "liquidity_depth", "execution_costs"]},
}
MAJORS = {
    "trend": "price_action", "breakout": "price_action", "momentum": "quant_lab",
    "volume": "onchain_lab", "structure": "price_action", "risk": "risk_school",
    "mean_reversion": "quant_lab", "candle": "market_microstructure", "news": "news_school",
}
COURSE_INTERVAL = 4


def _num(value, default=0.0):
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else default
    except (TypeError, ValueError, OverflowError):
        return default


def ensure_university(civ, core_specs):
    uni = civ.setdefault("universities", {})
    uni.setdefault("name", "University of Financial Markets")
    uni.setdefault("departments", {k: dict(v) for k, v in DEPARTMENTS.items()})
    uni.setdefault("faculty", [
        {"id": "prof_quant", "department": "quant_lab"},
        {"id": "prof_price_action", "department": "price_action"},
        {"id": "prof_risk", "department": "risk_school"},
        {"id": "prof_onchain", "department": "onchain_lab"},
        {"id": "prof_news", "department": "news_school"},
        {"id": "prof_ml", "department": "ml_institute"},
        {"id": "prof_portfolio", "department": "portfolio_school"},
        {"id": "prof_execution", "department": "market_microstructure"},
    ])
    uni.setdefault("scan_count", 0)
    uni.setdefault("students", {})
    uni.setdefault("research", {"papers": [], "grants": [], "experiments": []})
    for spec in core_specs:
        cid = spec["id"]
        major = MAJORS.get(cid, "ml_institute")
        student = uni["students"].setdefault(cid, {
            "major": major,
            "minor": "risk_school" if major != "risk_school" else "portfolio_school",
            "skills": {skill: 50.0 for dept in DEPARTMENTS.values() for skill in dept["skills"]},
            "credits": 0, "completed_courses": [], "gpa": 50.0,
            "research_points": 0.0, "last_course_scan": -COURSE_INTERVAL,
            "last_seen_trade_count": 0, "transcript": [],
        })
        student.setdefault("skills", {})
        for dept in DEPARTMENTS.values():
            for skill in dept["skills"]:
                student["skills"].setdefault(skill, 50.0)
    return uni


def update_university(system, civ, core_specs, timestamp=None):
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    uni = ensure_university(civ, core_specs)
    uni["scan_count"] = int(uni.get("scan_count", 0)) + 1
    research = uni["research"]
    resolved = 0
    for spec in core_specs:
        cid = spec["id"]
        core = (system.get("cores") or {}).get(cid) or {}
        student = uni["students"][cid]
        trades = ((core.get("portfolio") or {}).get("closed_trades") or [])
        seen = min(int(student.get("last_seen_trade_count", 0)), len(trades))
        new_trades = [t for t in trades[seen:] if isinstance(t, dict) and isinstance(t.get("pnl_usd"), (int, float))]
        for trade in new_trades:
            pnl = _num(trade.get("pnl_usd"))
            resolved += 1
            dept = DEPARTMENTS[student["major"]]
            delta = 1.0 if pnl > 0 else -0.8 if pnl < 0 else 0.0
            for skill in dept["skills"]:
                student["skills"][skill] = round(max(0.0, min(100.0, _num(student["skills"].get(skill), 50) + delta)), 2)
            student["research_points"] = round(_num(student.get("research_points")) + (2 if pnl > 0 else 1), 2)
            student["transcript"].append({
                "at": timestamp, "type": "resolved_paper_case", "pnl_usd": round(pnl, 4),
                "major": student["major"], "note": "One observation, not causal evidence.",
            })
        student["last_seen_trade_count"] = len(trades)
        student["transcript"] = student["transcript"][-100:]

        if uni["scan_count"] - int(student.get("last_course_scan", -COURSE_INTERVAL)) >= COURSE_INTERVAL:
            dept = DEPARTMENTS[student["major"]]
            completed = student["completed_courses"]
            course = next((c for c in dept["courses"] if c not in completed), None)
            if course is None:
                weakest = min(dept["skills"], key=lambda s: _num(student["skills"].get(s), 50))
                student["skills"][weakest] = round(min(100.0, _num(student["skills"].get(weakest), 50) + 0.5), 2)
                course = "advanced_review:" + weakest
            else:
                completed.append(course)
                student["credits"] += 3
                for skill in dept["skills"]:
                    student["skills"][skill] = round(min(100.0, _num(student["skills"].get(skill), 50) + 1.0), 2)
            minor = DEPARTMENTS[student["minor"]]
            cross_skill = minor["skills"][uni["scan_count"] % len(minor["skills"])]
            student["skills"][cross_skill] = round(min(100.0, _num(student["skills"].get(cross_skill), 50) + 0.25), 2)
            student["gpa"] = round(sum(_num(v, 50) for v in student["skills"].values()) / max(1, len(student["skills"])), 2)
            student["last_course_scan"] = uni["scan_count"]
            student["transcript"].append({"at": timestamp, "type": "course_completed", "course": course, "gpa": student["gpa"]})

    research["experiments"].append({"at": timestamp, "scan": uni["scan_count"], "resolved_cases": resolved,
                                    "note": "Paper-trading evidence; require out-of-sample validation."})
    research["experiments"] = research["experiments"][-100:]
    if resolved:
        research["papers"].append({"at": timestamp, "title": "Cohort review of paper trades",
                                   "sample_size": resolved, "status": "draft",
                                   "conclusion": "Descriptive only; no causal claim from small samples."})
        research["papers"] = research["papers"][-50:]
    if uni["scan_count"] % 24 == 0 and uni["students"]:
        winner = max(uni["students"].items(), key=lambda item: (_num(item[1].get("gpa"), 50), _num(item[1].get("research_points"))))
        research["grants"].append({"at": timestamp, "core_id": winner[0], "amount_virtual_usd": 100.0,
                                   "reason": "GPA and research process", "real_funds": False})
        research["grants"] = research["grants"][-50:]
    return uni


def education_score_bonus(system, core_id, candidate):
    """A bounded ranking adjustment; cannot bypass any safety or quorum rule."""
    civ = system.get("civilization") or {}
    student = ((civ.get("universities") or {}).get("students") or {}).get(core_id) or {}
    dept = DEPARTMENTS.get(student.get("major", MAJORS.get(core_id, "ml_institute")), {})
    skills = student.get("skills") or {}
    pred = (candidate or {}).get("prediction") or {}
    ensemble = pred.get("strategy_ensemble") or (candidate or {}).get("ensemble") or {}
    strategies = ensemble.get("strategies") or {}
    mapping = {
        "structure": ("market_structure", "trend_following"), "trend": ("trend_following",),
        "breakout": ("breakout",), "risk": ("risk_control",), "position_sizing": ("risk_control",),
        "drawdown": ("risk_control",), "liquidity": ("volume_flow",), "volume": ("volume_flow",),
        "onchain": ("volume_flow",), "statistics": ("mean_reversion",), "calibration": ("mean_reversion",),
        "model_validation": ("mean_reversion",), "features": ("momentum",), "drift": ("momentum",),
        "portfolio": ("risk_control",), "correlation": ("risk_control",), "diversification": ("risk_control",),
        "execution": ("risk_control",), "spread": ("volume_flow",), "market_impact": ("volume_flow",),
    }
    evidence = [_num(strategies.get(name), 50) for skill in dept.get("skills", []) for name in mapping.get(skill, ())]
    if not evidence:
        return 0.0
    skill_mean = sum(_num(skills.get(skill), 50) for skill in dept["skills"]) / max(1, len(dept["skills"]))
    evidence_mean = sum(evidence) / len(evidence)
    return round(max(-2.0, min(2.0, (skill_mean - 50.0) / 25.0 + (evidence_mean - 50.0) / 50.0)), 3)


def university_summary(civ):
    uni = (civ or {}).get("universities") or {}
    students = uni.get("students") or {}
    return {
        "name": uni.get("name", "University of Financial Markets"),
        "departments": len(uni.get("departments") or {}),
        "faculty": len(uni.get("faculty") or {}),
        "students": len(students),
        "completed_courses": sum(len(s.get("completed_courses") or []) for s in students.values()),
        "credits": sum(int(s.get("credits", 0) or 0) for s in students.values()),
        "research_papers": len((uni.get("research") or {}).get("papers") or []),
        "research_grants": len((uni.get("research") or {}).get("grants") or []),
        "average_gpa": round(sum(_num(s.get("gpa"), 50) for s in students.values()) / max(1, len(students)), 2),
    }


def format_university_report(summary):
    return (
        "\n━━ دانشگاه عالی بازارهای مالی ━━\n"
        f"دانشکده‌ها: {summary.get('departments', 0)} | استادان: {summary.get('faculty', 0)} | دانشجویان: {summary.get('students', 0)}\n"
        f"درس‌های گذرانده‌شده: {summary.get('completed_courses', 0)} | واحدها: {summary.get('credits', 0)} | معدل مهارتی: {summary.get('average_gpa', 50):.1f}/100\n"
        f"گزارش‌های پژوهشی: {summary.get('research_papers', 0)} | کمک‌هزینه‌های مجازی: {summary.get('research_grants', 0)}\n"
        "رشته‌ها: پرایس‌اکشن، آمار کمی، ریسک، آن‌چین، اخبار/کلان، یادگیری ماشین، سبد و ریزساختار بازار."
    )
