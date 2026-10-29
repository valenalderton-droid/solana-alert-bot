"""Constrained, evidence-gated self-improvement for the Solana alert bot.

This module may tune a small allowlisted set of strategy parameters from
chronologically held-out, resolved paper alerts. It never edits Python source,
secrets, GitHub workflows, project allowlists, or the hard safety gates.
Internet headlines are retained as research evidence, never executable rules.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMPROVEMENT_FILE = ROOT / "data" / "self_improvement.json"
MIN_RESOLVED = 20
MIN_VALIDATION = 8
PROBABILITY_CANDIDATES = (0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70)
MIN_NET_EDGE_PCT = 0.10
ESTIMATED_ROUND_TRIP_COST_PCT = 0.30


def _parse_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def _read_registry():
    try:
        value = json.loads(IMPROVEMENT_FILE.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value.setdefault("version", 1)
            value.setdefault("rules", [])
            value.setdefault("history", [])
            value.setdefault("research_evidence", [])
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "version": 1,
        "updated_at": None,
        "rules": [],
        "history": [],
        "research_evidence": [],
        "guardrails": [
            "paper-only validation",
            "source code and secrets are immutable to the agent",
            "Solana-only and approved non-meme checks cannot be disabled",
            "hard liquidity, stale-data, stop and risk-off protections cannot be removed",
            "no rule changes without chronological holdout evidence"
        ]
    }


def _save_registry(registry):
    IMPROVEMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    temp = IMPROVEMENT_FILE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(IMPROVEMENT_FILE)


def _joined_resolved(predictions, outcomes):
    pred_by_id = {
        str(row.get("signal_id")): row
        for row in predictions
        if isinstance(row, dict) and row.get("signal_id")
    }
    rows = []
    for outcome in outcomes:
        if not isinstance(outcome, dict) or outcome.get("result") == "data_gap":
            continue
        signal_id = str(outcome.get("signal_id", ""))
        pred = pred_by_id.get(signal_id)
        if not pred:
            continue
        try:
            ret = float(outcome["return_pct"])
            prob = float(pred["probability"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(ret) or not math.isfinite(prob) or not 0 <= prob <= 1:
            continue
        created = _parse_time(pred.get("created_at"))
        rows.append({
            "signal_id": signal_id,
            "created_at": created or datetime.min.replace(tzinfo=timezone.utc),
            "probability": prob,
            "return_pct": ret - ESTIMATED_ROUND_TRIP_COST_PCT
        })
    rows.sort(key=lambda row: (row["created_at"], row["signal_id"]))
    # Ignore duplicate outcomes for the same signal.
    unique, seen = [], set()
    for row in rows:
        if row["signal_id"] not in seen:
            seen.add(row["signal_id"])
            unique.append(row)
    return unique


def _stats(rows):
    if not rows:
        return {"n": 0, "mean_net_return_pct": None, "win_rate": None}
    returns = [row["return_pct"] for row in rows]
    return {
        "n": len(rows),
        "mean_net_return_pct": round(sum(returns) / len(returns), 4),
        "win_rate": round(sum(value > 0 for value in returns) / len(returns), 4)
    }


def choose_probability_threshold(predictions, outcomes, current_threshold=0.60):
    """Choose only from a fixed allowlist and only on chronological holdout."""
    rows = _joined_resolved(predictions, outcomes)
    if len(rows) < MIN_RESOLVED:
        return {
            "threshold": round(float(current_threshold), 3),
            "changed": False,
            "reason": f"Need at least {MIN_RESOLVED} joined resolved alerts; have {len(rows)}",
            "sample_size": len(rows)
        }

    split = int(len(rows) * 0.60)
    train, validation = rows[:split], rows[split:]
    if len(validation) < MIN_VALIDATION:
        return {
            "threshold": round(float(current_threshold), 3),
            "changed": False,
            "reason": "Chronological validation holdout is too small",
            "sample_size": len(rows)
        }

    # Require enough training evidence that a candidate isn't a tiny-sample accident.
    eligible = [threshold for threshold in PROBABILITY_CANDIDATES
                if len([row for row in train if row["probability"] >= threshold]) >= 8]
    baseline = _stats([row for row in validation
                       if row["probability"] >= float(current_threshold)])
    if baseline["n"] < MIN_VALIDATION:
        return {
            "threshold": round(float(current_threshold), 3),
            "changed": False,
            "reason": "Baseline threshold has too few validation examples",
            "sample_size": len(rows),
            "validation_baseline": baseline
        }

    best = None
    for threshold in eligible:
        sample = [row for row in validation if row["probability"] >= threshold]
        stats = _stats(sample)
        if stats["n"] < MIN_VALIDATION:
            continue
        if stats["mean_net_return_pct"] is None or baseline["mean_net_return_pct"] is None:
            continue
        edge = stats["mean_net_return_pct"] - baseline["mean_net_return_pct"]
        if (stats["mean_net_return_pct"] > 0 and edge >= MIN_NET_EDGE_PCT
                and stats["win_rate"] >= baseline["win_rate"] - 0.05):
            candidate = (edge, stats["mean_net_return_pct"], threshold, stats)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return {
            "threshold": round(float(current_threshold), 3),
            "changed": False,
            "reason": "No allowlisted threshold beat baseline on holdout by the required net edge",
            "sample_size": len(rows),
            "validation_baseline": baseline
        }

    threshold = float(best[2])
    # Bound any one update to a two-percentage-point step to reduce oscillation.
    threshold = min(float(current_threshold) + 0.02,
                    max(float(current_threshold) - 0.02, threshold))
    threshold = min(max(threshold, min(PROBABILITY_CANDIDATES)),
                    max(PROBABILITY_CANDIDATES))
    changed = abs(threshold - float(current_threshold)) >= 0.005
    return {
        "threshold": round(threshold, 3),
        "changed": changed,
        "reason": "Candidate improved chronological holdout net return and retained win-rate guardrail",
        "sample_size": len(rows),
        "validation_baseline": baseline,
        "validation_candidate": best[3],
        "selected_candidate": best[2]
    }


def update_self_improvement(predictions, outcomes, news_observations=None,
                            current_threshold=0.60, now=None):
    """Update the parameter rule registry and save a concise audit trail."""
    now = now or datetime.now(timezone.utc)
    registry = _read_registry()
    decision = choose_probability_threshold(predictions, outcomes, current_threshold)
    rules = registry.setdefault("rules", [])
    active = next((rule for rule in rules
                   if rule.get("id") == "validated_probability_floor"), None)

    if active is None:
        active = {
            "id": "validated_probability_floor",
            "type": "min_probability",
            "value": round(float(current_threshold), 3),
            "enabled": True,
            "status": "baseline",
            "created_at": now.isoformat()
        }
        rules.append(active)

    before = float(active.get("value", current_threshold))
    # The environment/user threshold remains the floor; self-improvement can
    # tune only within a narrow band around it, never loosen the configured base.
    configured_floor = float(current_threshold)
    candidate = float(decision["threshold"])
    if decision["changed"] and candidate >= configured_floor:
        active["value"] = round(min(0.70, max(configured_floor, candidate)), 3)
        active["status"] = "validated_candidate"
        active["last_validated_at"] = now.isoformat()
        active["validation"] = {
            key: value for key, value in decision.items()
            if key in ("reason", "sample_size", "validation_baseline",
                       "validation_candidate", "selected_candidate")
        }
        if active["value"] != before:
            active["updated_at"] = now.isoformat()
    elif active.get("status") != "baseline":
        # Do not delete a proven rule due to missing data. Mark as pending review.
        active["status"] = "unchanged_insufficient_or_no_edge"

    # Store internet research only as source-linked evidence. Headlines are not
    # allowed to create or execute rules directly.
    for item in (news_observations or [])[-12:]:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        evidence = {
            "title": str(item.get("title", ""))[:250],
            "url": str(item.get("url", ""))[:500],
            "published_at": item.get("published_at"),
            "sentiment": item.get("sentiment", "NEUTRAL / UNCLEAR"),
            "confidence": item.get("confidence", "UNKNOWN"),
            "feed_category": item.get("feed_category", "Uncategorized")
        }
        if evidence not in registry.setdefault("research_evidence", []):
            registry["research_evidence"].append(evidence)
    registry["research_evidence"] = registry.get("research_evidence", [])[-100:]
    registry.setdefault("history", []).append({
        "at": now.isoformat(),
        "action": "updated" if active.get("value") != before else "kept",
        "rule_id": active["id"],
        "old_value": round(before, 3),
        "new_value": round(float(active.get("value", before)), 3),
        "decision": decision
    })
    registry["history"] = registry["history"][-200:]
    _save_registry(registry)

    return {
        "min_probability": round(max(configured_floor, float(active.get("value", configured_floor))), 3),
        "changed": active.get("value") != before,
        "rule_status": active.get("status"),
        "decision": decision,
        "registry_path": str(IMPROVEMENT_FILE.relative_to(ROOT))
    }
