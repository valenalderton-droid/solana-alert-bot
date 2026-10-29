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
RULE_FILES_DIR = ROOT / "data" / "experience_rules"
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



def diagnose_failure_modes(predictions, outcomes, min_group_size=3):
    """Create cautious research priorities from resolved paper-alert errors."""
    prediction_by_id = {
        str(row.get("signal_id")): row for row in (predictions or [])
        if isinstance(row, dict) and row.get("signal_id")
    }
    joined, seen = [], set()
    for outcome in outcomes or []:
        if not isinstance(outcome, dict) or outcome.get("result") == "data_gap":
            continue
        signal_id = str(outcome.get("signal_id", ""))
        pred = prediction_by_id.get(signal_id)
        if not signal_id or signal_id in seen or not pred:
            continue
        try:
            ret = float(outcome.get("return_pct"))
            probability = float(pred.get("probability", 0) or 0)
            liquidity = float(pred.get("liquidity_usd", 0) or 0)
            volume = float(pred.get("volume_24h_usd", 0) or 0)
            deviation = float(pred.get("entry_deviation_pct", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(v) for v in (ret, probability, liquidity, volume, deviation)):
            continue
        seen.add(signal_id)
        ensemble = pred.get("strategy_ensemble") or {}
        quality = pred.get("signal_quality") or {}
        risk_flags = ensemble.get("risk_flags", []) if isinstance(ensemble, dict) else []
        tags = set()
        if deviation >= 2.0:
            tags.add("entry_chase")
        if liquidity < 250000:
            tags.add("thin_liquidity")
        if volume < 250000:
            tags.add("low_volume")
        if float(ensemble.get("score", 0) or 0) < 65:
            tags.add("weak_ensemble")
        if any("volatil" in str(flag).lower() or "risk" in str(flag).lower()
               for flag in (risk_flags or [])):
            tags.add("high_risk_flags")
        if probability < 0.65:
            tags.add("lower_probability")
        if float(quality.get("score", 100) or 0) < 70:
            tags.add("weak_quality")
        joined.append({"return_pct": ret - ESTIMATED_ROUND_TRIP_COST_PCT, "tags": tags})

    tag_to_topic = {
        "entry_chase": "strategy", "thin_liquidity": "liquidity",
        "low_volume": "liquidity", "weak_ensemble": "strategy",
        "high_risk_flags": "strategy", "lower_probability": "strategy",
        "weak_quality": "strategy"
    }
    groups = {}
    for tag, topic in tag_to_topic.items():
        sample = [row["return_pct"] for row in joined if tag in row["tags"]]
        if len(sample) < min_group_size:
            continue
        mean_return = sum(sample) / len(sample)
        loss_rate = sum(value < 0 for value in sample) / len(sample)
        groups[tag] = {
            "n": len(sample),
            "mean_net_return_pct": round(mean_return, 4),
            "negative_return_rate": round(loss_rate, 4),
            "research_topic": topic,
            "warning": "association only; not a causal finding"
        }
    focus = []
    for tag, stats in sorted(groups.items(), key=lambda item: (item[1]["mean_net_return_pct"], -item[1]["n"])):
        if stats["mean_net_return_pct"] < 0 or stats["negative_return_rate"] >= 0.60:
            if stats["research_topic"] not in focus:
                focus.append(stats["research_topic"])
    return {
        "sample_size": len(joined), "minimum_group_size": int(min_group_size),
        "groups": groups, "research_focus": focus[:4],
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

EXPERIENCE_RULE_DEFINITIONS = {
    "avoid_entry_chase": {
        "tag": "entry_chase", "field": "entry_deviation_pct", "operator": "gte",
        "value": 2.0, "reason": "Repeated losing outcomes after chasing a moved price"
    },
    "avoid_thin_liquidity": {
        "tag": "thin_liquidity", "field": "liquidity_usd", "operator": "lt",
        "value": 250000.0, "reason": "Repeated losses in thin-liquidity pools"
    },
    "avoid_low_volume": {
        "tag": "low_volume", "field": "volume_24h_usd", "operator": "lt",
        "value": 250000.0, "reason": "Repeated losses in low-volume markets"
    },
    "avoid_weak_ensemble": {
        "tag": "weak_ensemble", "field": "ensemble_score", "operator": "lt",
        "value": 65.0, "reason": "Repeated losses when strategy agreement is weak"
    },
    "avoid_lower_probability": {
        "tag": "lower_probability", "field": "probability", "operator": "lt",
        "value": 0.65, "reason": "Repeated losses on lower-confidence model signals"
    },
    "avoid_weak_quality": {
        "tag": "weak_quality", "field": "quality_score", "operator": "lt",
        "value": 70.0, "reason": "Repeated losses on weak-quality setups"
    }
}


def build_experience_rules(diagnostics, minimum_samples=12,
                           maximum_negative_rate=0.70):
    """Derive auditable candidate filters from repeated resolved outcomes.

    These are strategy-level rules only. They cannot change meme exclusion,
    network identity, data validation, or source code. A rule is activated only
    after a sufficiently large group shows negative net expectancy and a high
    negative-return rate after estimated costs.
    """
    groups = (diagnostics or {}).get("groups", {})
    rules = []
    for rule_id, definition in EXPERIENCE_RULE_DEFINITIONS.items():
        stats = groups.get(definition["tag"])
        if not isinstance(stats, dict):
            continue
        try:
            n = int(stats.get("n", 0))
            mean_return = float(stats.get("mean_net_return_pct"))
            negative_rate = float(stats.get("negative_return_rate"))
        except (TypeError, ValueError, OverflowError):
            continue
        if (n < minimum_samples or not math.isfinite(mean_return)
                or not math.isfinite(negative_rate) or mean_return >= 0
                or negative_rate < maximum_negative_rate):
            continue
        rules.append({
            "id": rule_id, "enabled": True, "status": "experience_derived",
            "field": definition["field"], "operator": definition["operator"],
            "value": definition["value"], "reason": definition["reason"],
            "evidence": {"samples": n, "mean_net_return_pct": round(mean_return, 4),
                         "negative_return_rate": round(negative_rate, 4)},
            "updated_at": datetime.now(timezone.utc).isoformat()
        })
    return rules


def sync_experience_rule_files(rules, now=None):
    """Manage only validated allowlisted rule JSON files via the safe file editor."""
    from managed_file_editor import ManagedFileEditor

    now = now or datetime.now(timezone.utc)
    editor = ManagedFileEditor(RULE_FILES_DIR)
    active = {}
    rejected = []
    for rule in rules or []:
        if not isinstance(rule, dict) or rule.get("enabled") is not True:
            continue
        try:
            payload = editor.validate(rule)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"id": str(rule.get("id", ""))[:80], "reason": str(exc)[:160]})
            continue
        active[payload["id"]] = {**rule, **payload}
        if not active[payload["id"]].get("updated_at"):
            active[payload["id"]]["updated_at"] = now.isoformat()

    actions = {"created": [], "edited": [], "deleted": [], "active": [], "unchanged": [], "rejected": rejected}
    for rule_id, rule in active.items():
        try:
            result = editor.write(rule)
        except (OSError, ValueError, TypeError) as exc:
            actions["rejected"].append({"id": rule_id, "reason": str(exc)[:160]})
            continue
        actions[result].append(rule_id + ".json")
        actions["active"].append(rule_id + ".json")

    # Only the editor's hard-coded allowlist can be deleted. Unrelated files stay untouched.
    for rule_id in editor_ids():
        if rule_id not in active and editor.delete(rule_id):
            actions["deleted"].append(rule_id + ".json")
    return actions


def editor_ids():
    """Return the static IDs the managed editor is allowed to own."""
    from managed_file_editor import RULE_SCHEMAS
    return tuple(RULE_SCHEMAS)


def evaluate_experience_rules(candidate, rules):
    """Return the first active learned rule that matches a new candidate."""
    candidate = candidate if isinstance(candidate, dict) else {}
    for rule in rules or []:
        if not isinstance(rule, dict) or not rule.get("enabled", False):
            continue
        field = rule.get("field")
        try:
            value = float(candidate[field])
            threshold = float(rule["value"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(value) or not math.isfinite(threshold):
            continue
        operator = rule.get("operator")
        matched = (value >= threshold if operator == "gte" else
                   value < threshold if operator == "lt" else False)
        if matched:
            return {"blocked": True, "rule_id": rule.get("id"),
                    "reason": rule.get("reason", "Matched an experience-derived rule"),
                    "evidence": rule.get("evidence", {})}
    return {"blocked": False, "rule_id": None, "reason": "No learned rule matched"}


def update_self_improvement(predictions, outcomes, news_observations=None,
                            current_threshold=0.60, now=None):
    """Update the parameter rule registry and save a concise audit trail."""
    now = now or datetime.now(timezone.utc)
    registry = _read_registry()
    registry["learning_diagnostics"] = diagnose_failure_modes(predictions, outcomes)
    # Rebuild strategy rules from resolved outcomes each run. A rule that no
    # longer meets the evidence criteria is automatically retired.
    registry["experience_rules"] = build_experience_rules(registry["learning_diagnostics"])
    file_actions = sync_experience_rule_files(registry["experience_rules"], now=now)
    rules = registry.setdefault("rules", [])
    active = next((rule for rule in rules
                   if rule.get("id") == "validated_probability_floor"), None)

    latest_outcome = max(
        (str(row.get("resolved_at", row.get("signal_id", "")))
         for row in outcomes if isinstance(row, dict)),
        default=""
    )
    signature = f"{len(outcomes)}:{latest_outcome}"
    last_eval = _parse_time(registry.get("last_evaluated_at"))
    if (last_eval and registry.get("last_outcome_signature") == signature
            and (now - last_eval).total_seconds() < 6 * 3600):
        current_value = float(current_threshold)
        if active and active.get("enabled", True):
            current_value = max(current_value, float(active.get("value", current_value)))
        return {
            "min_probability": round(current_value, 3),
            "changed": False,
            "rule_status": active.get("status", "baseline") if active else "baseline",
            "decision": {
                "threshold": round(current_value, 3),
                "changed": False,
                "reason": "Self-improvement cooldown: no new resolved outcomes; re-evaluate within six hours",
                "sample_size": len(_joined_resolved(predictions, outcomes))
            },
            "registry_path": "data/self_improvement.json",
            "experience_rules": registry.get("experience_rules", []),
            "experience_rule_count": len(registry.get("experience_rules", [])),
            "rule_file_actions": file_actions
        }

    decision = choose_probability_threshold(predictions, outcomes, current_threshold)

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
        active["enabled"] = True
        active["failed_validations"] = 0
        active["status"] = "validated_candidate"
        active["last_validated_at"] = now.isoformat()
        active["validation"] = {
            key: value for key, value in decision.items()
            if key in ("reason", "sample_size", "validation_baseline",
                       "validation_candidate", "selected_candidate")
        }
        if active["value"] != before:
            active["updated_at"] = now.isoformat()
    elif (decision.get("sample_size", 0) >= MIN_RESOLVED
          and "validation_baseline" in decision
          and "No allowlisted threshold beat baseline" in decision.get("reason", "")):
        # Retire only the learned soft rule after repeated failed holdout checks.
        # The configured base threshold and hard safety rules remain active.
        active["failed_validations"] = int(active.get("failed_validations", 0)) + 1
        if active["failed_validations"] >= 3:
            active["enabled"] = False
            active["status"] = "retired_after_failed_validation"
        else:
            active["status"] = "pending_revalidation"

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
    registry["last_evaluated_at"] = now.isoformat()
    registry["last_outcome_signature"] = signature
    registry.setdefault("history", []).append({
        "at": now.isoformat(),
        "action": ("retired" if not active.get("enabled", True)
                   else "updated" if active.get("value") != before else "kept"),
        "rule_id": active["id"],
        "old_value": round(before, 3),
        "new_value": round(float(active.get("value", before)), 3),
        "decision": decision
    })
    registry["history"] = registry["history"][-200:]
    _save_registry(registry)

    return {
        "min_probability": round(max(
            configured_floor,
            float(active.get("value", configured_floor)) if active.get("enabled", True) else configured_floor
        ), 3),
        "changed": active.get("value") != before or not active.get("enabled", True),
        "rule_status": active.get("status"),
        "decision": decision,
        "registry_path": "data/self_improvement.json",
        "experience_rules": registry.get("experience_rules", []),
        "experience_rule_count": len(registry.get("experience_rules", [])),
        "rule_file_actions": file_actions
    }
