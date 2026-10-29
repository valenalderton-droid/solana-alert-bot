"""Safe editor for the bot's narrowly allowlisted experience-rule JSON files.

This editor cannot touch Python source, workflow files, project allowlists,
secrets, or arbitrary paths. All edits are validated and atomically replaced.
"""
import json
import math
import os
from pathlib import Path

RULE_SCHEMAS = {
    "avoid_entry_chase": ("entry_deviation_pct", "gte", 1.5, 4.0),
    "avoid_thin_liquidity": ("liquidity_usd", "lt", 100000.0, 500000.0),
    "avoid_low_volume": ("volume_24h_usd", "lt", 100000.0, 500000.0),
    "avoid_weak_ensemble": ("ensemble_score", "lt", 50.0, 75.0),
    "avoid_lower_probability": ("probability", "lt", 0.55, 0.70),
    "avoid_weak_quality": ("quality_score", "lt", 55.0, 80.0),
}
MAX_FILE_BYTES = 32768


class ManagedFileEditor:
    """CRUD operations limited to fixed, bot-owned JSON rule names."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def _path(self, rule_id):
        rule_id = str(rule_id or "")
        if rule_id not in RULE_SCHEMAS:
            raise ValueError("Rule ID is not allowlisted")
        path = (self.directory / (rule_id + ".json")).resolve()
        if path.parent != self.directory or path.suffix != ".json":
            raise ValueError("Unsafe managed-file path")
        return path

    def validate(self, rule):
        if not isinstance(rule, dict):
            raise ValueError("Rule must be an object")
        rule_id = str(rule.get("id", ""))
        if rule_id not in RULE_SCHEMAS:
            raise ValueError("Rule ID is not allowlisted")
        field, operator, low, high = RULE_SCHEMAS[rule_id]
        if rule.get("field") != field or rule.get("operator") != operator:
            raise ValueError("Rule field/operator does not match its fixed schema")
        try:
            value = float(rule["value"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("Rule threshold must be numeric")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Rule threshold is outside its safe bounds")
        if rule.get("enabled") is not True:
            raise ValueError("Only enabled validated rules can be written")
        evidence = rule.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("Rule evidence must be an object")
        return {
            "schema_version": 1,
            "managed_by": "solana-alert-bot-self-improver",
            "id": rule_id,
            "enabled": True,
            "status": str(rule.get("status", "experience_derived"))[:80],
            "field": field,
            "operator": operator,
            "value": value,
            "reason": str(rule.get("reason", ""))[:500],
            "evidence": evidence,
            "updated_at": str(rule.get("updated_at", ""))[:80],
        }

    def read(self, rule_id):
        path = self._path(rule_id)
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Managed file exceeds size limit")
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(value, dict) or value.get("id") != rule_id:
            raise ValueError("Existing managed file failed identity validation")
        return value

    def write(self, rule):
        payload = self.validate(rule)
        path = self._path(payload["id"])
        self.directory.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if len(serialized.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("Managed file exceeds size limit")
        previous = self.read(payload["id"])
        if previous is not None:
            old = dict(previous)
            new = dict(payload)
            old.pop("updated_at", None)
            new.pop("updated_at", None)
            if old == new:
                return "unchanged"
        temp = path.with_suffix(".json.tmp")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        return "created" if previous is None else "edited"

    def delete(self, rule_id):
        path = self._path(rule_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
