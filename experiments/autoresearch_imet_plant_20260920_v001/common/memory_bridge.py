"""Lossless transport prototype; no LLM, training, or benefit claim.

The calling agent supplies literal queries, reads complete matching cards, then
records adoption/rejection. Original packs remain unchanged. This deliberately
does not reproduce MLEvolve's multi-agent hypothesis/review pipeline.
"""
import hashlib
import json
from pathlib import Path

LEVELS = {"architecture", "tuning", "debug"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class MemoryBridge:
    def __init__(self, specs, task, data_manifest_sha256):
        self.packs = {}
        self.cards = {}
        for channel, spec in specs.items():
            raw = Path(spec["path"]).read_bytes()
            if digest(raw) != spec["sha256"]:
                raise ValueError("pack hash mismatch: " + channel)
            pack = json.loads(raw)
            if pack.get("data_manifest_sha256") != data_manifest_sha256:
                raise ValueError("data manifest mismatch: " + channel)
            if pack.get("task_id", task) != task:
                raise ValueError("task mismatch: " + channel)
            if pack.get("contains_source_code") or pack.get("contains_source_weights"):
                raise ValueError("source code/weight transfer prohibited")
            if pack.get("schema") not in {"mlevolve_search_experience_v1", "inner_tuning_memory_v1"}:
                raise ValueError("unsupported schema: " + str(pack.get("schema")))
            self.packs[channel] = {"sha256": digest(raw), "metadata": {k: v for k, v in pack.items() if k != "cards"}}
            for card in pack["cards"]:
                if card.get("task", task) != task:
                    raise ValueError("card task mismatch")
                key = channel + ":" + card["id"]
                if key in self.cards:
                    raise ValueError("duplicate card: " + key)
                self.cards[key] = card

    def catalogue(self):
        return [{"key": key, **{field: card.get(field) for field in
                ("granularity", "title", "when", "match_terms")}}
                for key, card in self.cards.items()]

    def grep(self, terms, granularity=None):
        if not terms or any(not isinstance(t, str) or not t or len(t) > 120 or
                            "\n" in t or "\x00" in t for t in terms):
            raise ValueError("short nonempty literal terms required")
        if granularity is not None and granularity not in LEVELS:
            raise ValueError("unsupported granularity")
        return [key for key, card in self.cards.items()
                if (granularity is None or card.get("granularity") == granularity)
                and any(t.casefold() in json.dumps(card, ensure_ascii=False).casefold() for t in terms)]

    def read(self, keys):
        # No projection, truncation, or deduplication across differing channels.
        return {"packs": self.packs, "cards": [{"key": key, "body": self.cards[key]}
                                                for key in dict.fromkeys(keys)]}

    def prompt(self, keys):
        return json.dumps(self.read(keys), ensure_ascii=False, allow_nan=False)

    def receipt(self, keys, decisions):
        selected = set(keys)
        if any(d.get("key") not in selected or d.get("action") not in
               {"adopt", "adapt", "reject", "defer"} or not d.get("reason") for d in decisions):
            raise ValueError("decisions must reference read cards and explain an action")
        body = self.prompt(keys)
        return {"schema": "autoresearch_memory_transport_v1", "pack_hashes":
                {k: v["sha256"] for k, v in self.packs.items()}, "read_ids": list(dict.fromkeys(keys)),
                "prompt_sha256": digest(body.encode()), "prompt_body": body,
                "decisions": decisions, "execution_verified": False}
