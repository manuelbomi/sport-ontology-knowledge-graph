"""
entity_resolution.py
────────────────────
Entity resolution (also called *entity linking* or *record linkage*) is the process
of recognising that multiple strings refer to the same real-world entity.

This is a constant headache in fan data: the same person might appear as
  "Marcus T."   in the ticketing system
  "marcus_la"   on the mobile app
  "@MarcusT"    on social media
  "MARCUS"      in an in-venue kiosk scan

Without resolving these, you end up with 4 fragmented records instead of one.

Three strategies are shown here, in increasing sophistication:

  1. Exact alias lookup        — pre-registered synonyms in a dict
  2. Fuzzy string matching     — difflib SequenceMatcher (no ML needed)
  3. Graph-based resolution    — find duplicates in the knowledge graph by
                                 triangulating on shared attributes/neighbours
"""

import difflib
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENTITY REGISTRY
#    A simple in-memory store of canonical entities + their known aliases.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CanonicalEntity:
    """One ground-truth entity with all the names we've seen for it."""
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    def all_names(self) -> list[str]:
        return [self.canonical_name] + self.aliases


class EntityRegistry:
    """
    Stores canonical entities and supports three resolution methods.

    In production this would be backed by a database and use NER +
    ML-based similarity instead of difflib, but the logic is the same.
    """

    def __init__(self):
        self._entities: dict[str, CanonicalEntity] = {}
        self._alias_index: dict[str, str] = {}  # alias_lower → entity_id

    def register(self, entity: CanonicalEntity):
        self._entities[entity.entity_id] = entity
        for name in entity.all_names():
            self._alias_index[name.lower()] = entity.entity_id

    def add_alias(self, entity_id: str, alias: str):
        if entity_id in self._entities:
            self._entities[entity_id].aliases.append(alias)
            self._alias_index[alias.lower()] = entity_id

    # ── Method 1: exact alias lookup ─────────────────────────────────────────

    def resolve_exact(self, mention: str) -> Optional[CanonicalEntity]:
        """O(1) lookup against the pre-registered alias table."""
        eid = self._alias_index.get(mention.strip().lower())
        return self._entities.get(eid) if eid else None

    # ── Method 2: fuzzy string matching ──────────────────────────────────────

    def resolve_fuzzy(self, mention: str, threshold: float = 0.70) -> Optional[CanonicalEntity]:
        """
        Score `mention` against every known alias using SequenceMatcher.
        Returns the best match above `threshold`, or None.

        Good enough for typos and abbreviated names.
        Not great for nicknames ("LBJ" vs "LeBron James") — that's what
        the alias table handles.
        """
        mention_lower = mention.lower().strip()
        best_score = 0.0
        best_id = None

        for alias, eid in self._alias_index.items():
            score = difflib.SequenceMatcher(None, mention_lower, alias).ratio()
            if score > best_score:
                best_score = score
                best_id = eid

        if best_score >= threshold:
            return self._entities.get(best_id)
        return None

    def resolve(self, mention: str) -> Optional[CanonicalEntity]:
        """Try exact first, fall back to fuzzy."""
        return self.resolve_exact(mention) or self.resolve_fuzzy(mention)

    def all_entities(self, entity_type: Optional[str] = None):
        if entity_type:
            return [e for e in self._entities.values() if e.entity_type == entity_type]
        return list(self._entities.values())


# ─────────────────────────────────────────────────────────────────────────────
# 2. POPULATE WITH NBA DATA
# ─────────────────────────────────────────────────────────────────────────────

registry = EntityRegistry()

# Players — lots of common nicknames to register
registry.register(CanonicalEntity(
    entity_id="player_2544",
    canonical_name="LeBron James",
    entity_type="Player",
    aliases=["LBJ", "King James", "lebron", "L. James", "lebron james 23",
             "the king", "bron", "LeBron"],
    attributes={"team": "Lakers", "position": "Forward", "number": 23},
))

registry.register(CanonicalEntity(
    entity_id="player_201939",
    canonical_name="Stephen Curry",
    entity_type="Player",
    aliases=["Steph", "Chef Curry", "steph curry", "S. Curry",
             "stephen curry", "curry 30", "wardell curry"],
    attributes={"team": "Warriors", "position": "Guard", "number": 30},
))

registry.register(CanonicalEntity(
    entity_id="player_1628",
    canonical_name="Jayson Tatum",
    entity_type="Player",
    aliases=["JT", "tatum", "jayson tatum", "J. Tatum", "JT0"],
    attributes={"team": "Celtics", "position": "Forward", "number": 0},
))

registry.register(CanonicalEntity(
    entity_id="player_203507",
    canonical_name="Giannis Antetokounmpo",
    entity_type="Player",
    aliases=["Giannis", "Greek Freak", "giannis", "G. Antetokounmpo",
             "the greek freak", "giannis 34"],
    attributes={"team": "Bucks", "position": "Forward", "number": 34},
))

# Fans — typical fragmentation across systems
registry.register(CanonicalEntity(
    entity_id="fan_0001",
    canonical_name="Marcus Thompson",
    entity_type="Fan",
    aliases=["Marcus T.", "marcus_la", "@MarcusT", "MARCUS", "M. Thompson"],
    attributes={"city": "Los Angeles", "follows": "Lakers"},
))

registry.register(CanonicalEntity(
    entity_id="fan_0002",
    canonical_name="Aisha Rahman",
    entity_type="Fan",
    aliases=["aisha_boston", "A. Rahman", "AishaR", "aisha"],
    attributes={"city": "Boston", "follows": "Celtics"},
))

# Teams
registry.register(CanonicalEntity(
    entity_id="team_lakers",
    canonical_name="Los Angeles Lakers",
    entity_type="Team",
    aliases=["Lakers", "LA Lakers", "lakers", "LAL", "the lakers"],
))

registry.register(CanonicalEntity(
    entity_id="team_celtics",
    canonical_name="Boston Celtics",
    entity_type="Team",
    aliases=["Celtics", "BOS", "boston celtics", "the celtics"],
))

registry.register(CanonicalEntity(
    entity_id="team_warriors",
    canonical_name="Golden State Warriors",
    entity_type="Team",
    aliases=["Warriors", "GSW", "golden state", "Dubs", "the warriors"],
))


# ─────────────────────────────────────────────────────────────────────────────
# 3. METHOD 3: GRAPH-BASED DUPLICATE DETECTION
#    Sometimes you don't have a pre-built alias table and need to *discover*
#    duplicates in an existing dataset.
#    Strategy: if two "Fan" records share a city + a followed team,
#    flag them as potential duplicates for human review.
# ─────────────────────────────────────────────────────────────────────────────

def find_potential_duplicates(fan_records: list[dict]) -> list[tuple]:
    """
    Heuristic: two fan records are *possibly* the same person if they share
    both their city and their followed team.

    In practice you'd add email domain, device fingerprint, purchase history, etc.
    This is the simplest version of "triangulation via shared attributes".
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for record in fan_records:
        key = (record.get("city", ""), record.get("follows", ""))
        buckets[key].append(record)

    duplicates = []
    for key, group in buckets.items():
        if len(group) > 1:
            # Every pair in this group is a candidate duplicate
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    duplicates.append((group[i], group[j], key))
    return duplicates


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("METHOD 1: Exact alias lookup")
    print("=" * 60)
    test_mentions = [
        "LBJ",          # registered alias → LeBron James
        "King James",   # registered alias → LeBron James
        "Chef Curry",   # → Stephen Curry
        "JT",           # → Jayson Tatum
        "marcus_la",    # → Marcus Thompson (fan)
        "Lakers",       # → Los Angeles Lakers
        "GSW",          # → Golden State Warriors
    ]
    for mention in test_mentions:
        entity = registry.resolve_exact(mention)
        result = entity.canonical_name if entity else "NOT FOUND"
        print(f"  {mention!r:20s} → {result}")

    print("\n" + "=" * 60)
    print("METHOD 2: Fuzzy matching (handles typos / partial names)")
    print("=" * 60)
    fuzzy_tests = [
        "lebrown james",    # typo
        "stephan curry",    # misspelling
        "jayson tatun",     # typo
        "gianniss",         # close to "giannis"
        "LA Lakers",        # close to registered alias "LA Lakers"
    ]
    for mention in fuzzy_tests:
        entity = registry.resolve_fuzzy(mention)
        result = entity.canonical_name if entity else "no match above threshold"
        print(f"  {mention!r:22s} → {result}")

    print("\n" + "=" * 60)
    print("METHOD 3: Graph-based duplicate detection")
    print("=" * 60)
    # Simulate incoming fan records from different systems
    incoming_fans = [
        {"source": "ticketing", "name": "Marcus T.",   "city": "Los Angeles", "follows": "Lakers"},
        {"source": "app",       "name": "marcus_la",   "city": "Los Angeles", "follows": "Lakers"},
        {"source": "social",    "name": "@MarcusT",    "city": "Los Angeles", "follows": "Lakers"},
        {"source": "ticketing", "name": "Aisha R.",    "city": "Boston",      "follows": "Celtics"},
        {"source": "app",       "name": "aisha_boston","city": "Boston",      "follows": "Celtics"},
        {"source": "kiosk",     "name": "Jordan M.",   "city": "San Francisco","follows": "Warriors"},
    ]

    dupes = find_potential_duplicates(incoming_fans)
    for r1, r2, key in dupes:
        print(f"\n  Possible duplicate (city={key[0]}, team={key[1]}):")
        print(f"    [{r1['source']}] {r1['name']}")
        print(f"    [{r2['source']}] {r2['name']}")
        # Confirm via exact or fuzzy resolution
        e1 = registry.resolve(r1["name"])
        e2 = registry.resolve(r2["name"])
        if e1 and e2 and e1.entity_id == e2.entity_id:
            print(f"    ✓ Confirmed same entity: {e1.canonical_name} ({e1.entity_id})")
        else:
            print(f"    ? Could not auto-confirm — flagged for human review")
