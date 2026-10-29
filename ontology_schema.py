"""
ontology_schema.py
──────────────────
Defines the NBA fan domain ontology as Python classes.

An ontology answers: "What *kinds* of things exist in this domain,
and what are the rules about how they can relate to each other?"

Here we model it with three building blocks:

  EntityType   ─── a class of things (Player, Team, Fan, ...)
  RelationType ─── a named, directed link between two EntityTypes
  Ontology     ─── the complete blueprint (all types + all allowed relations)

Running this file prints a summary and a few validation examples.
"""

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENTITY TYPES
#    Think of these as classes in a class hierarchy. A Player IS-A Person.
#    A Team is NOT a Person, but a Team HAS Players.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityType:
    name: str
    parent: Optional[str] = None   # IS-A hierarchy (e.g. Player is-a Person)
    description: str = ""

    def __repr__(self):
        parent_str = f" (is-a {self.parent})" if self.parent else ""
        return f"EntityType({self.name}{parent_str})"


# Every EntityType we recognize in this domain
ENTITY_TYPES = {
    # Abstract root — everything is a "Thing"
    "Thing":   EntityType("Thing",  description="root concept"),

    # People
    "Person":  EntityType("Person", parent="Thing"),
    "Player":  EntityType("Player", parent="Person",
                          description="an NBA athlete on a team roster"),
    "Fan":     EntityType("Fan",    parent="Person",
                          description="a person who follows one or more teams"),
    "Coach":   EntityType("Coach",  parent="Person"),

    # Organizations & places
    "Team":    EntityType("Team",   parent="Thing",
                          description="an NBA franchise"),
    "Venue":   EntityType("Venue",  parent="Thing",
                          description="arena or stadium"),
    "City":    EntityType("City",   parent="Thing"),

    # Occurrences
    "Event":   EntityType("Event",  parent="Thing",
                          description="a specific game, playoff series, or all-star event"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. RELATION TYPES
#    A RelationType is a named, directed edge between two EntityTypes.
#    It has a *domain* (the source type) and a *range* (the target type).
#
#    This is what separates an ontology from a plain schema:
#    constraints on what can connect to what.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RelationType:
    name: str
    domain: str    # source EntityType
    range: str     # target EntityType
    inverse: Optional[str] = None   # e.g. PLAYS_FOR inverse is HAS_PLAYER
    description: str = ""

    def __repr__(self):
        return f"RelationType({self.domain} ──{self.name}──► {self.range})"


RELATION_TYPES = {
    # Fan ↔ Team
    "FOLLOWS":        RelationType("FOLLOWS",       "Fan",    "Team",
                                   description="fan supports / roots for a team"),
    "ATTENDED":       RelationType("ATTENDED",      "Fan",    "Event"),

    # Player ↔ Team
    "PLAYS_FOR":      RelationType("PLAYS_FOR",     "Player", "Team",
                                   inverse="HAS_PLAYER"),
    "HAS_PLAYER":     RelationType("HAS_PLAYER",    "Team",   "Player"),

    # Coach ↔ Team
    "COACHES":        RelationType("COACHES",       "Coach",  "Team"),

    # Team ↔ Venue / City
    "HOME_COURT":     RelationType("HOME_COURT",    "Team",   "Venue"),
    "PLAYS_IN":       RelationType("PLAYS_IN",      "Team",   "City"),

    # Person ↔ City
    "LIVES_IN":       RelationType("LIVES_IN",      "Person", "City"),

    # Event ↔ Venue
    "HOSTED_AT":      RelationType("HOSTED_AT",     "Event",  "Venue"),

    # Player ↔ Player (rivalries, relationships)
    "TEAMMATE_OF":    RelationType("TEAMMATE_OF",   "Player", "Player"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. ONTOLOGY CLASS
#    Wraps both dicts and adds validation helpers.
# ─────────────────────────────────────────────────────────────────────────────

class Ontology:
    def __init__(self, entity_types: dict, relation_types: dict):
        self.entity_types = entity_types
        self.relation_types = relation_types

    def is_valid_relation(self, relation_name: str, source_type: str, target_type: str) -> bool:
        """
        Check whether (source_type) ──relation──► (target_type) is allowed.

        Also checks the IS-A hierarchy: a relation valid for 'Person' is also
        valid for 'Player' and 'Fan' because they inherit from Person.
        """
        rel = self.relation_types.get(relation_name)
        if rel is None:
            return False
        return (
            self._is_a(source_type, rel.domain) and
            self._is_a(target_type, rel.range)
        )

    def _is_a(self, child: str, ancestor: str) -> bool:
        """True if child is the same as ancestor or inherits from it."""
        if child == ancestor:
            return True
        et = self.entity_types.get(child)
        if et is None or et.parent is None:
            return False
        return self._is_a(et.parent, ancestor)

    def ancestors(self, entity_type: str) -> list[str]:
        """Return the full parent chain, including the type itself."""
        chain = [entity_type]
        et = self.entity_types.get(entity_type)
        while et and et.parent:
            chain.append(et.parent)
            et = self.entity_types.get(et.parent)
        return chain

    def relations_for_type(self, entity_type: str) -> list[RelationType]:
        """List all relations whose domain is compatible with entity_type."""
        return [
            r for r in self.relation_types.values()
            if self._is_a(entity_type, r.domain)
        ]

    def summary(self):
        print("=" * 60)
        print("NBA FAN DOMAIN ONTOLOGY")
        print("=" * 60)
        print(f"\n{len(self.entity_types)} entity types:\n")
        for et in self.entity_types.values():
            indent = "  " if et.parent else ""
            parent = f"  (is-a {et.parent})" if et.parent else ""
            print(f"  {indent}{et.name}{parent}")
            if et.description:
                print(f"      └─ {et.description}")

        print(f"\n{len(self.relation_types)} relation types:\n")
        for rt in self.relation_types.values():
            print(f"  {rt.domain} ──{rt.name}──► {rt.range}")


# Build the singleton
NBA_ONTOLOGY = Ontology(ENTITY_TYPES, RELATION_TYPES)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NBA_ONTOLOGY.summary()

    print("\n" + "=" * 60)
    print("VALIDATION EXAMPLES")
    print("=" * 60)

    checks = [
        # (relation, source_type, target_type, expected)
        ("FOLLOWS",    "Fan",    "Team",   True),   # valid
        ("FOLLOWS",    "Player", "Team",  False),   # Players don't follow teams (in this ontology)
        ("PLAYS_FOR",  "Player", "Team",   True),   # valid
        ("PLAYS_FOR",  "Fan",    "Team",  False),   # Fans don't play for teams
        ("LIVES_IN",   "Player", "City",  True),    # Player is-a Person, Person LIVES_IN City → valid
        ("LIVES_IN",   "Fan",    "City",  True),    # Fan is-a Person → also valid
        ("HOME_COURT", "Team",   "Venue", True),    # valid
        ("HOME_COURT", "Fan",    "Venue", False),   # Fan can't have a home court
        ("ATTENDED",   "Fan",    "Event", True),    # valid
    ]

    for rel, src, tgt, expected in checks:
        result = NBA_ONTOLOGY.is_valid_relation(rel, src, tgt)
        status = "OK" if result == expected else "UNEXPECTED"
        mark   = "✓" if result else "✗"
        print(f"  [{mark}] {src} ──{rel}──► {tgt}   ({status})")

    print("\n" + "=" * 60)
    print("ANCESTRY CHAIN")
    print("=" * 60)
    for t in ["Player", "Fan", "Coach", "Team"]:
        chain = NBA_ONTOLOGY.ancestors(t)
        print(f"  {t}: {' → '.join(chain)}")

    print("\n" + "=" * 60)
    print("ALLOWED RELATIONS FOR Fan")
    print("=" * 60)
    for r in NBA_ONTOLOGY.relations_for_type("Fan"):
        print(f"  Fan ──{r.name}──► {r.range}")
