"""
knowledge_graph.py
──────────────────
Builds an NBA fan knowledge graph and runs several traversal queries.

Concepts demonstrated:
  Entity types     → nodes with a 'type' attribute (Fan, Player, Team, ...)
  Relationships    → directed, labelled edges (FOLLOWS, PLAYS_FOR, ...)
  Ontology schema  → the set of allowed types and relations (see ontology_schema.py)
  Graph traversal  → following edges to answer questions that would be awkward as SQL JOINs
  Multi-hop query  → chaining 3+ edge traversals to answer "who goes where?"
  Visualization    → saves knowledge_graph.png

The graph is a MultiDiGraph so we can have more than one edge type between the
same pair of nodes (e.g. a player could be both a FAN-favourite and a TEAMMATE).
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# 1. NODE DATA
#    Each node has at minimum: type (from the ontology) and name (display label).
#    Player nodes also carry position and jersey_number.
# ─────────────────────────────────────────────────────────────────────────────

fans = {
    "marcus": {"type": "Fan", "name": "Marcus"},
    "aisha":  {"type": "Fan", "name": "Aisha"},
    "jordan": {"type": "Fan", "name": "Jordan"},
    "wei":    {"type": "Fan", "name": "Wei"},
    "priya":  {"type": "Fan", "name": "Priya"},
}

players = {
    "lebron":  {"type": "Player", "name": "LeBron James",  "position": "Forward", "number": 23},
    "ad":      {"type": "Player", "name": "Anthony Davis",  "position": "Center",  "number": 3},
    "tatum":   {"type": "Player", "name": "Jayson Tatum",   "position": "Forward", "number": 0},
    "curry":   {"type": "Player", "name": "Stephen Curry",  "position": "Guard",   "number": 30},
    "giannis": {"type": "Player", "name": "Giannis Antetokounmpo", "position": "Forward", "number": 34},
}

teams = {
    "lakers":   {"type": "Team", "name": "Los Angeles Lakers"},
    "celtics":  {"type": "Team", "name": "Boston Celtics"},
    "warriors": {"type": "Team", "name": "Golden State Warriors"},
    "bucks":    {"type": "Team", "name": "Milwaukee Bucks"},
}

cities = {
    "la":        {"type": "City", "name": "Los Angeles"},
    "boston":    {"type": "City", "name": "Boston"},
    "sf":        {"type": "City", "name": "San Francisco"},
    "milwaukee": {"type": "City", "name": "Milwaukee"},
}

venues = {
    "crypto":  {"type": "Venue", "name": "Crypto.com Arena"},
    "td":      {"type": "Venue", "name": "TD Garden"},
    "chase":   {"type": "Venue", "name": "Chase Center"},
    "fiserv":  {"type": "Venue", "name": "Fiserv Forum"},
}

events = {
    "finals_g2": {"type": "Event", "name": "NBA Finals Game 2, 2024"},
    "xmas_lal_bos": {"type": "Event", "name": "Lakers vs Celtics, Christmas 2024"},
    "allstar_2024": {"type": "Event", "name": "All-Star Game 2024"},
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

G = nx.MultiDiGraph()

# Add all nodes
for group in (fans, players, teams, cities, venues, events):
    for node_id, attrs in group.items():
        G.add_node(node_id, **attrs)

# Fans live in cities
G.add_edge("marcus", "la",        rel="LIVES_IN")
G.add_edge("aisha",  "boston",    rel="LIVES_IN")
G.add_edge("jordan", "sf",        rel="LIVES_IN")
G.add_edge("wei",    "la",        rel="LIVES_IN")
G.add_edge("priya",  "milwaukee", rel="LIVES_IN")

# Fans follow teams
G.add_edge("marcus", "lakers",   rel="FOLLOWS")
G.add_edge("aisha",  "celtics",  rel="FOLLOWS")
G.add_edge("jordan", "warriors", rel="FOLLOWS")
G.add_edge("wei",    "lakers",   rel="FOLLOWS")   # two LA fans follow the same team
G.add_edge("priya",  "bucks",    rel="FOLLOWS")

# Fans have a favourite player (a different edge type from FOLLOWS)
G.add_edge("marcus", "lebron", rel="FAVORITE_PLAYER")
G.add_edge("aisha",  "tatum",  rel="FAVORITE_PLAYER")
G.add_edge("jordan", "curry",  rel="FAVORITE_PLAYER")

# Players play for teams
G.add_edge("lebron",  "lakers",   rel="PLAYS_FOR")
G.add_edge("ad",      "lakers",   rel="PLAYS_FOR")
G.add_edge("tatum",   "celtics",  rel="PLAYS_FOR")
G.add_edge("curry",   "warriors", rel="PLAYS_FOR")
G.add_edge("giannis", "bucks",    rel="PLAYS_FOR")

# Teams play in cities and have a home court
G.add_edge("lakers",   "la",        rel="PLAYS_IN")
G.add_edge("celtics",  "boston",    rel="PLAYS_IN")
G.add_edge("warriors", "sf",        rel="PLAYS_IN")
G.add_edge("bucks",    "milwaukee", rel="PLAYS_IN")

G.add_edge("lakers",   "crypto", rel="HOME_COURT")
G.add_edge("celtics",  "td",     rel="HOME_COURT")
G.add_edge("warriors", "chase",  rel="HOME_COURT")
G.add_edge("bucks",    "fiserv", rel="HOME_COURT")

# Events hosted at venues
G.add_edge("finals_g2",     "chase",  rel="HOSTED_AT")
G.add_edge("xmas_lal_bos",  "crypto", rel="HOSTED_AT")
G.add_edge("allstar_2024",  "chase",  rel="HOSTED_AT")

# Fans attend events
G.add_edge("marcus", "xmas_lal_bos",  rel="ATTENDED")
G.add_edge("aisha",  "xmas_lal_bos",  rel="ATTENDED")
G.add_edge("jordan", "finals_g2",     rel="ATTENDED")
G.add_edge("jordan", "allstar_2024",  rel="ATTENDED")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAVERSAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def neighbors_by_rel(node: str, relation: str, direction="out") -> list[str]:
    """Return all nodes connected to `node` by `relation`."""
    if direction == "out":
        return [v for _, v, d in G.out_edges(node, data=True) if d.get("rel") == relation]
    else:
        return [u for u, _, d in G.in_edges(node, data=True) if d.get("rel") == relation]


def fans_at_venue(venue_id: str) -> list[dict]:
    """
    Multi-hop query: find all fans who attended an event hosted at this venue.

    Path: Venue <──HOSTED_AT── Event <──ATTENDED── Fan
    """
    results = []
    # events at this venue
    events_here = neighbors_by_rel(venue_id, "HOSTED_AT", direction="in")
    for event in events_here:
        # fans who attended that event
        attendees = neighbors_by_rel(event, "ATTENDED", direction="in")
        for fan in attendees:
            results.append({
                "fan":   G.nodes[fan]["name"],
                "event": G.nodes[event]["name"],
                "venue": G.nodes[venue_id]["name"],
            })
    return results


def fans_in_city_whose_team_plays_at(city_id: str, venue_id: str) -> list[dict]:
    """
    Multi-hop query:
      City <──LIVES_IN── Fan ──FOLLOWS──► Team ──HOME_COURT──► Venue
    """
    results = []
    fans_here = neighbors_by_rel(city_id, "LIVES_IN", direction="in")
    for fan in fans_here:
        for team in neighbors_by_rel(fan, "FOLLOWS"):
            if venue_id in neighbors_by_rel(team, "HOME_COURT"):
                results.append({
                    "fan":   G.nodes[fan]["name"],
                    "team":  G.nodes[team]["name"],
                    "venue": G.nodes[venue_id]["name"],
                })
    return results


def team_roster(team_id: str) -> list[str]:
    """All players on a team (PLAYS_FOR edges pointing to team_id)."""
    return neighbors_by_rel(team_id, "PLAYS_FOR", direction="in")


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN QUERIES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 58)
    print("QUERY 1: Who does Marcus follow and who is his fav player?")
    print("=" * 58)
    for team in neighbors_by_rel("marcus", "FOLLOWS"):
        print(f"  team  → {G.nodes[team]['name']}")
    for p in neighbors_by_rel("marcus", "FAVORITE_PLAYER"):
        print(f"  fav   → {G.nodes[p]['name']}")

    print("\n" + "=" * 58)
    print("QUERY 2: Lakers roster")
    print("=" * 58)
    for p in team_roster("lakers"):
        node = G.nodes[p]
        print(f"  #{node['number']:>2}  {node['name']} ({node['position']})")

    print("\n" + "=" * 58)
    print("QUERY 3: Fans in LA whose team plays at Crypto.com Arena")
    print("=" * 58)
    # This is 3 hops: LA → fans → teams → venue
    for r in fans_in_city_whose_team_plays_at("la", "crypto"):
        print(f"  {r['fan']} follows {r['team']} → {r['venue']}")

    print("\n" + "=" * 58)
    print("QUERY 4: Who attended events at Chase Center?")
    print("=" * 58)
    for r in fans_at_venue("chase"):
        print(f"  {r['fan']} attended {r['event']}")

    print("\n" + "=" * 58)
    print("QUERY 5: Shortest path  Marcus → Aisha")
    print("=" * 58)
    # undirected version for path finding
    UG = G.to_undirected()
    try:
        path = nx.shortest_path(UG, "marcus", "aisha")
        names = [G.nodes[n]["name"] for n in path]
        print("  " + " → ".join(names))
    except nx.NetworkXNoPath:
        print("  No path found")

    # ─────────────────────────────────────────────────────────────────────
    # 5. VISUALIZE
    # ─────────────────────────────────────────────────────────────────────
    print("\nGenerating knowledge_graph.png ...")

    color_map = {
        "Fan":    "#4C9AFF",   # blue
        "Player": "#FF8C42",   # orange
        "Team":   "#FF6B6B",   # red
        "City":   "#51CF66",   # green
        "Venue":  "#FFD43B",   # yellow
        "Event":  "#CC5DE8",   # purple
    }

    node_colors = [color_map[G.nodes[n]["type"]] for n in G.nodes()]
    labels = {n: G.nodes[n]["name"] for n in G.nodes()}

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(G, seed=7, k=1.6)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, alpha=0.92)
    nx.draw_networkx_labels(G, pos, labels, font_size=7.5, font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=14,
                           edge_color="#888", alpha=0.55, width=1.2)

    edge_labels = {(u, v): d["rel"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=5.5, alpha=0.8)

    # Legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in color_map.items()]
    plt.legend(handles=patches, loc="lower left", fontsize=8)

    plt.title("NBA Fan Knowledge Graph", fontsize=14, pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("knowledge_graph.png", dpi=150)
    print("Saved to knowledge_graph.png")
