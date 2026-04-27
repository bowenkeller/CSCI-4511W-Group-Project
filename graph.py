
from dataclasses import dataclass, field
import csv


def _parse_csv_list(value: str) -> list[str]:
    """Parses comma-separated cells like 'A, B, C' into ['A', 'B', 'C']."""
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "one way"


@dataclass
class Edge:
    target: str
    cost: float = 1.0
    needs: set[str] = field(default_factory=set)


@dataclass
class Node:
    title: str
    needs: set[str] = field(default_factory=set)
    receives: set[str] = field(default_factory=set)
    edges: list[Edge] = field(default_factory=list)


@dataclass
class Player:
    inventory: set[str] = field(default_factory=set)
    visited: set[str] = field(default_factory=set)


class Graph:
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj: dict[str, list[tuple[str, float]]] = {}
        self.node_data: dict[str, Node] = {}
        self.node_set: set[str] = set()

    def add_node(self, node: str | Node) -> None:
        if isinstance(node, Node):
            title = node.title
            if title not in self.node_data:
                self.node_data[title] = node
            else:
                # Merge metadata if node already exists.
                self.node_data[title].needs.update(node.needs)
                self.node_data[title].receives.update(node.receives)
        else:
            title = node
            if title not in self.node_data:
                self.node_data[title] = Node(title=title)

        if title not in self.adj:
            self.adj[title] = []

        self.node_set.add(title)

    def has_node(self, title: str) -> bool:
        return title in self.node_set

    def add_edge(
        self,
        u: str,
        v: str,
        cost: float = 1.0,
        needs: list[str] | None = None,
        one_way: bool = False,
    ) -> None:
        edge_needs = set(needs or [])

        self.add_node(u)
        self.add_node(v)

        self.adj[u].append((v, cost))
        self.node_data[u].edges.append(Edge(target=v, cost=cost, needs=edge_needs))

        if not one_way and not self.directed:
            self.adj[v].append((u, cost))
            self.node_data[v].edges.append(Edge(target=u, cost=cost, needs=edge_needs))

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        return self.adj.get(node, [])

    def get_next(self, cur: Node, player: Player | None = None, heuristic=None) -> list[Edge]:
        edges = self.node_data[cur.title].edges

        if player is not None:
            edges = [
                e for e in edges
                if self.node_data[e.target].needs <= player.inventory
            ]

        if heuristic is not None:
            return heuristic(edges)
        return sorted(edges, key=lambda e: e.cost)

    @property
    def nodes(self) -> list[str]:
        return list(self.node_set)


def load_graph_from_csv(filename: str, directed: bool = False) -> Graph:
    """
    Expects columns:
    Nodes, Needs, Receives, Edges, Time, One Way (BOOL)
    """
    graph = Graph(directed=directed)

    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            title = row.get("Nodes", "").strip()
            if not title:
                edge_text = row.get("Edges", "").strip()
                if not edge_text:
                    continue

                endpoints = [part.strip() for part in edge_text.split("//")]
                if len(endpoints) != 2 or not endpoints[0] or not endpoints[1]:
                    continue

                u, v = endpoints
                if not graph.has_node(u):
                    continue

                cost_raw = row.get("Time", "")
                cost = float(cost_raw) if str(cost_raw).strip() else 1.0
                one_way = _parse_bool(row.get("One Way", "false"))

                graph.add_edge(
                    u,
                    v,
                    cost=cost,
                    needs=_parse_csv_list(row.get("Needs", "")),
                    one_way=one_way,
                )
                continue

            node = Node(
                title=title,
                needs=set(_parse_csv_list(row.get("Needs", ""))),
                receives=set(_parse_csv_list(row.get("Receives", ""))),
            )
            graph.add_node(node)

            edge_text = row.get("Edges", "").strip()
            if not edge_text:
                continue

            endpoints = [part.strip() for part in edge_text.split("//")]
            if len(endpoints) != 2 or not endpoints[0] or not endpoints[1]:
                continue

            u, v = endpoints
            cost_raw = row.get("Time", "")
            cost = float(cost_raw) if str(cost_raw).strip() else 1.0
            one_way = _parse_bool(row.get("One Way", "false"))

            graph.add_edge(
                u,
                v,
                cost=cost,
                needs=_parse_csv_list(row.get("Needs", "")),
                one_way=one_way,
            )

    return graph