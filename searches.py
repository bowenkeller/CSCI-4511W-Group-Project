from collections import deque
from dataclasses import dataclass
from typing import Callable
import heapq
from graph import *

 
 
 
@dataclass
class SearchResult:
    """
    Stores the result of a search

    path: sequence of nodes taken to solution state
    cost: summation of weight of the path taken
    nodes_expanded: # of nodes expanded
    visited_order: order in which nodes were expanded
    """
    path: list[str]          
    cost: float              
    nodes_expanded: int      
    visited_order: list[str] 
    found: bool = False
 
    def __str__(self) -> str:
        if not self.found:
            if not self.path:
                return (
                    "No path found.\n"
                    f"Nodes expanded: {self.nodes_expanded}\n"
                    f"Visit order:    {' -> '.join(self.visited_order)}"
                )
            return (
                "No complete path found. Best partial progress:\n"
                f"Path:           {' -> '.join(self.path)}\n"
                f"Cost:           {self.cost:.4f}\n"
                f"Nodes expanded: {self.nodes_expanded}\n"
                f"Visit order:    {' -> '.join(self.visited_order)}"
            )
        return (
            f"Path:           {' -> '.join(self.path)}\n"
            f"Cost:           {self.cost:.4f}\n"
            f"Nodes expanded: {self.nodes_expanded}\n"
            f"Visit order:    {' -> '.join(self.visited_order)}"
        )

 

Heuristic = Callable[[str, str], float]


def _get_edges_for_state(
    graph: Graph,
    node: str,
    path: list[str],
    inventory: frozenset,
    use_player_rules: bool,
) -> tuple[list[Edge], frozenset]:
    """Returns valid outgoing edges and inventory after collecting this node's receives."""
    new_inv = inventory | graph.node_data[node].receives
    cur_node = graph.node_data[node]

    if use_player_rules:
        tmp = Player(inventory=set(new_inv), visited=set(path))
        edges = graph.get_next(cur_node, player=tmp)
    else:
        edges = graph.get_next(cur_node)

    warp_nodes = [
        "Warp Fire",
        "Warp Water",
        "Warp Forest",
        "Warp Shadow",
        "Warp Spirit",
        "Warp Light",
    ]

    for warp in warp_nodes:
        if warp in graph.node_data:
            warp_node = graph.node_data[warp]

            if warp_node.needs <= set(new_inv):
                for e in warp_node.edges:
                    edges.append(e)

    return edges, new_inv


def _register_non_dominated_state(
    state_frontier: dict[str, list[tuple[frozenset, float]]],
    node: str,
    inventory: frozenset,
    cost: float,
) -> bool:
    """
    Keeps only non-dominated (inventory, cost) states per node.

    A state is dominated if an existing state at the same node has a superset
    inventory and a cost <= this state's cost.
    """
    current = state_frontier.setdefault(node, [])

    for inv_existing, cost_existing in current:
        if inv_existing.issuperset(inventory) and cost_existing <= cost:
            return False

    state_frontier[node] = [
        (inv_existing, cost_existing)
        for inv_existing, cost_existing in current
        if not (inventory.issuperset(inv_existing) and cost <= cost_existing)
    ]
    state_frontier[node].append((inventory, cost))
    return True


def _is_active_state(
    state_frontier: dict[str, list[tuple[frozenset, float]]],
    node: str,
    inventory: frozenset,
    cost: float,
) -> bool:
    return any(
        inv_existing == inventory and cost_existing == cost
        for inv_existing, cost_existing in state_frontier.get(node, [])
    )



def bfs(
    graph: Graph,
    start: str,
    goal: str,
    player: Player | None = None,
) -> SearchResult:
    """
    Standard BFS implementation.
    If a player is provided, collects receives on each visit and filters
    neighbors whose target node requires items not yet in inventory.
    """
    start_inv = frozenset(player.inventory) if player else frozenset()

    if start == goal:
        return SearchResult([start], 0.0, 1, [start], found=True)

    # frontier: (node_title, path, cost, inventory)
    frontier: deque[tuple[str, list[str], float, frozenset]] = deque(
        [(start, [start], 0.0, start_inv)]
    )
    state_frontier: dict[str, list[tuple[frozenset, float]]] = {}
    _register_non_dominated_state(state_frontier, start, start_inv, 0.0)
    visit_order: list[str] = []
    nodes_expanded = 0
    best_path: list[str] = [start]
    best_cost = 0.0
    best_inv = start_inv

    while frontier:
        node, path, cost, inv = frontier.popleft()

        if not _is_active_state(state_frontier, node, inv, cost):
            continue

        visit_order.append(node)
        nodes_expanded += 1

        if len(path) > len(best_path) or (len(path) == len(best_path) and cost < best_cost):
            best_path = path
            best_cost = cost
            best_inv = inv

        if node == goal:
            if player is not None:
                player.inventory = set(inv)
                player.visited = set(path)
            return SearchResult(path, cost, nodes_expanded, visit_order, found=True)

        edges, new_inv = _get_edges_for_state(
            graph,
            node,
            path,
            inv,
            use_player_rules=player is not None,
        )

        for edge in edges:
            next_cost = cost + edge.cost
            if _register_non_dominated_state(state_frontier, edge.target, new_inv, next_cost):
                frontier.append((edge.target, path + [edge.target], cost + edge.cost, new_inv))

    if player is not None:
        player.inventory = set(best_inv)
        player.visited = set(best_path)
    return SearchResult(best_path, best_cost, nodes_expanded, visit_order, found=False)
 

 
def greedy_best_first(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    player: Player | None = None,
) -> SearchResult:
    if start == goal:
        return SearchResult([start], 0.0, 1, [start], found=True)

    start_inv = frozenset(player.inventory) if player else frozenset()

    counter = 0
    # frontier: (h, counter, node_title, path, cost, inventory)
    frontier: list[tuple[float, int, str, list[str], float, frozenset]] = [
        (heuristic(start, goal), counter, start, [start], 0.0, start_inv)
    ]
    state_frontier: dict[str, list[tuple[frozenset, float]]] = {}
    _register_non_dominated_state(state_frontier, start, start_inv, 0.0)
    visit_order: list[str] = []
    nodes_expanded = 0
    best_path: list[str] = [start]
    best_cost = 0.0
    best_inv = start_inv

    while frontier:
        h, _, node, path, cost, inv = heapq.heappop(frontier)

        if not _is_active_state(state_frontier, node, inv, cost):
            continue

        visit_order.append(node)
        nodes_expanded += 1

        if len(path) > len(best_path) or (len(path) == len(best_path) and cost < best_cost):
            best_path = path
            best_cost = cost
            best_inv = inv

        if node == goal:
            if player is not None:
                player.inventory = set(inv)
                player.visited = set(path)
            return SearchResult(path, cost, nodes_expanded, visit_order, found=True)

        edges, new_inv = _get_edges_for_state(
            graph,
            node,
            path,
            inv,
            use_player_rules=player is not None,
        )

        for edge in edges:
            next_cost = cost + edge.cost
            if _register_non_dominated_state(state_frontier, edge.target, new_inv, next_cost):
                counter += 1
                heapq.heappush(
                    frontier,
                    (heuristic(edge.target, goal), counter, edge.target, path + [edge.target], next_cost, new_inv)
                )

    if player is not None:
        player.inventory = set(best_inv)
        player.visited = set(best_path)
    return SearchResult(best_path, best_cost, nodes_expanded, visit_order, found=False)


 
def a_star(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    player: Player | None = None,
) -> SearchResult:
    if start == goal:
        return SearchResult([start], 0.0, 1, [start], found=True)

    start_inv = frozenset(player.inventory) if player else frozenset()

    counter = 0
    # frontier: (f, counter, node_title, path, g_cost, inventory)
    frontier: list[tuple[float, int, str, list[str], float, frozenset]] = [
        (heuristic(start, goal), counter, start, [start], 0.0, start_inv)
    ]

    # g_cost keyed on (node, inventory) since same node reached with diff inventory is a different state
    g_cost: dict[tuple[str, frozenset], float] = {(start, start_inv): 0.0}
    state_frontier: dict[str, list[tuple[frozenset, float]]] = {}
    _register_non_dominated_state(state_frontier, start, start_inv, 0.0)
    visit_order: list[str] = []
    nodes_expanded = 0
    best_path: list[str] = [start]
    best_cost = 0.0
    best_inv = start_inv

    while frontier:
        f, _, node, path, g, inv = heapq.heappop(frontier)

        if g > g_cost.get((node, inv), float('inf')):
            continue
        if not _is_active_state(state_frontier, node, inv, g):
            continue

        visit_order.append(node)
        nodes_expanded += 1

        if len(path) > len(best_path) or (len(path) == len(best_path) and g < best_cost):
            best_path = path
            best_cost = g
            best_inv = inv

        if node == goal:
            if player is not None:
                player.inventory = set(inv)
                player.visited = set(path)
            return SearchResult(path, g, nodes_expanded, visit_order, found=True)

        edges, new_inv = _get_edges_for_state(
            graph,
            node,
            path,
            inv,
            use_player_rules=player is not None,
        )

        for edge in edges:
            new_g = g + edge.cost
            state = (edge.target, new_inv)

            if new_g < g_cost.get(state, float('inf')) and _register_non_dominated_state(
                state_frontier,
                edge.target,
                new_inv,
                new_g,
            ):
                g_cost[state] = new_g
                counter += 1
                f_new = new_g + heuristic(edge.target, goal)
                heapq.heappush(
                    frontier,
                    (f_new, counter, edge.target, path + [edge.target], new_g, new_inv)
                )

    if player is not None:
        player.inventory = set(best_inv)
        player.visited = set(best_path)
    return SearchResult(best_path, best_cost, nodes_expanded, visit_order, found=False)