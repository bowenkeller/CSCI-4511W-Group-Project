from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable
import heapq
import random

from graph import Edge, Graph, Player


@dataclass
class SearchResult:
    """A normalized result object returned by all search algorithms."""

    algorithm: str
    path: list[str]
    cost: float
    nodes_expanded: int
    visited_order: list[str]
    found: bool = False
    stop_reason: str = ""

    def __str__(self) -> str:
        status = "Found goal" if self.found else "No complete path found"
        path = " -> ".join(self.path) if self.path else "[none]"
        reason = f"\nStop reason:    {self.stop_reason}" if self.stop_reason else ""
        return (
            f"{self.algorithm}: {status}\n"
            f"Path:           {path}\n"
            f"Traversal time: {self.cost:.4f}\n"
            f"Nodes expanded: {self.nodes_expanded}"
            f"{reason}"
        )


Heuristic = Callable[[str, str, frozenset[str] | None], float]
State = tuple[str, frozenset[str]]


WARP_NODES = {
    "Warp Fire",
    "Warp Water",
    "Warp Forest",
    "Warp Shadow",
    "Warp Spirit",
    "Warp Light",
}


def zero_heuristic(node: str, goal: str, inventory: frozenset[str] | None = None) -> float:
    """Admissible default heuristic: no estimated remaining cost."""
    return 0.0


def make_reverse_cost_heuristic(graph: Graph) -> Heuristic:
    """
    Estimate distance to each possible goal by running Dijkstra backward while
    ignoring item requirements. This is useful for A* and GBFS because it points
    the search toward nodes physically closer to the goal without overestimating
    travel time in the relaxed graph.
    """
    reverse_edges: dict[str, list[tuple[str, float]]] = {node: [] for node in graph.nodes}
    for source in graph.nodes:
        for edge in graph.node_data[source].edges:
            reverse_edges.setdefault(edge.target, []).append((source, edge.cost))

    cache: dict[str, dict[str, float]] = {}

    def distances_to(goal: str) -> dict[str, float]:
        if goal in cache:
            return cache[goal]

        distances = {goal: 0.0}
        heap: list[tuple[float, str]] = [(0.0, goal)]

        while heap:
            distance, node = heapq.heappop(heap)
            if distance > distances.get(node, float("inf")):
                continue
            for previous, edge_cost in reverse_edges.get(node, []):
                new_distance = distance + edge_cost
                if new_distance < distances.get(previous, float("inf")):
                    distances[previous] = new_distance
                    heapq.heappush(heap, (new_distance, previous))

        cache[goal] = distances
        return distances

    def heuristic(node: str, goal: str, inventory: frozenset[str] | None = None) -> float:
        return distances_to(goal).get(node, 0.0)

    return heuristic


def make_loading_zones_heuristic(graph: Graph) -> Heuristic:
    """
    Estimate remaining distance as the number of loading-zone transitions to
    the goal in a relaxed graph that ignores item requirements and edge costs.

    This lets you test a different idea of closeness than travel time: how many
    graph edges/load zones remain before the target. It is mainly useful for
    GBFS/A* experiments and may behave differently from reverse-cost.
    """
    reverse_edges: dict[str, list[str]] = {node: [] for node in graph.nodes}
    for source in graph.nodes:
        for edge in graph.node_data[source].edges:
            reverse_edges.setdefault(edge.target, []).append(source)

    cache: dict[str, dict[str, int]] = {}

    def distances_to(goal: str) -> dict[str, int]:
        if goal in cache:
            return cache[goal]

        distances = {goal: 0}
        queue: deque[str] = deque([goal])

        while queue:
            node = queue.popleft()
            for previous in reverse_edges.get(node, []):
                if previous not in distances:
                    distances[previous] = distances[node] + 1
                    queue.append(previous)

        cache[goal] = distances
        return distances

    def heuristic(node: str, goal: str, inventory: frozenset[str] | None = None) -> float:
        return float(distances_to(goal).get(node, 0))

    return heuristic


def make_nearest_item_heuristic(graph: Graph) -> Heuristic:
    """
    Estimate distance to the nearest currently uncollected item.

    Unlike reverse-cost and loading-zones, this heuristic is progression-driven
    rather than goal-distance-driven. It uses the current inventory to prefer
    states that can reach a new item soon. This is experimental and should not
    be treated as an optimality-preserving A* heuristic for the final goal.
    """
    item_nodes = [node for node in graph.nodes if graph.node_data[node].receives]
    all_items = set().union(*(graph.node_data[node].receives for node in graph.nodes))
    cache: dict[tuple[str, frozenset[str]], float] = {}

    def shortest_relaxed_distance_to_needed_item(node: str, inventory: frozenset[str]) -> float:
        key = (node, inventory)
        if key in cache:
            return cache[key]

        if graph.node_data[node].receives - set(inventory):
            cache[key] = 0.0
            return 0.0

        # If every item has already been collected, this heuristic has no item
        # target left and falls back to zero.
        if all_items <= set(inventory):
            cache[key] = 0.0
            return 0.0

        distances = {node: 0.0}
        heap: list[tuple[float, str]] = [(0.0, node)]

        while heap:
            distance, current = heapq.heappop(heap)
            if distance > distances.get(current, float("inf")):
                continue
            if graph.node_data[current].receives - set(inventory):
                cache[key] = distance
                return distance
            for edge in graph.node_data[current].edges:
                # This is intentionally relaxed: ignore item requirements so the
                # heuristic stays cheap and acts as a rough attraction to items.
                new_distance = distance + edge.cost
                if new_distance < distances.get(edge.target, float("inf")):
                    distances[edge.target] = new_distance
                    heapq.heappush(heap, (new_distance, edge.target))

        cache[key] = 0.0
        return 0.0

    def heuristic(node: str, goal: str, inventory: frozenset[str] | None = None) -> float:
        if node == goal:
            return 0.0
        return shortest_relaxed_distance_to_needed_item(node, inventory or frozenset())

    return heuristic


def _inventory_after_collecting(graph: Graph, node: str, inventory: frozenset[str]) -> frozenset[str]:
    return inventory | frozenset(graph.node_data[node].receives)


def _valid_edges_for_state(graph: Graph, node: str, inventory: frozenset[str]) -> list[Edge]:
    """Collect the current node's item, then return all legal outgoing moves."""
    new_inventory = _inventory_after_collecting(graph, node, inventory)
    edges = graph.valid_edges(node, new_inventory)

    # Warp nodes are modeled as special nodes. Once their needs are satisfied,
    # their outgoing edges become available from the current state.
    for warp in WARP_NODES:
        if warp not in graph.node_data:
            continue
        warp_node = graph.node_data[warp]
        if warp_node.needs <= set(new_inventory):
            for edge in warp_node.edges:
                target = graph.node_data[edge.target]
                if edge.needs <= set(new_inventory) and target.needs <= set(new_inventory):
                    edges.append(edge)

    return sorted(edges, key=lambda edge: edge.cost)


def _register_non_dominated_state(
    states_by_node: dict[str, list[tuple[frozenset[str], float]]],
    node: str,
    inventory: frozenset[str],
    cost: float,
) -> bool:
    """
    Return True when this state is worth exploring.

    A state is dominated when the same node has already been reached with a
    superset of inventory at an equal or lower path cost.
    """
    current = states_by_node.setdefault(node, [])

    for existing_inventory, existing_cost in current:
        if existing_inventory.issuperset(inventory) and existing_cost <= cost:
            return False

    states_by_node[node] = [
        (existing_inventory, existing_cost)
        for existing_inventory, existing_cost in current
        if not (inventory.issuperset(existing_inventory) and cost <= existing_cost)
    ]
    states_by_node[node].append((inventory, cost))
    return True


def _is_active_state(
    states_by_node: dict[str, list[tuple[frozenset[str], float]]],
    node: str,
    inventory: frozenset[str],
    cost: float,
) -> bool:
    return any(
        existing_inventory == inventory and existing_cost == cost
        for existing_inventory, existing_cost in states_by_node.get(node, [])
    )


def _finish_player(player: Player | None, path: Iterable[str], inventory: frozenset[str]) -> None:
    if player is not None:
        player.inventory = set(inventory)
        player.visited = set(path)


def bfs(
    graph: Graph,
    start: str,
    goal: str,
    player: Player | None = None,
    max_expanded: int | None = None,
) -> SearchResult:
    """Breadth-first search. Finds fewest transitions, not lowest time cost."""
    start_inventory = frozenset(player.inventory) if player else frozenset()
    frontier: deque[tuple[str, list[str], float, frozenset[str]]] = deque(
        [(start, [start], 0.0, start_inventory)]
    )
    states_by_node: dict[str, list[tuple[frozenset[str], float]]] = {}
    _register_non_dominated_state(states_by_node, start, start_inventory, 0.0)

    visit_order: list[str] = []
    best_path, best_cost, best_inventory = [start], 0.0, start_inventory

    while frontier:
        node, path, cost, inventory = frontier.popleft()
        if not _is_active_state(states_by_node, node, inventory, cost):
            continue

        visit_order.append(node)
        inventory_here = _inventory_after_collecting(graph, node, inventory)

        if len(inventory_here) > len(best_inventory) or len(path) > len(best_path):
            best_path, best_cost, best_inventory = path, cost, inventory_here

        if node == goal:
            _finish_player(player, path, inventory_here)
            return SearchResult("BFS", path, cost, len(visit_order), visit_order, found=True)

        if max_expanded is not None and len(visit_order) >= max_expanded:
            _finish_player(player, best_path, best_inventory)
            return SearchResult(
                "BFS",
                best_path,
                best_cost,
                len(visit_order),
                visit_order,
                found=False,
                stop_reason=f"max expanded states reached ({max_expanded})",
            )

        for edge in _valid_edges_for_state(graph, node, inventory):
            next_cost = cost + edge.cost
            if _register_non_dominated_state(states_by_node, edge.target, inventory_here, next_cost):
                frontier.append((edge.target, path + [edge.target], next_cost, inventory_here))

    _finish_player(player, best_path, best_inventory)
    return SearchResult("BFS", best_path, best_cost, len(visit_order), visit_order, found=False)


def greedy_best_first(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    player: Player | None = None,
    max_expanded: int | None = None,
) -> SearchResult:
    """Greedy best-first search. Expands the state with the smallest h(n)."""
    start_inventory = frozenset(player.inventory) if player else frozenset()
    counter = 0
    frontier: list[tuple[float, int, str, list[str], float, frozenset[str]]] = [
        (heuristic(start, goal, start_inventory), counter, start, [start], 0.0, start_inventory)
    ]
    states_by_node: dict[str, list[tuple[frozenset[str], float]]] = {}
    _register_non_dominated_state(states_by_node, start, start_inventory, 0.0)

    visit_order: list[str] = []
    best_path, best_cost, best_inventory = [start], 0.0, start_inventory

    while frontier:
        _, _, node, path, cost, inventory = heapq.heappop(frontier)
        if not _is_active_state(states_by_node, node, inventory, cost):
            continue

        visit_order.append(node)
        inventory_here = _inventory_after_collecting(graph, node, inventory)

        if len(inventory_here) > len(best_inventory) or len(path) > len(best_path):
            best_path, best_cost, best_inventory = path, cost, inventory_here

        if node == goal:
            _finish_player(player, path, inventory_here)
            return SearchResult("Greedy Best-First", path, cost, len(visit_order), visit_order, found=True)

        if max_expanded is not None and len(visit_order) >= max_expanded:
            _finish_player(player, best_path, best_inventory)
            return SearchResult(
                "Greedy Best-First",
                best_path,
                best_cost,
                len(visit_order),
                visit_order,
                found=False,
                stop_reason=f"max expanded states reached ({max_expanded})",
            )

        for edge in _valid_edges_for_state(graph, node, inventory):
            next_cost = cost + edge.cost
            if _register_non_dominated_state(states_by_node, edge.target, inventory_here, next_cost):
                counter += 1
                heapq.heappush(
                    frontier,
                    (heuristic(edge.target, goal, inventory_here), counter, edge.target, path + [edge.target], next_cost, inventory_here),
                )

    _finish_player(player, best_path, best_inventory)
    return SearchResult("Greedy Best-First", best_path, best_cost, len(visit_order), visit_order, found=False)


def a_star(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    player: Player | None = None,
    max_expanded: int | None = None,
) -> SearchResult:
    """A* search. Expands the state with the smallest f(n) = g(n) + h(n)."""
    start_inventory = frozenset(player.inventory) if player else frozenset()
    counter = 0
    frontier: list[tuple[float, int, str, list[str], float, frozenset[str]]] = [
        (heuristic(start, goal, start_inventory), counter, start, [start], 0.0, start_inventory)
    ]
    states_by_node: dict[str, list[tuple[frozenset[str], float]]] = {}
    _register_non_dominated_state(states_by_node, start, start_inventory, 0.0)
    best_g: dict[State, float] = {(start, start_inventory): 0.0}

    visit_order: list[str] = []
    best_path, best_cost, best_inventory = [start], 0.0, start_inventory

    while frontier:
        _, _, node, path, cost, inventory = heapq.heappop(frontier)
        if cost > best_g.get((node, inventory), float("inf")):
            continue
        if not _is_active_state(states_by_node, node, inventory, cost):
            continue

        visit_order.append(node)
        inventory_here = _inventory_after_collecting(graph, node, inventory)

        if len(inventory_here) > len(best_inventory) or len(path) > len(best_path):
            best_path, best_cost, best_inventory = path, cost, inventory_here

        if node == goal:
            _finish_player(player, path, inventory_here)
            return SearchResult("A*", path, cost, len(visit_order), visit_order, found=True)

        if max_expanded is not None and len(visit_order) >= max_expanded:
            _finish_player(player, best_path, best_inventory)
            return SearchResult(
                "A*",
                best_path,
                best_cost,
                len(visit_order),
                visit_order,
                found=False,
                stop_reason=f"max expanded states reached ({max_expanded})",
            )

        for edge in _valid_edges_for_state(graph, node, inventory):
            next_cost = cost + edge.cost
            next_state = (edge.target, inventory_here)
            if next_cost >= best_g.get(next_state, float("inf")):
                continue
            if not _register_non_dominated_state(states_by_node, edge.target, inventory_here, next_cost):
                continue

            best_g[next_state] = next_cost
            counter += 1
            heapq.heappush(
                frontier,
                (
                    next_cost + heuristic(edge.target, goal, inventory_here),
                    counter,
                    edge.target,
                    path + [edge.target],
                    next_cost,
                    inventory_here,
                ),
            )

    _finish_player(player, best_path, best_inventory)
    return SearchResult("A*", best_path, best_cost, len(visit_order), visit_order, found=False)


def genetic_algorithm(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    player: Player | None = None,
    population_size: int = 40,
    generations: int = 50,
    max_steps: int = 150,
    mutation_rate: float = 0.08,
    elite_count: int = 5,
    seed: int | None = None,
) -> SearchResult:
    """
    Basic genetic algorithm experiment for the Zelda graph.

    Each individual is a fixed-length list of random keys in the range [0, 1).
    During evaluation, each key chooses one legal edge from the current state.
    The simulation still respects inventory, Needs, Receives, edge costs, and
    warp behavior through _valid_edges_for_state().

    GA is stochastic and does not guarantee the lowest-cost path. It is included
    as an experimental baseline/metaheuristic, not as an optimal search method.
    """
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")

    rng = random.Random(seed)
    start_inventory = frozenset(player.inventory) if player else frozenset()
    all_progress_items = set().union(*(node.receives for node in graph.node_data.values()))

    def make_individual() -> list[float]:
        return [rng.random() for _ in range(max_steps)]

    def ordered_edges(node: str, inventory: frozenset[str]) -> list[Edge]:
        # Prefer cheaper edges, edges closer to the goal, and edges that collect new items.
        inventory_here = _inventory_after_collecting(graph, node, inventory)
        edges = _valid_edges_for_state(graph, node, inventory)
        return sorted(
            edges,
            key=lambda edge: (
                heuristic(edge.target, goal, inventory_here),
                edge.cost,
                -len(graph.node_data[edge.target].receives - set(inventory_here)),
                edge.target,
            ),
        )

    def evaluate(individual: list[float]) -> tuple[float, SearchResult, frozenset[str]]:
        node = start
        inventory = start_inventory
        path = [start]
        visited_order: list[str] = []
        cost = 0.0

        for gene in individual:
            visited_order.append(node)
            inventory_here = _inventory_after_collecting(graph, node, inventory)

            if node == goal:
                result = SearchResult("Genetic Algorithm", path, cost, len(visited_order), visited_order, found=True)
                return cost, result, inventory_here

            edges = ordered_edges(node, inventory)
            if not edges:
                break

            # Genes near 0 choose the best-ranked legal edge. Larger genes explore alternatives.
            index = min(int(gene * len(edges)), len(edges) - 1)
            edge = edges[index]
            node = edge.target
            inventory = inventory_here
            cost += edge.cost
            path.append(node)

        inventory = _inventory_after_collecting(graph, node, inventory)
        if node == goal:
            result = SearchResult("Genetic Algorithm", path, cost, len(visited_order), visited_order, found=True)
            return cost, result, inventory

        # Penalize incomplete routes. The penalty rewards being physically closer to the goal
        # and having collected more progression items, but any completed route beats incomplete ones.
        remaining_estimate = heuristic(node, goal, inventory)
        collected = len(set(inventory) & all_progress_items)
        fitness = 1_000_000.0 + cost + (10.0 * remaining_estimate) - (100.0 * collected)
        result = SearchResult("Genetic Algorithm", path, cost, len(visited_order), visited_order, found=False)
        return fitness, result, inventory

    def crossover(parent_a: list[float], parent_b: list[float]) -> list[float]:
        if max_steps == 1:
            child = parent_a[:]
        else:
            cut = rng.randint(1, max_steps - 1)
            child = parent_a[:cut] + parent_b[cut:]
        for i in range(max_steps):
            if rng.random() < mutation_rate:
                # Mostly small local mutation, occasionally a complete random reset.
                if rng.random() < 0.8:
                    child[i] = min(0.999999, max(0.0, child[i] + rng.uniform(-0.20, 0.20)))
                else:
                    child[i] = rng.random()
        return child

    elite_count = max(1, min(elite_count, population_size - 1))
    population = [make_individual() for _ in range(population_size)]
    best_result: SearchResult | None = None
    best_inventory = start_inventory
    best_fitness = float("inf")
    nodes_expanded = 0

    for _ in range(generations):
        scored: list[tuple[float, list[float], SearchResult, frozenset[str]]] = []
        for individual in population:
            fitness, result, inventory = evaluate(individual)
            nodes_expanded += result.nodes_expanded
            scored.append((fitness, individual, result, inventory))
            if fitness < best_fitness:
                best_fitness = fitness
                best_result = result
                best_inventory = inventory

        scored.sort(key=lambda item: item[0])
        elites = [individual for _, individual, _, _ in scored[:elite_count]]

        # Tournament selection gives better individuals a higher chance without removing randomness.
        def select_parent() -> list[float]:
            tournament = rng.sample(scored, k=min(4, len(scored)))
            tournament.sort(key=lambda item: item[0])
            return tournament[0][1]

        next_population = [elite[:] for elite in elites]
        while len(next_population) < population_size:
            next_population.append(crossover(select_parent(), select_parent()))
        population = next_population

    assert best_result is not None
    best_result.nodes_expanded = nodes_expanded
    _finish_player(player, best_result.path, best_inventory)
    return best_result

