from collections import deque
from dataclasses import dataclass
from typing import Callable
import heapq
 

 
class Graph:
    """
    Temporary implementation of a graph structure

    Nodes are stored as strings and edges floats representing weight

    Can support directed or undirected graphs
    """

    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj: dict[str, list[tuple[str, float]]] = {}
 
    def add_node(self, node: str) -> None:
        if node not in self.adj:
            self.adj[node] = []
 
    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))
 
    def neighbors(self, node: str) -> list[tuple[str, float]]:
        return self.adj.get(node, [])
 
    @property
    def nodes(self) -> list[str]:
        return list(self.adj.keys())
 
 
 
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
 
    def __str__(self) -> str:
        if not self.path:
            return "No path found."
        return (
            f"Path:           {' -> '.join(self.path)}\n"
            f"Cost:           {self.cost:.4f}\n"
            f"Nodes expanded: {self.nodes_expanded}\n"
            f"Visit order:    {' -> '.join(self.visited_order)}"
        )
 
 
_NO_RESULT = SearchResult(path=[], cost=float('inf'), nodes_expanded=0, visited_order=[])
 
 

Heuristic = Callable[[str, str], float]



def bfs(graph: Graph, start: str, goal: str) -> SearchResult:
    """
    Standard BFS implementation for the graph strucute defined above

    No heuristic considered while searching
    """
    if start == goal:
        return SearchResult([start], 0.0, 1, [start])

    frontier: deque[tuple[str, list[str], float]] = deque([(start, [start], 0.0)])
    visited: set[str] = {start}
    visit_order: list[str] = []
    nodes_expanded = 0

    while frontier:
        node, path, cost = frontier.popleft()
        visit_order.append(node)
        nodes_expanded += 1

        if node == goal:
            return SearchResult(path, cost, nodes_expanded, visit_order)

        for neighbor, weight in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, path + [neighbor], cost + weight))

    return _NO_RESULT
 

 
def greedy_best_first(graph: Graph, start: str, goal: str, heuristic: Heuristic) -> SearchResult:
    if start == goal:
        return SearchResult([start], 0.0, 1, [start])
 
    counter = 0
    frontier: list[tuple[float, int, str, list[str], float]] = [
        (heuristic(start, goal), counter, start, [start], 0.0)
    ]
    visited: set[str] = set()
    visit_order: list[str] = []
    nodes_expanded = 0
 
    while frontier:
        h, _, node, path, cost = heapq.heappop(frontier)
 
        if node in visited:
            continue
        visited.add(node)
        visit_order.append(node)
        nodes_expanded += 1
 
        if node == goal:
            return SearchResult(path, cost, nodes_expanded, visit_order)
 
        for neighbor, weight in graph.neighbors(node):
            if neighbor not in visited:
                counter += 1
                heapq.heappush(frontier, (heuristic(neighbor, goal), counter, neighbor, path + [neighbor], cost + weight))
 
    return _NO_RESULT


 
def a_star(graph: Graph, start: str, goal: str, heuristic: Heuristic) -> SearchResult:
    if start == goal:
        return SearchResult([start], 0.0, 1, [start])

    counter = 0
    frontier: list[tuple[float, int, str, list[str], float]] = [
        (heuristic(start, goal), counter, start, [start], 0.0)
    ]

    g_cost: dict[str, float] = {start: 0.0}
    visit_order: list[str] = []
    nodes_expanded = 0

    while frontier:
        f, _, node, path, g = heapq.heappop(frontier)

        if g > g_cost.get(node, float('inf')):
            continue

        visit_order.append(node)
        nodes_expanded += 1

        if node == goal:
            return SearchResult(path, g, nodes_expanded, visit_order)

        for neighbor, weight in graph.neighbors(node):
            new_g = g + weight

            if new_g < g_cost.get(neighbor, float('inf')):
                g_cost[neighbor] = new_g
                counter += 1
                f_new = new_g + heuristic(neighbor, goal)
                heapq.heappush(
                    frontier,
                    (f_new, counter, neighbor, path + [neighbor], new_g)
                )

    return _NO_RESULT