import argparse
from graph import load_graph_from_csv, Player
from searches import bfs, greedy_best_first


def baseline_heuristic(node: str, goal: str) -> float:
    """Simple baseline heuristic for greedy best-first search."""
    return 0.0 if node == goal else 1.0


def print_result(name: str, result, inventory: set[str], show_visit_order: bool) -> None:
    print(f"=== {name} ===")
    print(f"Found goal: {result.found}")
    if result.path:
        print(f"Path: {' -> '.join(result.path)}")
    else:
        print("Path: [none]")
    print(f"Cost: {result.cost:.4f}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    if show_visit_order:
        print(f"Visit order: {' -> '.join(result.visited_order)}")
    print(f"Final inventory: {sorted(inventory)}")
    print()


def run_comparison(csv_file: str, start: str, goal: str, show_visit_order: bool = False) -> None:
    graph = load_graph_from_csv(csv_file)

    print("Nodes in graph:")
    for title in sorted(graph.node_data):
        node = graph.node_data[title]
        edge_summary = [(e.target, e.cost) for e in node.edges]
        print(
            f"{node.title}: needs={sorted(node.needs)}, "
            f"receives={sorted(node.receives)}, "
            f"edges={edge_summary}"
        )
    print()

    if start not in graph.node_set:
        raise ValueError(f"Start node not found: {start}")
    if goal not in graph.node_set:
        raise ValueError(f"Goal node not found: {goal}")

    bfs_player = Player(inventory={"Child"})
    greedy_player = Player(inventory={"Child"})

    bfs_result = bfs(graph, start, goal, player=bfs_player)
    greedy_result = greedy_best_first(
        graph,
        start,
        goal,
        baseline_heuristic,
        player=greedy_player,
    )

    print(f"CSV:   {csv_file}")
    print(f"Start: {start}")
    print(f"Goal:  {goal}\n")

    print_result("BFS", bfs_result, bfs_player.inventory, show_visit_order)
    print_result("Greedy Best-First", greedy_result, greedy_player.inventory, show_visit_order)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BFS and Greedy Best-First on the same CSV graph."
    )
    parser.add_argument(
        "--csv",
        default="zelda.csv",
        help="Path to CSV graph file (default: zelda.csv)",
    )
    parser.add_argument("--start", required=True, help="Start node title")
    parser.add_argument("--goal", required=True, help="Goal node title")
    parser.add_argument(
        "--show-visit-order",
        action="store_true",
        help="Include full visit order in output",
    )
    args = parser.parse_args()

    run_comparison(args.csv, args.start, args.goal, show_visit_order=args.show_visit_order)


if __name__ == "__main__":
    main()
