def rename_edges(
    edges: list[tuple[Any, Any]],
    renaming_map: dict,
) -> list[tuple[Any, Any]]:
    renamed_edges = []
    for edge in edges:
        if edge[0] in renaming_map and edge[1] in renaming_map:
            renamed_edges.append((renaming_map[edge[0]], renaming_map[edge[1]]))
        else:
            renamed_edges.append(edge)
