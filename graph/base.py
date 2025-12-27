from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Set


@dataclass(frozen=True)
class Vertex:
    id: int
    type: int


@dataclass(frozen=True)
class Edge:
    left_id: int
    right_id: int
    type: int = 0
    weight: float = 1.0


class Graph:
    _vertices: Set[Vertex]
    _edges: Set[Edge]
    _vertices_map: Dict[int, Vertex]
    _vertices_to_edges_map: Dict[int, Set[Edge]]

    def __init__(self, vertices: Optional[List[Vertex]] = None, edges: Optional[List[Edge]] = None):
        self._vertices = set()
        self._edges = set()
        self._vertices_map = {}
        self._vertices_to_edges_map = {}

        if vertices:
            for vertex in vertices:
                self.add_vertex(vertex)
        if edges:
            for edge in edges:
                self.add_edge(edge)

    def add_vertex(self, vertex: Vertex):
        if vertex.id in self._vertices_map:
            raise ValueError(f"Vertex with id {vertex.id} already exists")
        self._vertices.add(vertex)
        self._vertices_map[vertex.id] = vertex
        self._vertices_to_edges_map[vertex.id] = set()

    def add_edge(self, edge: Edge):
        self._check_vertex_exists(edge.left_id)
        self._check_vertex_exists(edge.right_id)

        self._edges.add(edge)
        self._vertices_to_edges_map[edge.left_id].add(edge)
        self._vertices_to_edges_map[edge.right_id].add(edge)

    def remove_vertex(self, vertex: Vertex):
        self._check_vertex_exists(vertex.id)
        self._vertices.remove(self._vertices_map[vertex.id])
        edges_to_remove = set(self._vertices_to_edges_map[vertex.id])
        for edge in edges_to_remove:
            self.remove_edge(edge)

        del self._vertices_to_edges_map[vertex.id]
        del self._vertices_map[vertex.id]

    def remove_edge(self, edge: Edge):
        if edge not in self._edges:
            raise ValueError(f"Edge {edge} not found")

        self._vertices_to_edges_map[edge.left_id].remove(edge)
        self._vertices_to_edges_map[edge.right_id].remove(edge)

        self._edges.remove(edge)

    def clear(self):
        self._vertices.clear()
        self._edges.clear()
        self._vertices_map.clear()
        self._vertices_to_edges_map.clear()

    def get_vertex_by_id(self, vertex_id: int) -> Vertex:
        self._check_vertex_exists(vertex_id)
        return self._vertices_map[vertex_id]

    def get_vertex_edges(self, vertex: Vertex) -> Set[Edge]:
        self._check_vertex_exists(vertex.id)
        return set(self._vertices_to_edges_map[vertex.id])

    def get_vertex_output_edges(self, vertex: Vertex) -> Set[Edge]:
        return {edge for edge in self.get_vertex_edges(vertex) if edge.left_id == vertex.id}

    def get_vertex_input_edges(self, vertex: Vertex) -> Set[Edge]:
        return {edge for edge in self.get_vertex_edges(vertex) if edge.right_id == vertex.id}

    def get_vertices(self) -> Set[Vertex]:
        return set(self._vertices)

    def get_edges(self) -> Set[Edge]:
        return set(self._edges)

    def has_vertex(self, vertex_id: int) -> bool:
        return vertex_id in self._vertices_map

    def _check_vertex_exists(self, vertex_id: int) -> None:
        if vertex_id not in self._vertices_map:
            raise ValueError(f"Vertex {vertex_id} not found")
