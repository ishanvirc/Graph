import networkx as nx
import matplotlib.pyplot as plt
import heapq

class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_node(self, node):
        if node not in self.adj_list:
            self.adj_list[node] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj_list:
            self.add_node(node1)
        if node2 not in self.adj_list:
            self.add_node(node2)

        self.adj_list[node1].append(node2)
        self.adj_list[node2].append(node1)

# DFS Implementation
    def dfs(self, start):
        visited = set()
        stack = [start]

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                print(vertex, end=' ')
                visited.add(vertex)
                stack.extend(neighbor for neighbor in self.adj_list[vertex] if neighbor not in visited)
        print()


    # BFS Implementation
    def bfs(self, start):
        visited = set()
        queue = [start]

        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                print(vertex, end=' ')
                visited.add(vertex)
                queue.extend(neighbor for neighbor in self.adj_list[vertex] if neighbor not in visited)
        print()

    # Dijkstra's Algorithm
    def dijkstra(self, start):
        distances = {node: float('infinity') for node in self.adj_list}
        distances[start] = 0
        pq = [(0, start)]

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)

            if current_distance > distances[current_vertex]:
                continue

            for neighbor in self.adj_list[current_vertex]:
                distance = current_distance + 1  # Assuming each edge has a weight of 1

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return distances

    # A* Algorithm
    def a_star(self, start, goal, heuristic):
        open_set = set([start])
        closed_set = set()
        g = {node: float('infinity') for node in self.adj_list}
        g[start] = 0
        parents = {start: start}

        while open_set:
            current = min(open_set, key=lambda x: g[x] + heuristic(x, goal))
            if current == goal:
                path = []
                while parents[current] != current:
                    path.append(current)
                    current = parents[current]
                path.append(start)
                path.reverse()
                return path

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.adj_list[current]:
                if neighbor in closed_set:
                    continue
                tentative_g_score = g[current] + 1  # Assuming each edge has a weight of 1

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g[neighbor]:
                    continue

                parents[neighbor] = current
                g[neighbor] = tentative_g_score

        return []


    def display_graph(self):
        for node in self.adj_list:
            print(f"{node}: {self.adj_list[node]}")

    def visualize(self):
        G = nx.Graph()
        for node in self.adj_list:
            G.add_node(node)
            for neighbour in self.adj_list[node]:
                G.add_edge(node, neighbour)

        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

# Example usage
g = Graph()
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(2, 4)
g.add_edge(2, 3)
g.add_edge(5, 4)
g.add_edge(2, 6)
g.add_edge(2, 5)
g.add_edge(1, 5)
g.add_edge(1, 6)

print("DFS:")
g.dfs(1)
print("\nBFS:")
g.bfs(1)
print("\nDijkstra's Algorithm:")
print(g.dijkstra(1))
print("\nA* Algorithm:")
print(g.a_star(1, 5, lambda x, y: abs(x - y)))  # Example heuristic function

g.display_graph()
g.visualize()
