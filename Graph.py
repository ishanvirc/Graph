import networkx as nx
import matplotlib.pyplot as plt
import heapq
import random

class Graph:
    def __init__(self):
        self.adj_list = {} #stores edges and weights. 
        self.heuristics = {}  # Stores heuristics for A*

    def add_node(self, node, heauristic):
        if node not in self.adj_list:
            self.adj_list[node] = []
            self.heuristics[node] = heauristic

    def add_edge(self, node1, node2, weight):
        if node1 not in self.adj_list:
            self.add_node(node1)
        if node2 not in self.adj_list:
            self.add_node(node2)
        # Append a tuple of (node, weight)
        self.adj_list[node1].append((node2, weight))
        self.adj_list[node2].append((node1, weight)) 

    # DFS Implementation
    def dfs(self, start):
        visited = set()
        stack = [start]

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                print(vertex, end=' ')
                visited.add(vertex)
                stack.extend(neighbor for neighbor, _ in self.adj_list[vertex] if neighbor not in visited)
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
                queue.extend(neighbor for neighbor, _ in self.adj_list[vertex] if neighbor not in visited)
        print()

    # Dijkstra's Algorithm
    def dijkstra(self, start):
        distances = {node: float('infinity') for node in self.adj_list}
        distances[start] = 0
        pq = [(0, start)]

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            for neighbor, weight in self.adj_list[current_vertex]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        return distances


    def a_star(self, start, goal):
        # Initialize the open set with the start node, and the closed set as empty.
        open_set = set([start])
        closed_set = set()

        # Initialize g scores to infinity for all nodes, except the start node, which is 0.
        # g score represents the cost from the start node to the current node.
        g = {node: float('infinity') for node in self.adj_list}
        g[start] = 0

        # Initialize f scores to infinity for all nodes.
        # f score represents the total cost from start to the goal passing by the current node.
        # It's the sum of g score and the heuristic.
        f = {node: float('infinity') for node in self.adj_list}
        f[start] = self.heuristics[start]

        # Initialize the parent map to reconstruct the path later.
        parents = {start: start}

        # Loop until there are no more nodes to evaluate.
        while open_set:
            # Choose the node in open set with the lowest f score.
            current = min(open_set, key=lambda x: f[x])

            # If the goal is reached, reconstruct and return the path.
            if current == goal:
                path = []
                while parents[current] != current:
                    path.append(current)
                    current = parents[current]
                path.append(start)
                path.reverse()  # Reverse the path to start from the beginning
                return path

            # Move the current node from open to closed set.
            open_set.remove(current)
            closed_set.add(current)

            # Iterate through the neighbors of the current node.
            for neighbor, weight in self.adj_list[current]:
                # Skip if the neighbor is already evaluated.
                if neighbor in closed_set:
                    continue

                # Calculate tentative g score for the current path.
                tentative_g_score = g[current] + weight

                # If this path to neighbor is better than any previous one, record it.
                if tentative_g_score < g[neighbor]:
                    # Record the current path as the best for this neighbor.
                    parents[neighbor] = current
                    g[neighbor] = tentative_g_score
                    # Update the f score with the new g score plus heuristic.
                    f[neighbor] = g[neighbor] + self.heuristics[neighbor]

                    # If the neighbor is not in open set, add it for evaluation.
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        # If the loop ends without finding a path, return an empty path.
        return []
    

    #Lowest-Cost-first algorithm: 
    def lcfs(self, start, goal):
        open_set = set([start])
        closed_set = set()
        cost = {node: float('infinity') for node in self.adj_list}
        cost[start] = 0
        parents = {start: start}

        while open_set:
            current = min(open_set, key=lambda x: cost[x])
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

            for neighbor, weight in self.adj_list[current]:
                if neighbor in closed_set:
                    continue
                tentative_cost_score = cost[current] + weight
                
                if tentative_cost_score < cost[neighbor]:
                    parents[neighbor] = current
                    cost[neighbor] = tentative_cost_score
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        return []


    def display_graph(self):
        for node in self.adj_list:
            print(f"{node}: {self.adj_list[node]}")


    def visualize(self):
        G = nx.Graph()
        for node in self.adj_list:
            G.add_node(node)
            for neighbor, weight in self.adj_list[node]:
                G.add_edge(node, neighbor, weight=weight)

        pos = nx.spring_layout(G)  # Node positions
        nx.draw_networkx_nodes(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()})
        plt.show()




# Example usage
G = nx.Graph()
g = Graph()

# Adding nodes with heuristic values
for a in range(1, 8): 
    b =  random.choice(range(1,15))
    G.add_node(a, heuristic=b)
    g.add_node(a, b)

#  Adding an edge with weight
# front-end
G.add_edge(1, 2, weight=1.5)
G.add_edge(3, 4, weight=2)
G.add_edge(2, 4, weight=5)
G.add_edge(1, 3, weight=4)
G.add_edge(3, 6, weight=1)
G.add_edge(6, 5, weight=10)
G.add_edge(6, 7, weight=10)

# back-end
g.add_edge(1, 2, 1.5)
g.add_edge(3, 4, 2)
g.add_edge(2, 4, 5)
g.add_edge(1, 3, 4)
g.add_edge(3, 6, 1)
g.add_edge(6, 5, 10)
g.add_edge(6, 7, 10)

print("DFS:")
g.dfs(1)
print("\nBFS:")
g.bfs(1)
print("\nDijkstra's Algorithm:")
print(g.dijkstra(1))
print("\nLCFS Algorithm:")
print(g.lcfs(1,5))
print("\nA* Algorithm:")
print(g.a_star(1, 7)) 

# Drawing the graph
pos = nx.spring_layout(G)  # positions for all nodes

# Nodes
nx.draw_networkx_nodes(G, pos)

# Edges
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Node labels with heuristic values
node_labels = {node:f"{node}\n(H:{G.nodes[node]['heuristic']})" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels)

plt.show()