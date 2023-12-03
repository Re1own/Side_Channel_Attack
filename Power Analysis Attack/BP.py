import numpy as np
class MarkovChain:
    def __init__(self, nodes, edges, observations):
        self.nodes = nodes
        self.edges = edges
        self.observations = observations
        self.beliefs = {node: np.array([0.5, 0.5]) for node in nodes}  # Uniform initial beliefs

    def get_neighbors(self, node):
        neighbors = []
        for edge in self.edges:
            if edge[0] == node:
                neighbors.append(edge[1])
            elif edge[1] == node:
                neighbors.append(edge[0])
        return neighbors

    def get_potential(self, state1, state2):
        return 0.8 if state1 == state2 else 0.2

    def get_observation_potential(self, node, state):
        # Potential based on observation (0.6 for correct, 0.4 for incorrect observation)
        observed_state = self.observations[self.nodes.index(node)]
        return 0.6 if state == observed_state else 0.4

class BeliefPropagationForMarkovChain:
    def __init__(self, markov_chain):
        self.mc = markov_chain
        self.messages = {}
        for edge in markov_chain.edges:
            self.messages[edge] = np.ones(2)  # edge: (from_node, to_node)
            self.messages[(edge[1], edge[0])] = np.ones(2)  # 反方向的边

    def send_message(self, from_node, to_node):
        message = np.zeros(2)
        for state in [0, 1]:
            total = 0
            for from_state in [0, 1]:
                total += self.mc.get_potential(from_state, state) * self.mc.beliefs[from_node][from_state]
            message[state] = total
        return message

    def update_beliefs(self):
        for node in self.mc.nodes:
            belief = np.array([1.0, 1.0])
            for neighbor in self.mc.get_neighbors(node):
                if (neighbor, node) in self.messages:
                    belief *= self.messages[(neighbor, node)]
                else:
                    belief *= self.messages[(node, neighbor)]

            for state in [0, 1]:
                belief[state] *= self.mc.get_observation_potential(node, state)

            self.mc.beliefs[node] = belief / sum(belief)

    def run(self, iterations=10):
        for _ in range(iterations):
            new_messages = {}
            for edge in self.mc.edges:
                new_messages[edge] = self.send_message(*edge)
            self.messages.update(new_messages)
            self.update_beliefs()

# Define the Markov Chain model
nodes = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')]
observations = [0, 0, 1, 1, 1]  # Observed states



markov_chain = MarkovChain(nodes, edges, observations)
bp = BeliefPropagationForMarkovChain(markov_chain)


bp.run()


for node in markov_chain.nodes:
    print(f"Node {node}: {markov_chain.beliefs[node]}")

