import numpy as np

class VariableNode:
    def __init__(self, name, initial_belief):
        self.name = name
        self.belief = initial_belief


observed_potentials = [np.array([0.3, 0.7]), np.array([0.6, 0.4])]


key_bit_1 = VariableNode('Key Bit 1', observed_potentials[0])
key_bit_2 = VariableNode('Key Bit 2', observed_potentials[1])

# 信念传播
key_bit_1.belief *= observed_potentials[0]
key_bit_2.belief *= observed_potentials[1]

# 归一化信念
key_bit_1.belief /= key_bit_1.belief.sum()
key_bit_2.belief /= key_bit_2.belief.sum()


print(f"Belief for Key Bit 1: {key_bit_1.belief}")
print(f"Belief for Key Bit 2: {key_bit_2.belief}")
