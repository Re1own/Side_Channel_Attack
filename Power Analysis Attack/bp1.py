class BinaryXORGraph:
    def __init__(self, trace_length):
        self.trace_length = trace_length
        self.nodes = self._create_nodes()
        self.edges = self._create_edges()

    def _create_nodes(self):
        # 创建密钥和明文的节点
        return ['K{}'.format(i) for i in range(self.trace_length)] + \
               ['P{}'.format(i) for i in range(self.trace_length)]

    def _create_edges(self):
        # 创建表示XOR关系的边
        return [('K{}'.format(i), 'P{}'.format(i)) for i in range(self.trace_length)]

    def get_potential(self, key_bit, plaintext_bit, trace_value):
        # 计算XOR结果
        xor_result = key_bit ^ plaintext_bit

        # 如果XOR结果与跟踪数据匹配，则概率高；否则低
        return 0.9 if xor_result == trace_value else 0.1


# 示例：创建一个16位的跟踪数据的图模型
graph = BinaryXORGraph(16)
print("Nodes:", graph.nodes)
print("Edges:", graph.edges)


import random

def generate_binary_data(length):
    return [random.randint(0, 1) for _ in range(length)]

# 生成密钥和明文
key = generate_binary_data(16)
plaintext = generate_binary_data(16)

# 生成跟踪数据
trace = [k ^ p for k, p in zip(key, plaintext)]

print("Key:", key)
print("Plaintext:", plaintext)
print("Trace:", trace)

# 打印最终的信念
# ...

# 假设我们已经有了跟踪数据 'trace'，以及图模型 'BinaryXORGraph'

class BeliefPropagation:
    def __init__(self, graph, trace):
        self.graph = graph
        self.trace = trace
        self.beliefs = {node: [0.5, 0.5] for node in graph.nodes}  # 初始化信念

    def calculate_potential(self, key_bit, plaintext_bit, trace_bit):
        # 势函数：如果密钥位和明文位的XOR结果与跟踪数据匹配，则概率更高
        return 0.9 if key_bit ^ plaintext_bit == trace_bit else 0.1

    def send_message(self, from_node, to_node, trace_bit):
        message = [0, 0]
        for from_bit in [0, 1]:
            for to_bit in [0, 1]:
                potential = self.calculate_potential(from_bit, to_bit, trace_bit)
                message[to_bit] += potential * self.beliefs[from_node][from_bit]

        total = sum(message)
        normalized_message = [m / total for m in message]
        # print(f"Message from {from_node} to {to_node}: {normalized_message}")  # 打印消息
        return normalized_message

    def update_beliefs(self):
        new_beliefs = {}
        for node in self.graph.nodes:
            belief = [1.0, 1.0]
            for edge in self.graph.edges:
                if node in edge:
                    other_node = edge[0] if edge[1] == node else edge[1]
                    trace_index = int(node[1:]) if node.startswith('K') else int(other_node[1:])
                    message = self.send_message(other_node, node, self.trace[trace_index])
                    belief = [b * m for b, m in zip(belief, message)]

            total = sum(belief)
            new_beliefs[node] = [b / total for b in belief]

        self.beliefs = new_beliefs
        # for node, belief in new_beliefs.items():
        #     print(f"Updated belief for {node}: {belief}")  # 打印更新后的信念

    def run(self, iterations=10):
        for _ in range(iterations):
            self.update_beliefs()

# 创建图模型实例
graph = BinaryXORGraph(16)

# 生成跟踪数据
trace = [k ^ p for k, p in zip(key, plaintext)]

# 创建Belief Propagation实例并运行
bp = BeliefPropagation(graph, trace)
bp.run(iterations=50)  # 增加迭代次数尝试

# 打印最终的信念
for node in graph.nodes:
    print(f"Node {node}: {bp.beliefs[node]}")

new_trace = [0]*16
new_trace[13] = 0
new_trace[15] = 0
# 使用Belief Propagation预测密钥和明文
bp = BeliefPropagation(graph, new_trace)
bp.run()  # 运行BP算法

# 解释BP算法的结果
predicted_key = []
predicted_plaintext = []
for node in graph.nodes:
    # 选择具有最高概率的状态作为预测值
    predicted_state = 0 if bp.beliefs[node][0] > bp.beliefs[node][1] else 1
    if node.startswith('K'):
        predicted_key.append(predicted_state)
    else:
        predicted_plaintext.append(predicted_state)

# 重构密钥和明文
predicted_key = ''.join(map(str, predicted_key))
predicted_plaintext = ''.join(map(str, predicted_plaintext))

print("Predicted Key:", predicted_key)
print("Predicted Plaintext:", predicted_plaintext)

