import numpy as np
from scipy.stats import norm

# 模拟AES加密时的能耗数据收集
def collect_power_data(key_byte, samples=1000):
    mean_power = key_byte * 10  # 假设能耗与密钥字节成正比
    power_data = np.random.normal(mean_power, 5, samples)  # 加入一些噪声
    return power_data

# 创建模板
def create_templates():
    templates = {}
    for key_byte in range(256):  # AES密钥字节的所有可能值
        power_data = collect_power_data(key_byte)
        templates[key_byte] = {
            'mean': np.mean(power_data),
            'std': np.std(power_data)
        }
    return templates

# 计算概率
def calculate_probabilities(templates, observed_power):
    probabilities = {}
    for key_byte, template in templates.items():
        prob = norm.pdf(observed_power, template['mean'], template['std'])
        probabilities[key_byte] = prob
    return probabilities

# 创建模板
templates = create_templates()

# 假设的密钥
true_key = [123, 45, 67, 89, 12, 34, 56, 78, 90, 11, 22, 33, 44, 55, 66, 77]

# 模拟攻击每个密钥字节
predicted_key = []
for i in range(16):
    observed_power = np.mean(collect_power_data(true_key[i]))
    probabilities = calculate_probabilities(templates, observed_power)
    predicted_byte = max(probabilities, key=probabilities.get)
    predicted_key.append(predicted_byte)

# 输出预测的密钥
print("Predicted Key:", predicted_key)
