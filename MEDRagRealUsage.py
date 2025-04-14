import json
benchmark = json.load(open("benchmark.json"))

from utils import QADataset

dataset_name = "mmlu"
dataset = QADataset(dataset_name)

print(len(dataset))


print(dataset[0])
