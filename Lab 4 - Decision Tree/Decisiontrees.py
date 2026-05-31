import csv
import math
import os
def load_csv(filename):
    data=[]
    filepath=os.path.join(os.path.dirname(__file__),filename)
    with open(filepath,'r') as file:
        reader=csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row[1:])
    return data
def entropy(data):
    total=len(data)
    label_counts={}
    for row in data:
        label=row[-1]
        label_counts[label] = label_counts.get(label,0)+1
    ent=0
    for count in label_counts.values():
        p=count/total
        ent-=p*math.log2(p)
    return ent
def information_gain(data, feature_index):
    total_entropy = entropy(data)
    feature_values = {}
    for row in data:
        value = row[feature_index]
        if value not in feature_values:
            feature_values[value] = []
        feature_values[value].append(row)
    weighted_entropy = 0
    for subset in feature_values.values():
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted_entropy
def best_feature(data):
    gains = []
    for i in range(len(data[0]) - 1):
        gains.append(information_gain(data, i))
    return gains.index(max(gains))
def majority_class(data):
    labels = [row[-1] for row in data]
    return max(set(labels), key=labels.count)
def build_tree(data, features):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(features) == 0:
        return majority_class(data)
    best = best_feature(data)
    best_name = features[best]
    tree = {best_name: {}}
    values = set(row[best] for row in data)
    for value in values:
        subset = []
        for row in data:
            if row[best] == value:
                reduced_row = row[:best] + row[best + 1:]
                subset.append(reduced_row)
        new_features = features[:best] + features[best + 1:]
        tree[best_name][value] = build_tree(subset, new_features)
    return tree
def predict(tree, features, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    root_index = features.index(root)
    value = sample[root_index]
    if value not in tree[root]:
        return "Unknown"
    subtree = tree[root][value]
    new_features = features[:root_index] + features[root_index + 1:]
    new_sample = sample[:root_index] + sample[root_index + 1:]
    return predict(subtree, new_features, new_sample)
data = load_csv("play_tennis.csv")
features = ["Outlook", "Temperature", "Humidity", "Wind"]
tree = build_tree(data, features)
print("Decision Tree:")
print(tree)
sample = ["Sunny", "Cool", "High", "Strong"]
result = predict(tree, features, sample)
print("\nTest Sample:", sample)
print("Prediction:", result)