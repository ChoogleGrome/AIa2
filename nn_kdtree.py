import pandas as pd
import numpy as np
import sys

train_dir = sys.argv[1]
test_dir = sys.argv[2]
dimensions = sys.argv[3]

train_raw_data = pd.read_csv(train_dir, delim_whitespace=True)
test_raw_data = pd.read_csv(test_dir, delim_whitespace=True)

class Node:
    def __init__(self, data=None, axis=None, left=None, right=None):
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, data):
        self.root = self.build_kd_tree(data)

    def build_kd_tree(self, data):
        n, d = data.shape
        if n == 0:
            return None
        else:
            # choose axis based on max spread
            variances = np.var(data[:,:-1], axis=0)
            axis = np.argmax(variances)
            # sort data along axis
            data = data[data[:,axis].argsort()]
            # find median
            mid = n // 2
            # create node and recursively build sub-trees
            node = Node(data[mid], axis)
            node.left = self.build_kd_tree(data[:mid])
            node.right = self.build_kd_tree(data[mid+1:])
            return node

    def knn_search(self, query, k=1):
        # initialize priority queue for nearest neighbors
        pq = []
        # initialize best distance to infinity
        best_dist = np.inf
        # recursive function to search KD tree
        def search_kd_tree(node):
            nonlocal best_dist
            if node is None:
                return
            # compute distance between query and node
            dist = np.linalg.norm(query - node.data)
            # if distance is smaller than current best distance, add to priority queue
            if dist < best_dist:
                pq.append((dist, node.data[-1]))
                # keep priority queue sorted by distance
                pq.sort(key=lambda x: x[0])
                # if priority queue has more than k elements, remove the farthest element
                if len(pq) > k:
                    pq.pop()
                # update best distance
                best_dist = pq[-1][0]
            # check if query is in left or right sub-tree
            if query[node.axis] < node.data[node.axis]:
                search_kd_tree(node.left)
            else:
                search_kd_tree(node.right)

        # start recursive search from root node
        search_kd_tree(self.root)

        # return k nearest neighbors
        return pq[0][1]


# extract the 11 attribute columns and the quality column as a NumPy array
data = train_raw_data.iloc[:,:12].values
test_data = test_raw_data.values

# build the KD tree using the 11 attribute columns as predictors and the quality column as the response
tree = KDTree(data)

# # define a query point consisting of the 11 attribute values and the quality value
# query = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0.0])

# # find the quality of the product using 1-nearest neighbor search
# quality = tree.knn_search(query, k=1)

for i in test_data:
    temp = np.array(i)
    temp = np.append(temp, 0.0)
    print(temp)
    print(tree.knn_search(temp, k=1))

