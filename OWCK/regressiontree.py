# Author: Bas van Stein <bas9112@gmail.com>
# 
from __future__ import print_function
import numpy as np

# Regression tree class in python.

class IncrementalRegressionTree:
	"""
	A simple Regression Tree class that can be 
	extended by splitting a leaf node in two new node.
	Parameters:
		min_leaf_split = minimum number of samples to use for splitting a node, default is one
		max_depth = maximum depth of the tree. None is infinite.
		min_samples_leaf
		criterion = ["mse","median","gini","mae"]
	"""

	def __init__(self, min_leaf_split=1,max_depth=None, min_samples_leaf=1, regression=True, criterion="mse", verbose=False):
		self.min_leaf_split = min_leaf_split
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.node_count = 0
		self.depth = 1
		self.tree_ = None
		self.verbose = verbose
		self.criterion= criterion
		if self.verbose:
			print("min samples is "+str(self.min_samples_leaf))
		self.leaf_id = 0
		self.regression = regression



	def mse(self, groups):
		mse = 0.0
		for group in groups:
			size = len(group[0]) #left or right
			if size == 0:
				continue
			mse += np.sum((group[1] - np.mean(group[1])) ** 2)
		return mse

	def mae(self, groups):
		mae = 0.0
		for group in groups:
			size = len(group[0]) #left or right
			if size == 0:
				continue
			mae += np.sum(np.abs(group[1] - np.mean(group[1])))
		return mae

	# Calculate the Gini index for a split dataset
	def median_split(self,groups):
		sizes = []
		for group in groups:
			size = len(group[0]) #left or right
			sizes.append(size)
		
		return 1. - float(min(sizes))/float(max(sizes))

	# Calculate the Gini index for a split dataset
	def gini_index(self,groups, class_values):
		gini = 0.0
		for class_value in class_values:
			for group in groups:
				size = len(group[0]) #left or right
				if size == 0:
					continue
				proportion = group[1].count(class_value) / float(size) #group[1] = target labels
				gini += (proportion * (1.0 - proportion))
		return gini

	# Calculate the Gini index for a split dataset
	def balanced_mse(self,groups):
		sizes = []
		mse = 0.0
		for group in groups:
			size = len(group[0]) #left or right
			sizes.append(size)
			if size == 0:
				continue
			mse += np.sum((group[1] - np.mean(group[1])) ** 2)
		balance = 2. - float(min(sizes)) / float(max(sizes))

		return balance * mse

	# Split a dataset based on an attribute and an attribute value
	def test_split(self,index, value, X, y):
		left, right = list(), list()
		lefttarget, righttarget = list(), list()
		for row,val in zip(X,y):
			if row[index] < value:
				left.append(row)
				lefttarget.append(val)
			else:
				right.append(row)
				righttarget.append(val)
		return (left, lefttarget), (right, righttarget)


	# Select the best split point for a dataset
	def get_split(self,X,y):
		class_values = np.unique(y)
		b_index, b_value, b_score, b_groups = 99999, float("inf"), float("inf"), None
		for index in range(len(X[0])):
			for row in X:
				if(self.criterion=="median"):
					index = np.random.randint(0,len(X[0]))
				groups = self.test_split(index, row[index], X, y)
				if (self.criterion == "mse"):
					n_score = self.mse(groups)
				elif (self.criterion == "mae"):
					n_score = self.mae(groups)
				elif (self.criterion == "median"):
					n_score = self.median_split(groups)
				elif(self.criterion=="gini"):
					n_score = self.gini_index(groups,class_values)
				elif(self.criterion=="balanced_mse"):
					n_score = self.balanced_mse(groups)
				#else error
				if n_score < b_score:
					b_index, b_value, b_score, b_groups = index, row[index], n_score, groups

		return {'index':b_index, 'value':b_value, 'groups':b_groups,'node_id':self.node_count}


	

	# Create a terminal node value
	def to_terminal(self,data):
		self.leaf_id += 1
		self.node_count += 1
		return {'id':self.leaf_id, 'node_id':self.node_count, 'val':np.mean(data[1])} #might want to add 'data':data,

	# Create child splits for a node or make terminal
	def split(self,node,parent=None,direction="",parent_group=None):
		left, right = node['groups']
		del(node['groups'])


		# check for a no split
		if not left[0]:
			node['right'] = self.to_terminal(right)
			node['left'] = None
			return
		if not right[0]:
			node['left'] = self.to_terminal(left)
			node['right'] = None
			return
		# check for max depth
		if self.depth >= self.max_depth:
			node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
			return
		# process left child
		if len(left[0]) <= self.min_leaf_split:
			node['left'] = self.to_terminal(left)
		else:
			temp_node = self.get_split(left[0],left[1])
			if (len(temp_node['groups'][0][1])>=self.min_samples_leaf and len(temp_node['groups'][1][1]) >= self.min_samples_leaf):
				node['left'] = temp_node
				self.depth+=1
				self.node_count+=1
				self.split(node['left'],node,"left",left)
			else:
				node['left'] = self.to_terminal(left)
		# process right child
		if len(right) <= self.min_leaf_split:
			node['right'] = self.to_terminal(right)
		else:
			temp_node = self.get_split(right[0],right[1])
			if (len(temp_node['groups'][0][1])>=self.min_samples_leaf and len(temp_node['groups'][1][1]) >= self.min_samples_leaf):
				node['right'] = temp_node
				self.depth+=1
				self.node_count+=1
				self.split(node['right'],node,"right",right)
			else:
				node['right'] = self.to_terminal(right)

	# Build a decision tree
	def fit(self,X,y):
		self.X = X
		self.y = y
		if (self.max_depth==None):
			self.max_depth = len(y)
		self.leaf_labels = np.zeros(len(y))
		self.tree_ = self.get_split(X,y)
		self.split(self.tree_)
		return self.tree_

	def getRoot():
		return self.tree_

	def __str__(self):
		return self.printtree__string(self.tree_)

	def __repr__(self):
		return "IncrementalRegressionTree"

	# Print a decision tree
	def printtree__string(self,node, depth=0):
		str = ""
		if isinstance(node, dict): #always
			if ('val' in node.keys()):
				str +=('|%s[%s, id=%s]' % ((depth*' ', node['val'], node['id'])))
			else:
				str += ('|%s[X%d < %.3f]' % ((depth*'-', (node['index']+1), node['value'])))+"\n"
				str += self.printtree__string(node['left'], depth+1)+"\n"
				str += self.printtree__string(node['right'], depth+1)
			
		return str

	def printtree_(self,node, depth=0):
		print(self.printtree__string( node, depth))

	# Make a prediction with a decision tree
	def predict(self, x):
		"Returns the prediction of the terminal node for x"
		x = np.array(x)
		if len(x.shape) == 1:
			return self._predict_rec(self.tree_,x)
		else:
			return [self._predict_rec(self.tree_, x1) for x1 in x]

	def _predict_rec(self,node, row):
		if row[node['index']] < node['value']:
			if ('val' not in node['left'].keys() ):
				return self._predict_rec(node['left'], row)
			else:
				return node['left']['val']
		else:
			if ('val' not in node['right'].keys() ):
				return self._predict_rec(node['right'], row)
			else:
				return node['right']['val']

	def apply(self, x):
		"Returns the leaf id of the terminal node for x"
		x = np.array(x)
		if len(x.shape) == 1:
			return self._apply_rec(self.tree_,x)
		else:
			return [self._apply_rec(self.tree_, x1) for x1 in x]

	def _apply_rec(self,node, row):
		if row[node['index']] < node['value']:
			if ('val' not in node['left'].keys() ):
				return self._apply_rec(node['left'], row)
			else:
				return node['left']['id']
		else:
			if ('val' not in node['right'].keys() ):
				return self._apply_rec(node['right'], row)
			else:
				return node['right']['id']

	def split_terminal(self, terminal_id, new_x, new_y):
		#first find the specific terminal node
		parent,direction = self.__find_terminal_parent(self.tree_, terminal_id)
		temp_node = self.get_split(new_x,new_y)
		if (len(temp_node['groups'][0][1])>=self.min_samples_leaf and len(temp_node['groups'][1][1]) >= self.min_samples_leaf):
			parent[direction] = temp_node
			#print(len(temp_node['groups'][0][1]),len(temp_node['groups'][1][1]))
			self.split(parent[direction], parent,direction, (new_x,new_y))
			return True
		return False

	def __find_terminal_parent(self,node, node_id):
		if isinstance(node, dict): #should be always
			if ('val' in node.keys()):
				return None,None
			elif (isinstance(node['left'],dict) and 'val' in node['left'].keys() and node['left']['id'] == node_id):
				return node,"left"
			elif (isinstance(node['right'],dict) and 'val' in node['right'].keys() and node['right']['id'] == node_id):
				return node,"right"
			else:
				#first check left subtree
				f_node, direction = self.__find_terminal_parent(node['left'], node_id)
				if (f_node == None):
					f_node, direction = self.__find_terminal_parent(node['right'], node_id)
		else:
			return None,None
		return f_node, direction

if __name__ == "__main__":
	#tests
	dataset = [[2.771244718,1.784783929],
		[1.728571309,1.169761413],
		[3.678319846,2.81281357],
		[3.961043357,2.61995032],
		[2.999208922,2.209014212],
		[7.497545867,3.162953546],
		[9.00220326,3.339047188],
		[7.444542326,0.476683375],
		[10.12493903,3.234550982],
		[6.642287351,3.319983761]]

	new_data = [[2.771244718,1.784783929],
		[1.728571309,1.169761413],
		[3.678319846,2.81281357],
		[3.961043357,2.61995032],
		[2.999208922,2.209014212]]
	new_y = [0,0,0,0,0]

	target = [0,0,0,0,0,1,1,1,1,1]
	tree = IncrementalRegressionTree(min_samples_leaf=3)
	tree.fit(dataset,target)

	leaf_label = tree.apply(dataset)
	predict = tree.predict(dataset)
	print(leaf_label)
	print(predict)
	print(tree)
	print("New tree:")
	tree.min_samples_leaf=2
	tree.split_terminal(1,new_data, new_y)
	leaf_label = tree.apply(dataset)
	print(leaf_label)
	print(tree)