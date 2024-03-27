# Decision Tree
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

# Concept
To build a decision tree from a dataset the following steps are needed to be considered.

- start with all examples at the root node.
- Calculate information gain for splitting on all possible features, and pick the one with the highest information gain
- Split the dataset according to the selected feature, and create left and right branches of the tree
- Keep repeating the splitting process until the stopping criteria are met

In this tutorial, we’ll implement the following functions, which will let us split a node into left and right branches using the feature with the highest information gain. The functions are:

Calculate the entropy at a node
- Split the dataset at a node into left and right branches based on a given feature
- Calculate the information gain from splitting on a given feature
- Choose the feature that maximizes information gain.
- 
We’ll then use the helper functions we’ve implemented to build a decision tree by repeating the splitting process until the stopping criteria is met.

# Problem Statement and Dataset
Suppose someone is starting a company that grows and sells wild mushrooms.

- Since not all mushrooms are edible, one would like to be able to tell whether a given mushroom is edible or poisonous based on its physical attributes
- We have some existing data that we can use for this task.
We have 10 examples of mushrooms. For each example, we have

Three features

- Cap Color (Brown or Red),
- Stalk Shape (Tapering (as in \/) or Enlarging (as in /\)), and
- Solitary (Yes or No)
And Label

- Edible (1 indicating yes or 0 indicating poisonous)

# Calculate Entropy

The function takes in a numpy array (y) that indicates whether the examples in that node are edible (1) or poisonous(0)

the $compute_entropy()$ function below

- Compute $P_1$, which is the fraction of examples that are edible (i.e. have value = 1 in y)
- The entropy is then calculated as

$H(p_1)=-p_1log_2(p_1)-(1-p_1)log_2(1-p_1)$

Here:

- The log is calculated with base 2
- For implementation purposes, $0log_2(0)=0$. That is, if $p_1$ = 0 or $p_1 = 1$, set the entropy to 0
- Make sure to check that the data at a node is not empty (i.e. len(y) != 0). Return 0 if it is.

```python
def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    entropy = 0.
    
    if len(y) != 0:
        p1 = len(y[y == 1]) / len(y)
        if p1 != 0 and p1 != 1:
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)       
    
    return entropy
```
# Information Gain
The following function called $information_gain$ that takes in the training data, the indices at a node and a feature to split on and returns the information gain from the split.

The information gain function can be computer through the following equation:

Information gain = H(p_1 ^ node) - ((w^ left H(p_1 ^ Left) + (w^ right H(p_1 ^ Right))

here

- H(p_1 ^ node) is entropy at the node
- H(p_1 ^ Left) and H(p_1 ^ Right) are the entropies at the left and the right branches resulting from the split
- w^left and w^right are the proportion of examples at the left and right branch, respectively.

```python
def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        feature (int):           Index of feature to split on
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    
    information_gain = 0
    
    
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    w_left =len(X_left)/len(X_node)
    w_right =len(X_right)/len(X_node)

    weighted_entropy = w_left*left_entropy+w_right*right_entropy
    
    information_gain = node_entropy-weighted_entropy 
    
    return information_gain
```
# Get the best split
The function $get_best_split()$ to get the best feature to split on by computing the information gain from each feature as we did above and returning the feature that gives the maximum information gain.

- The function takes in the training data, along with the indices of datapoint at that node.
- The output of the function is the feature that gives the maximum information gain.

```python
def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
  
    num_features = X.shape[1]
    
    best_feature = -1

    max_info_gain=0
    for feature in range(num_features):
        info_gain = compute_information_gain(X,y,node_indices,feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature  
   
    return best_feature
```
Combining all the function, we can use a dataset to buit the decision tree. A complete tutorial of this codebase can found in this medium blog [post](https://hasan-shahriar.medium.com/decision-trees-28bff34e7d90)
