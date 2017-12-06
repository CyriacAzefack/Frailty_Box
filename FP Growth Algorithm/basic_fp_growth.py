# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:52:26 2017

@author: cyriac.azefack
"""

import itertools
from graphviz import Digraph



class FPNode(object) :
    """
    Node in the FP-Tree
    """
    
    ID = 0
    def __init__(self, label, count, parent) :
        """
        Initialisation of a Node
        """
        FPNode.ID += 1
        self.label = label
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []
        self.id = FPNode.ID
        

    def add_child(self, label):
        """
        Add a new child to the node
        """
        child = FPNode(label, 1, self)
        self.children.append(child)

        return child

    def get_child (self, label) :
        """
        Retrieve the child with the given label
        """
        for child in self.children:
            if child.label == label:
                return child
        return None


class FPTree(object) :
    """
    FP-Tree
    """

    def __init__(self, transactions, threshold, root_label, root_count):
        """
        initialise the tree
        """
        self.frequent_items = self.find_frequent_items(transactions, threshold)
        
        self.headers = self.build_header_table(self.frequent_items)
        
        self.root = self.build_fp_tree(transactions, root_label, root_count, 
                                  self.frequent_items, self.headers)
        
        


    @staticmethod
    def find_frequent_items(transactions, threshold):
        """
        Dictionnary of frequent items (atleast 'threshold' occurences) { Item_Name : nb_occurences}
        """
        frequent_items = {}

        for trans in transactions :
            for item in trans:
                if item in frequent_items.keys():
                    frequent_items[item] += 1
                else:
                    frequent_items[item] = 1

        for key in list(frequent_items.keys()):
                if frequent_items[key] < threshold:
                    del frequent_items[key]

        return frequent_items
    
    def build_header_table(self, frequent_items):
        """
        Dictionnary {Item_Name : Node-Link}
        """
        headers = {}
        
        for item in frequent_items.keys():
            headers[item] = None
        
        return headers
    
    def build_fp_tree(self, transactions, root_label, root_count, frequent_items, headers) :
        """
        build the FP-Tree
        """
        
        self.trans_id = 0
        root = FPNode(root_label, root_count, None)
        
        for trans in transactions:
            self.trans_id += 1
            sorted_items = [x for x in trans if x in frequent_items]
            sorted_items.sort(key=lambda x: frequent_items[x], reverse=True)
            
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers)
            
            #self.plot_tree(self.trans_id, root)
        
        return root
    
    def insert_tree(self, sorted_items, node, headers):
        """
        Insert a list of frequent items to the tree
        """
        first_item = sorted_items[0]
        child = node.get_child(first_item)
        
        if child is not None:
            child.count += 1
        else:
            child = node.add_child(first_item)
            
            current = headers[first_item]
            
            
            #print(node.label + " : " + str(node.count))
            #print(sorted_items)
           
            if current is None:
                headers[first_item] = child
            else :
                #current = current.link
                while current.link is not None:
                    current = current.link
                current.link = child
        
        if len(sorted_items) > 1:
            self.insert_tree(sorted_items[1:], child, headers)
            
    def mine_patterns(self, threshold) :
        
        if self.has_single_path(self.root):
            return self.generate_pattern_list()
        else :
            return self.zip_patterns(self.mine_sub_trees(threshold))
        
    def mine_sub_trees(self, threshold):
        patterns = {}
        
        #From leaves to root
        mining_order = sorted(self.frequent_items, 
                              key=lambda x : self.frequent_items[x])
        
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item, 
            # trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.label)
                    parent = parent.parent

                for i in range(frequency):
                    conditional_tree_input.append(path)

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            subtree = FPTree(conditional_tree_input, threshold,
                             item, self.frequent_items[item])
            subtree_patterns = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns
    
    
    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if
        we are in a conditional FP tree.
        """
        suffix = self.root.label

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]

            return new_patterns

        return patterns
        
    
    def has_single_path(self, node):
        """
        True if the current tree has a single Path
        """

        l = len(node.children)
        if l > 0:
            return False
        elif  l == 0:
            return True
        else :
            return True and self.has_single_path(node.children[0])
    
    def generate_pattern_list(self) :
        """
        Generate the pattern list for a single path tree
        """
        patterns = {}
        items = self.frequent_items.keys()
        
        # If we are in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.label is None:
            suffix_value = []
        else:
            suffix_value = [self.root.label]
            patterns[tuple(suffix_value)] = self.root.count
        
        for i in range(1, len(items)+1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = min([self.frequent_items[x] for x in subset])
        
        return patterns
    
    def plot_tree(self, trans_id, root):
        """
        Display the FP-Tree
        """
        visited = []
        
        dot = Digraph('G', filename='tmp/tree_'+str(trans_id)+'.gv')
        
        current = root
        for child in current.children:
            self.add_edge(dot, current, child, visited)
        
        dot.view()
    
    def add_edge(self, dot, parent, child, visited):
        
        
        if parent not in visited :
            visited.append(parent)
            if parent.label is not None:
                dot.node(str(parent.id), parent.label+" ["+str(parent.count)+"]")
            else :
                dot.node(str(parent.id), "null [None]")
        
        if child not in visited :
            visited.append(child)
            dot.node(str(child.id), child.label+" ["+str(child.count)+"]")
            
        dot.edge(str(parent.id), str(child.id))
        for c in child.children:
            self.add_edge(dot, child, c, visited)
            

def find_frequent_patterns(transactions, support_treshold) :
    """
    Find the frequent patterns in the transactions
    """
    tree = FPTree(transactions, support_treshold, None, None)
    patterns = tree.mine_patterns(support_treshold)
    
    
    
    return patterns

if __name__ == "__main__":
    
    
    transactions = [["A", "B", "D", "E"], ["B", "C", "E"], ["A", "B", "D", "E"], 
                   ["A", "B", "C", "E"], ["A", "B", "C", "D", "E"], ["B", "C", "D"]]
    
    """
    transactions = [["kitchen", "breakfast", "coffee"], ["kitchen", "lunch", "coffee"], 
                    ["kitchen", "breakfast", "coffee"], ["kitchen", "lunch", "coffee"], 
                    ["kitchen", "breakfast", "coffee"], ["kitchen", "lunch"], 
                    ["kitchen", "breakfast", "coffee"], ["kitchen", "lunch", "coffee"]]
    """
    tree = FPTree(transactions, 2, None, None);
    
    print(find_frequent_patterns(transactions, 2))
    #tree.plot_tree(0, tree.root);
    
    
    