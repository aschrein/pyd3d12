# MIT License
# 
# Copyright (c) 2025 Anton Schreiner
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import random

# Compute graph basic building block
class AutoGradNode:
    def __init__(self):
        # scalar valued gradient accumulator for the final dL/dp
        self.grad = 0.0
        # dependencies for causation sort
        self.dependencies = []

    def zero_grad(self):
        self.grad = 0.0

    # Overload operators to build the computation graph
    def __add__(self, other): return Add(self, other)
    def __mul__(self, other): return Mul(self, other)
    def __sub__(self, other): return Sub(self, other)

    # Get a topologically sorted list of dependencies
    # starts from the leaf nodes and terminates at the root
    def get_topo_sorted_list_of_deps(self):
        visited = set()
        topo_order = []

        def dfs(node): # depth-first search
            if node in visited:
                return
            visited.add(node)
            for dep in node.dependencies:
                dfs(dep)
            topo_order.append(node)

        dfs(self)

        return topo_order

    def get_pretty_name(self): return self.__class__.__name__

    # Pretty print the computation graph in DOT format
    def pretty_print_dot_graph(self):
        topo_order = self.get_topo_sorted_list_of_deps()
        _str = ""
        _str += "digraph G {\n"
        for node in topo_order:
            _str += f"    {id(node)} [label=\"{node.get_pretty_name()}\"];\n"
            for dep in node.dependencies:
                _str += f"    {id(node)} -> {id(dep)};\n"
        _str += "}"
        return _str
    
    def backward(self):
        topo_order = self.get_topo_sorted_list_of_deps()

        for node in topo_order:
            node.zero_grad() # we don't want to accumulate gradients

        self.grad = 1.0 # seed the gradient at the output node

        # Reverse the topological order for backpropagation to start from the output
        for node in reversed(topo_order):
            # from the tip of the  graph down to leaf learnable parameters
            # Distribute gradients
            node._backward()

    # The job of this method is to propagate gradients backward through the network
    def _backward(self):
        assert False, "Not implemented in base class"

    # Materialize the numerical value at the node
    # i.e. Evaluate the computation graph
    def materialize(self):
        assert False, "Not implemented in base class"

# Any value that is not learnable
class Variable(AutoGradNode):
    def __init__(self, value, name=None):
        super().__init__()
        self.value = value
        self.name = name

    def get_pretty_name(self):
        if self.name:
            return f"Variable({self.name})"
        else:
            return str(self.value)

    def materialize(self): return self.value

    def _backward(self):
        pass

Constant = Variable

# Learnable parameter with initial random value 0..1
class LearnableParameter(AutoGradNode):
    def __init__(self):
        super().__init__()
        self.value = random.random()

    def get_pretty_name(self):
        return f"LearnableParameter({round(self.value, 2)})"

    def materialize(self): return self.value

    def _backward(self):
        pass

class Abs(AutoGradNode):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.dependencies = [a]

    def materialize(self): return abs(self.a.materialize())

    def _backward(self):
        self.a.grad += self.grad * (1.0 if self.a.materialize() > 0 else -1.0)

class Square(AutoGradNode):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.dependencies = [a]

    def materialize(self): return self.a.materialize() ** 2

    def _backward(self):
        self.a.grad += self.grad * 2.0 * self.a.materialize()

class Sub(AutoGradNode):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.dependencies = [a, b]

    def materialize(self): return self.a.materialize() - self.b.materialize()

    def _backward(self):
        self.a.grad += self.grad
        self.b.grad -= self.grad

class Add(AutoGradNode):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.dependencies = [a, b]

    def materialize(self): return self.a.materialize() + self.b.materialize()

    def _backward(self):
        self.a.grad += self.grad
        self.b.grad += self.grad

class Mul(AutoGradNode):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.dependencies = [a, b]

    def materialize(self): return self.a.materialize() * self.b.materialize()

    def _backward(self):
        self.a.grad += self.grad * self.b.materialize()
        self.b.grad += self.grad * self.a.materialize()

a = LearnableParameter()
b = LearnableParameter()

for epoch in range(3000):

    x = Variable(random.random(), name="x")
    z = x * x * a + b
    loss = Square(z - (x * x * Constant(1.777) + Constant(1.55))) # L2 loss to Ax^2+B

    print(f"Epoch {epoch}: loss = {loss.materialize()}; a = {a.materialize()}, b = {b.materialize()}")
    # Backward pass
    # Gradient reset happens internally in the backward pass
    loss.backward()

    # Update parameters
    learning_rate = 0.01333

    for node in [a, b]:
        # print(f"grad = {node.grad}")
        node.value -= learning_rate * node.grad

with open(".tmp/graph.dot", "w") as f:
    f.write(loss.pretty_print_dot_graph())

# Output:
# Epoch 2999: loss = 1.0971718506497338e-07; a = 1.7761125496912944, b = 1.5503818948331147
# Target: 1.777, 1.55
