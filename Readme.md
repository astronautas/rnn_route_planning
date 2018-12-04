# RNN-based Route Planning
This is an on-going research project. 

Classic shortest path algorithms, such as Dijkstra, generate optimal shortest path from A to B. To make the search faster, one can use heuristic-driven algorithms, such as A*. This leads to many problems:
* Developing good heuristics is error-prone, because they're hand-crafted to exploit some specific domain. Even tiny mistakes can cause speed / optimality loss.
* Good heuristics are computationally expensive. The better the heuristic estimate, the closer the estimate is to the cost of a truly optimal path.
* It is difficult to incorporate many features into a heuristic. One needs to decide how each of them impact the cost of an optimal path.

Machine Learning algorithms can approximate any function, thus they can be used to approximate the shortest path function, without explicitly programming it. They can learn feature weights as well. State of the art machine learning models make predictions fast too.

### Objective
The goal of the project is to compare the optimality of a data-driven approach with the fully optimal route planning algorithm Dijkstra in the domain of route planning.

### Tasks
* Develop an ETL chain. Extract the data from OSM, transform and load it for the ML algorithm.
* Develop an RNN model.
* Develop a test and path evaluation environment.

