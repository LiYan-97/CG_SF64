# CG_SF64
This code uses Computational Graph(CG) and Neural Network(NN) to solve the five-layer Traffic Demand Estimation(TDE) in Sioux Falls(SF) network. A **5TCG** (five-layer traffic computational graph) model is established, including traffic generation flow > traffic OD(Origin-Destination) flow > traffic path flow > traffic link flow > traffic intersection flow. 

It also includes **comparison of models accuracy** with 4TCG model which was established by _Wu Xin[1]_ and developed a **10-fold cross-validation** framework for CG.
# Reference 
[1] Wu, X., Guo, J., Xian, K., & Zhou, X. (2018b). Hierarchical travel demand estimation using multiple data sources: A forward and backward propagation algorithmic framework on a layered computational graph. Transportation Research Part C-Emerging Technologies, 96, 321-346. (https://www.sciencedirect.com/science/article/abs/pii/S0968090X18306685)
# Development settings
Python & Anaconda & Tensorflow

Py 3.8 & tf 2.2
