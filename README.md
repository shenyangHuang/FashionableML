# FashionableML
Deposit for reproducibility study of the paper "Three Factors Influencing Minima in SGD" as submitted to ICLR 2018 (https://openreview.net/forum?id=rJma2bZCW).

Reproducibility study completed by Shenyang Huang, Kaylee Kutschera, and Sacha Perry-Fagant.

Project Presentation:

https://docs.google.com/presentation/d/1d9oEmbm5fWb1WReUxb7EsKWXHjWrcmHX5LGxRCNc7E8/edit?usp=sharing

Instruction:
1. Controllable noise

run 20ReluNN.py

2. Memorization

run tf_mlp.py

To check different subsets, the subsets have to be manually changed on lines 41-45.
To check different ratios, the ratios have to be manually changed on lines 52 and 54.

3. CLR

open jupyter notebook

run CLRFashionMNIST.ipynb
