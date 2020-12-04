# A sparsity-inducing matrix A is attached to a normal convolution. The matrix acts as the hinge between filter
pruning and decomposition. By enforcing group sparsity to the columns and rows of the matrix, equivalent pruning and
decomposition operations can be obtained. For pruning, the product of W and the column-reduced matrix Ac, i.e. Wc acts
as the new convolutional filter. To save computation during decomposition the reduced matrices Wr and Ar are used as two
convolutional filters.


The hinge point between pruning and decomposition is group sparsity, see Fig.Consider a 4D convolutional filter, reshaped into a 2D matrix W ∈ R features×outputs. Group sparsity is added by introducing a sparsity-inducing matrix A. 
By applying group sparsity constraints on the columns of A, the output channel of the sparsity-inducing matrix A and equivalently of the matrix product W × A can be reduced by solving an optimization problem. This is equivalent to filter pruning.
On the other hand, if the group sparsity constraints are applied on the rows of A, then the inner channels of the matrix product W × A, namely, the output channel of W and the input channel of A, can be reduced.
To save the computation, the single heavyweight convolution W is converted to a lightweight and a 1×1 convolution with respect to the already reduced matrices Wr and Ar.


Starting from the perspective of compact tensor approximation, the connection between filter pruning and decomposition is analyzed. Although this perspective is the core of filter decomposition, it is still novel for network pruning. Actually, both of the methods approximate the weight tensor with compact representation that keeps the accuracy of the network.


Based on the analysis, we propose to use sparsityinducing matrices to hinge filter pruning and decomposition and bring them under the same formulation. This square matrix is inspired by filter decomposition and corresponds to a 1 × 1 convolution. By changing the way how the sparsity regularizer is applied to the matrix, our algorithm can achieve equivalent effect of either filter pruning or decomposition or both. To the best of our knowledge, this is the first work that tries to analyze the two methods under the same umbrella.


The third contribution is the developed binary search, gradient based learning rate adjustment, layer balancing, and annealing methods that are important for the success of the proposed algorithm. Those details are obtained by observing the influence of the proximal gradient method on the filter during the optimization.


The proposed method can be applied to various CNNs. We apply this method to VGG, ResNet, ResNeXt, WRN, and DenseNet.The proposed network compression method achieves stateof-the-art performance on those networks.



The rest of the paper is organized as follows. Sec. 2 discusses the related work. Sec. 3 explains the proposed network compression method. Sec. 4 describes the implementation considerations. The experimental results are shown in Sec. 5. Sec. 6
 concludes this paper.


2. Related Work

In this section, we firstly review the closely related work including decomposition-based and pruning-based compression methods. Then, we list other categories of network compression works.


2.1. Parameter Pruning for Network Compression

Non-structural pruning. To compress neural networks, network pruning disables the weak connections in a network that have a small influence on its prediction accuracy. Earlier pruning methods explore unstructured network weight pruning by deactivating connections corresponding to small weights or by applying sparsity regularization to the weight parameters. The resulting irregular weight parameters of the network are not implementationfriendly, which hinders the real acceleration rate of the pruned network over the original one.


Figure 2: The flowchart of the proposed algorithm.

Structural pruning. To circumvent the above problem, structural pruning approaches zero out structured groups of he convolutional filters. Specifically, group sparsity regularization has been investigated in recent works for the structural pruning of network parameters.Wen et al. and Alvarez et al. proposed to impose group sparsity regularization on network parameters to reduce the number of feature map channels in each layer. The success of this method triggered the studies of group sparsity based network pruning. Subsequent works improved group sparsity based approaches in different ways. One branch of works combined the group sparsity regularizer with other regularizers for network pruning. A lowrank regularizer as well as an exclusive sparsity regularizer were adopted for improving the pruning performance. Another branch of research investigated a better group-sparsity regularizer for parameter pruning including group ordered weighted `1 regularizer, out-in-channel sparsity regularization and guided attention for sparsity learning. In addition, some works also attempted to achieve group-sparse parameters in an indirect manner. In and, scaling factors were introduced to scale the outputs of specific structures or feature map channels to structurally prune network parameters.

















