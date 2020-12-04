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



The rest of the paper is organized as follows. Sec. 2 discusses the related work. Sec. 3 explains the proposed network compression method. Sec. 4 describes the implementation considerations. The experimental results are shown in Sec. 5. Sec. 6 concludes this paper.




















are obtained by observing the influence of the proximal
gradient method on the filter during the optimization.
