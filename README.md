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


2.2. Filter Decomposition for Network Compression

Another category of works compresses network parameters through tensor decomposition. Specifically, the original filter is decomposed into a lightweight one and a linear projection which contain much fewer parameters than the original, thus resulting in the reduction of parameters and computations. Early works apply matrix decomposition methods such as SVD or CP-decomposition to decompose 2D filters. In, Jaderberg et al. proposed to approximate the 2D filter set by a linear combination of a smaller basis set of 2D separable filters. Subsequent filter basis decomposition works polished the approach in by using a shared filter basis or by enabling more flexible filter decomposition. In addition, Zhang et al. took the input channel as the third dimension and directly compress the 3D parameter tensor.


2.3. Other methods

Other network compression methods include network quatization and knowledge distillation. Network quantization aims at a low-bit representation of network parameters to save storage and to accelerate inference. This method does not change the architecture of the fully-fledged network. Knowledge distillation transfers the knowledge of a teacher network to a student network. Current research in this direction focuses on the architectural design of the student network and the loss function.

3. The proposed method

This section explains the proposed method (Fig. 2). Specifically, it describes how group sparsity can hinge filter pruning and decomposition. The pair {x, y} denotes the input and target of the network. Without loss of clarity, we also use x to denote the input feature map of a layer. The output feature map of a layer is denoted by z. The filters of a convolutional layer are denoted by W while the introduced group sparsity matrix is denoted by A. The rows and columns of A are denoted by Ai, and Aj, respectively. The general structured groups of A are denoted by Ag.

3.1. Group sparsity

The convolution between the input feature map x and the filters can be converted to a matrix multiplication, i.e., where X ∈ RN×cwh, W ∈ Rcwh×n, and Z ∈ RN×n are the reshaped input feature map, output feature map, and convolutional filter, c, n, w × h, and N denotes the input channel, number of filters, filter size, and number of reshaped features, respectively. For the sake of brevity, the bias term is omitted here. The weight parameters W are usually trained with some regularization such as weight decay to avoid overfitting the network. To get structured pruning of the filter, structured sparsity regularization is used to constrain the filter, i.e. where D(·) and R(·) are the weight decay and sparsity regularization, µ and λ are the regularization factors.

Different from other group sparsity methods that directly regularize the matrix W, we enforce group sparsity constraints by incorporating a sparsity-inducing matrix A ∈ R
n×n, which can be converted to the filter of a 1 × 1 convolutional layer after the original layer. Then the original convolution in Eqn. 1 becomes Z = X × (W × A). To obtain a structured sparse matrix, group sparsity regularization is enforced on A. Thus, the loss function becomes Solving the problem in Eqn. 3 results in structured group sparsity in matrix A. By considering matrix W and A together, the actual effect is that the original convolutional filter is compressed.

In comparison with the filter selection method, the proposed method not only selects the filters in a layer, but also makes linear combinations of the filters to minimize the error between the original and the compact filter. On the other hand, different from other group sparsity constraints, there is no need to change the original filters W of the network too much during optimization of the sparsity problem. In our experiments, we set a much smaller learning rate for the pretrained weight W.

3.2. The hinge

The group sparsity term in Eqn. 3 controls how the network is compressed. This term has the form where Ag denotes the different groups of A, kAgk2 is the `2 norm of the group, and Φ(·) is a function of the group `2 norms.

If group sparsity regularization is added to the columns of A as in Fig. 3a, i.e., R(A) = Φ(kAjk2), a column pruned version Ac is obtained and the output channels of the corresponding 1 × 1 convolution are pruned. In this case, we can multiply W and Ac and use the result as the filter of the convolutional layer. This is equivalent to pruning the output channels of the convolutional layer with the filter W.

On the other hand, group sparsity can be also applied to the rows of A, i.e. R(A) = Φ(kAik2). In this case, a row-sparse matrix Ar is derived and the input channels of the 1 × 1 convolution can be pruned (See Fig. 3b). Accordingly, the corresponding output channels of the former convolution with filter W can be also pruned. However, since the output channel of the later convolution is not changed, multiplying out the two compression matrices do not save any computation. So a better choice is to leave them as two separate convolutional layers. This tensor manipulation method is equivalent to filter decomposition where a single convolution is decomposed into a lightweight one and a linear combination. In conclusion, by enforcing group sparsity to the columns and rows of the introduced matrix A, we can derive two tensor manipulation methods that are equivalent to the operation of filter pruning and decomposition, respectively. This provides a degree of freedom to choose the tensor manipulation method depending on the specifics of the underlying network.

3.3. Proximal gradient solver

To solve the problem defined by Eqn. 3, the parameter W can be updated with stochastic gradient descent(SGD) but with a small learning rate, i.e. Wt+1 = Wt − ηs∇G(Wt), where G(Wt) = L(· , f(· ;Wt, ·)) + µD(Wt). This is because it is not desired to modify the pretrained parameters too much during the optimization phase. The focus should be the sparsity matrix A.

The proximal gradient algorithm is used to optimize the matrix A in Eqn. 3. It consists of two steps, i.e. the gradient descent step and the proximal step. The parameters in A are first updated by SGD with the gradient of the loss function H(A) = L(y, f(x;W, A)), namely, where η is the learning rate and η >> ηs. Then the proximal operator chooses a neighborhood point of At+∆ that minimizes the group sparsity regularization, i.e.

The sparsity regularizer Φ(·) can have different forms, e.g., `1 norm, `1/2 norm, `1−2 norm, and logsum. All of them try to approximate the `0 norm. In this paper, we mainly use {`p : p = 1, 1/2} regularizers while we also include the `1−2 and logsum regularizers in the ablation studies. The proximal operators of the four regularizers have a closed-form solution. Briefly, the solution is the soft-thresholding operator for p = 1 and the halfthresholding operator for p = 1/2. The solutions are appended in the supplementary material. The gradient step and proximal step are interleaved in the optimization phase of the regularized loss until some predefined stopping criterion is achieved. After each epoch, groups with `2 norms smaller than a predefined threshold T are nullified. And the compression ratio in terms of FLOPs is calculated. When the difference between the current and the target compression ratio γc and γ∗ is lower than the stopping criterion α, the compression phase stops. The detailed compression algorithm that utilizes the proximal gradient is shown in Algorithm 1.

3.4. Binary search of the nullifying threshold 

After the compression phase stops, the resulting compression ratio is not exactly the same as the target compression ratio. To fit the target compression ratio, we use a binary search algorithm to determine the nullifying threshold T . The compression ratio γ is actually a monotonous function of the threshold T , i.e. γ = g(T ). However, the explicit expression of the function g(·) is not known. Given a target compression threshold γ∗, we want to derive the threshold used to nullifying the sparse groups, i.e. T∗ = g−1(γ∗), where g−1(·) is the inverse function of g(·). The binary search approach shown in Algorithm 2 starts with an initial threshold T0 and a step s. It adjusts the threshold T according to the values of the current and target compression ratio. The step s is halved if the target compression ratio sits between the previous one γn−1 and the current one γn. The searching procedure stops if final compression ratio γn is closed enough to the target, i.e., |γn − γ∗| ≤ C.

3.5. Gradient based adjustment of learning rate

In the ResNet basic block, both of the two 3 × 3 convolutional layers are attached a sparsity-inducing matrix A1 and A2, namely, 1×1 convolutional layers. We empirically find that the gradient of the first sparsity-inducing matrix is larger than that of the second. Thus, it is easier for the first matrix to jump to a point with larger average group `2 norms. This results in unbalanced compression of the two sparsity-inducing matrices since the same nullifying threshold is used for all of the layers. That is, much more channels of A2 are compressed than that of the first one. This is not a desired compression approach since both of the two layers are equally important. A balanced compression between them is preferred.
To solve this problem, we adjust the learning rate of the first and second sparsity-inducing matrices according to their gradients. Let the ratio of the average group `2 norm between the gradients of the matrices be Then the learning rate of the first convolution is divided byρm. We empirically set m = 1.35

3.6. Group `2 norm based layer balancing

The proximal gradient method depends highly on the group sparsity term. That is, if the initial `2 norm of a group is small, then it is highly likely that this group will be nullified. The problem is that the distribution of the group `2 norm across different layers varies a lot, which can result in quite unbalanced compression of the layers. In this case, a quite narrow bottleneck could appear in the compressed network and would hamper the performance. To solve this problem, we use the mean of the group `2 norm of a layer to recalibrate the regularization factor of the layer. That is, where λl is the regularization factor of the l-th layer. In this way, the layers with larger average group `2 norm get a larger punishment.

3.7. Regularization factor annealing

The compression procedure starts with a fixed regularization factor. However, towards the end of the compression phase, the fixed regularization factor may be so large that more than the desired groups are nullified in an epoch. Thus, to solve the problem, we anneal the regularization factor when the average group `2 norm shrinks below some threshold. The annealing also impacts the proximal step but has less influence while the gradient step plays a more active role in finding the local minimum.

3.8. Distillation loss in the finetuning phase

In Eqn. 3, the prediction loss and the groups sparsity regularization are used to solve the compression problem. After the compression phase, the derived model is further finetuned. During this phase, a distillation loss is exploited to force similar logit outputs of the original network and the pruned one. The vanilla distillation loss is used, i.e. where Lce(·) denotes the cross-entropy loss, σ(·) is the softmax function, zc and zo are the logit outputs of the compressed and the original network. For the sake of simplicity, the network parameters are omitted. We use a fixed balancing factor α = 0.4 and temperature T = 4.

4. Implementation Considerations ( 구현 고려 사항 )

4.1. Sparsity-inducing matrix in network blocks

In the analysis of Sec. 3, a 1×1 convolutional layer with the sparsity-inducing matrix is appended after the uncompressed layer. When it comes to different network blocks, we tweak it a little bit. As stated, both of the 3 × 3 convolutions in the ResNet basic block are appended with a 1 × 1 convolution. For the first sparsity-inducing matrix, group sparsity regularization can be enforced on either the columns or the rows of the matrix. As for the second matrix, group sparsity can be enforced on its rows due to the existence of the skip connection.
The ResNet and ResNeXt bottleneck block has the structure of 1 × 1 → 3 × 3 → 1 × 1 convolutions.
Here, the natural choice of sparsity-inducing matrices are the leading and the ending convolutions. For the ResNet bottleneck block, the two matrices select the input and output channels of the middle 3 × 3 convolution, respectively. Things become a little bit different for the ResNeXt bottleneck since the middle 3×3 convolution is a group convolution. So the aim becomes enforcing sparsity on the already existing groups of the group convolution. In order to do that, the parameters related to the groups in the two sparsityinducing matrices are concatenated. Then group sparsity is enforced on the new matrix. After the compression phase, a whole group can be nullified.

4.2. Initialization of W and A

For the ResNet and ResNeXt bottleneck block, 1 × 1 convolutions are already there. So the original network parameters are used directly. However, it is necessary to initialize the newly added sparsity-inducing matrix A. Two initialization methods are tried. The first one initializes W and A with the pretrained parameters and identity matrix, respectively. The second method first calculates the singular value decomposition of W, i.e. W = USVT . Then the left eigenvector U and the matrix SVT are used to initialize W and A. Note that the singular values are annexed by the right eigenvector. Thus, the columns of W, i.e. the filters of the convolutional layer ly on the surface of the unit sphere in the high-dimensional space.

5. Experimental Results

In this section, the proposed method is validated on three image classification datasets including CIFAR10, CIFAR100, and ImageNet2012. The network compression method is applied to ResNet, ResNeXt, VGG, and DenseNet on CIFAR10 and CIFAR100, WRN on CIFAR100, and ResNet50 on ImageNet2012. For ResNet20 and ResNet56 on CIFAR dataset, the residual block is the basic ResBlock with two 3 × 3 convolutional layers. For ResNet164 on CIFAR and ResNet50 on ImageNet, the residual block is a bottleneck block. The investigated models of ResNeXt are ResNeXt20 and ResNeXt164 with carlinality 32, and bottleneck width 1. WRN has 16 convolutional layers with widening factor 10.

The training protocol of the original network is as follows. The networks are trained for 300 epochs with SGD on CIFAR dataset. The momentum is 0.9 and the weight decay factor is 10−4. Batch size is 64. The learning rate starts with 0.1 and decays by 10 at Epoch 150 and 225. The ResNet50 model is loaded from the pretrained PyTorch model. The models are trained with Nvidia Titan Xp GPUs. The proposed network compression method is implemented by PyTorch. We fix the hyper parameters of the proposed method by empirical studies. The stop criterion α in Algorithm 1 is set to 0.1. The threshold T is set to 0.005.

5.1. Results on CIFAR10

The experimental results on CIFAR10 are shown in Table. 1. The Top-1 error rate, the percentage of the remaining FLOPs and parameters of the compressed models are listed in the table. For ResNet56, two operating points are reported. The operating point of 50% FLOP compression is investigated by a bunch of state-of-the-art compression methods. Our proposed method achieves the best performance under this constraint. At the compression ratio of 24%, our approach is clearly better than KSE. For ResNet and ResNeXt with 20 and 164 layers, our method shoots a lower error rate than SSS. For VGG and DenseNet, the proposed method reduces the Top-1 error rate by 0.41 % and 1.51 % compared with. In Fig. 4, we compare the FLOPs and number of parameters of the compressed model by KSE and the proposed method under different compression ratios. As shown in the figure, our compression method outperforms KSE easily. Fig. 10a and 10b show more comparison between SSS and our methods on CIFAR10. Our method forms a lower bound for SSS.






