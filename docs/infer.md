https://lilianweng.github.io/posts/2023-01-10-inference-optimization

# Tại sao llm lại khó suy diễn?

1. __Tốn bộ nhớ__: Vì toàn bộ tham số của mô hình và trạng thái trung gian cần được lưu trong bộ nhớ
  - kv cần được lưu trong bộ nhớ trong quá trình decode, ví dụ với batch_size = 512, ctx_len = 2048, kv cache chiếm 3T bộ nhớ, gấp 3 lần kích thước mô hình!
  - Chi phí suy diễn tốn O(n^2) với n là chiều dài của chuỗi.
2. __Khả năng song song hóa thấp__ quá trình suy diễn để sinh ra token mới là tự hồi quy nên khiến nó rất khó để song song hóa.

# Các phương pháp tối ưu

## Distillation
Knowledge Distillation (KD; Hinton et al. 2015, Gou et al. 2020) is a straightforward way to build a smaller, cheaper model (“student model”) to speed up inference by transferring skills from a pre-trained expensive model (“teacher model”) into the student. There is no much restriction on how the student architecture should be constructed, except for a matched output space with the teacher in order to construct a proper learning objective.
![](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/distillation.png)

A successful early trial is DistilBERT (Sanh et al. 2019) that is able to reduce the parameters of a BERT by 40% while maintaining 97% performance of BERT on fine-tuned downstream tasks and running 71% faster.

Distillation can be easily combined with quantization, pruning or sparsification techniques, where the teacher model is the original full-precision, dense model and the student is quantized, pruned, or trimmed to have higher sparsity level.

## Quantization
### Challenges for Transformer Quantization
Many studies on Transformer model quantization have the same observation: A simple low-precision (e.g. 8-bit) post-training quantization leads to significant performance drop mainly due to the high dynamic ranges of activation and a naive activation quantization strategy fails to maintain the capacity.
![](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/quantization-experiment-table.png)

Only quantizing model weights to 8-bit while keeping activation at full precision (`W8A32`) achieves much better results when activations are quantized to 8-bit irrespective of whether weights are in lower precision (`W8A8` and `W32A8`)

As the model size continues to grow to billions of parameters, outlier features of high magnitude start to emerge in all transformer layers, causing failure of simple low-bit quantization. Dettmers et al. (2022) observed such a phenomenon for OPT models larger than 6.7B parameters. Larger models have more layers with extreme outliers and these outlier features have a significant impact on the model performance. The scale of activation outliers in a few dimensions can be ~100× larger than most of the other values.

### Post-training quantization (PTQ)
#### Mixed-precision quantization
![](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/LLM-int8.png)

### Second order information for quantization
### Outlier smoothing
![](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/SmoothQuant.png)

## Pruning
Network pruning is to reduce the model size by trimming unimportant model weights or connections while the model capacity remains. It may or may not require re-training. Pruning can be unstructured or structured.

Magnitude pruning is simplest yet quite effective pruning method - weights with smallest absolute values are trimmed. In fact, some studies (Gale et al. 2019) found that simple magnitude pruning approaches can achieve comparable or better results than complicated pruning methods, such as variational dropout (Molchanov et al. 2017) and  regularization (Louizos et al. 2017). Magnitude pruning is simple to apply to large models and achieves reasonably consistent performance across a wide range of hyperparameters.

Iterative pruning (Renda et al. 2020) iterates step 2 (prune) & step 3 (retrain) multiple times: Only a small fraction of weights are pruned and the model is retrained in each iteration. The process repeats until a desired sparsity level is reached.

Lottery Ticket Hypothesis proposed a weight rewinding retraining technique: After pruning, the unpruned weights are reinitialized back to original values earlier in the training and then retrain with the same learning rate schedule.

Learning rate rewinding (Renda et al. 2020) only resets the learning rate back to its early value, while the unpruned weights stay unchanged since the end of the last train stage. They observed that (1) retraining with weight rewinding outperforms retraining with fine-tuning across networks and datasets and (2) learning rate rewinding matches or outperforms weight rewinding in all tested scenarios.

## Sparsity
### N:M Sparsity via Pruning
N:M sparsity is a structured sparsity pattern that works well with modern GPU hardware optimization, in which  out of every  consecutive elements are zeros. For example, the sparse tensor core of Nvidia A100 GPU has support for 2:4 sparsity for faster inference (Nvidia 2020).
![](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/2-to-4-sparsity.png)

To sparsify a dense neural network to follow a N:M structured sparsity pattern, Nvidia (2020) suggested using the three-step routine workflow for training a pruned network: train –> prune to satisfy 2:4 sparsity –> retrain.

https://lilianweng.github.io/posts/2023-01-10-inference-optimization/#nm-sparsity-via-pruning

## Sparsified Transformer
Scaling Transformer (Jaszczur et al. 2021) sparsifies both self-attention and FFN layers in transformer architecture, achieving __37x speedup__ for single-example inference.

Sparse FFN layer: Each FFN layer contains 2 MLP and one ReLU in-between. Because ReLU will introduce a lot of zeros, they implement a fixed structure on activations to enforce only 1 non-zero value in one block of N elements. The sparsity pattern is dynamic, different for each token.
![](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/sparse-FFN.png)

## Mixture-of-Experts
Mixture-of-experts (MoE) models depend on a collection of “expert” networks and each example only activates a subset of networks to get predictions. The idea originated back to the 1990s (Jacobs et al. 1991) and is strongly related to ensemble methods. For details on how to incorporate MoE module into transformer, please check my previous post on large model training techniques and a survey paper on MoE by Fedus et al. 2022.

With MoE architecture, only partial parameters are utilized at decoding time and therefore it saves inference cost. 

### Routing Strategy Improvement
MoE layer has a routing network to assign a subset of experts for each input token. The routing strategy in vanilla MoE models is to route each token toward preferred experts differently as they come up in the natural order. If a token is routed to experts that have reached their capacity, the token would be marked “overflowed” and skipped.

https://lilianweng.github.io/posts/2023-01-10-inference-optimization/#mixture-of-experts

## Architectural Optimization
![](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/efficient-transformer.png)

rwkv, h3 thuộc dạng này.