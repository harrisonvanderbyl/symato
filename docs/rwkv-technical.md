https://github.com/BlinkDL/RWKV-LM

# Công thức toán học
![](files/rwkv-00.png)
__RWKV is like [AFT](./aft.md) with special w_{k, k'}__

![](files/rwkv-04.jpg)
__Triển khai công thức GPT thành công thức RNN__

## Từ GPT tới RWKV
- Gọi `F[t]` là trạng thái hệ thống tại (thời điểm) t.
- Gọi `x[t]` là đầu vào mới tại t.
- Để GPT dự đoán F[t+1] cần cân nhắc F[0],F[1]...F[t] vì thế nó cần O(T^2) để sinh ra một chuỗi có độ dài T.

Công thức đơn giản hóa của GPT là:
`F[t+1] = sum_{i=0}^t exp(Q x[t] mul K F[i]) . (V F[i]) / sum_{i=0}^t exp(Q x[t] mul K F[i])`
Công thức này rất mạnh mẽ về mặt lý thuyết nhưng trên thực tế không có nghĩa là chúng ta có thể tận dụng hết khả năng của nó với những thuật toán tối ưu hóa thông thường. Loss landscape là rất khó với những phương pháp tối ưu đang có. Hay nói cách khác là cơ chế self-attn là phức tạp đối với các optimizers hiện tại.

So sánh với công thức đơn giản hóa của rwkv ở chế độ song song, nó giống như Apple's aft.
`F[t+1] = sigma(R_x[t]) . sum_{i=0}^t exp(W.(t-i)).exp(K F[i]).(V F[i]) / sum_{i=0}^t exp(W.(t-i)).exp(K F[i])`
R,K,V là các ma trận trọng số (có thể huấn luyện được), W là vector trọng số có thể huấn luyện được (phân rã thời gian cho từng channel).

Với GPT, đóng góp của F[t] vào F[t+1] cân đo bằng `exp(Q x[t] mul K F[i])`

với rwkv, đóng góp của F[t] vào F[t+1] cân đo bằng `sigma(R_x[t]) . exp(W.(t-i)).exp(K F[i])`

- `sigma` là hàm phi tuyến tính và ở đây chúng ta dùng hàm sigmoid
- Lưu ý `sigma(R x[t])` không phải là mẫu số mà ta gọi R là "receptance" (sự rung lắc trên từng đơn vị lực tác động)
- `exp(W.(t-i))` là hệ số phân rã theo thời gian (của từng channel). Ý tưởng này giống như scaling the attention by distance được Peng Bo đề xuất 2020 được gọi là "time-weighting" (xem https://github.com/BlinkDL/minGPT-tuned)

## Punchline
Ta có thể viết lại công thức gpt ở trên thành rnn (công thức hồi quy):
- F[1] = sigma(R x[0]) . exp(K F[0]).(V F[0])/exp(K F[0])
- F[2] = sigma(R x[1]) . exp(K F[1]).(V F[1]) + exp(W).exp(K F[0]).(V F[0])/exp(K F[1]) + exp(W).exp(K F[0])
Vì thế có thể dễ dàng thấy:
`F[t+1] = sigma(R x[t+1]) . exp(K F[t]).(V F[t]) + exp(W).A[t] / exp(K F[t]) + exp(W).B[t]` với A[t] và B[t] là tử số và mẫu số của bước t trước đó.

Peng Bo tin rằng rwkv có hiệu năng tốt là nhờ W is like repeatedly applying a diagonal matrix. 
Note `(P^{-1} D P)^n = P^{-1} D^n P`, so it is similar to repeatedly applying a general diagonalizable matrix (ma trận chéo hóa được). Hơn thế nữa nó có thể được biến thành continuous ODE (a bit similar to State Space Models). (TODO: tìm hiểu về diagonal matrix, ODE, State Space Models).

- - -

# How it works?
https://github.com/BlinkDL/RWKV-LM#how-it-works

RWKV is inspired by Apple's AFT (https://arxiv.org/abs/2105.14103).

Moreover it's using a number of my tricks, such as:

* __SmallInitEmb__: https://github.com/BlinkDL/SmallInitEmb (applicable to all transformers) which helps the embedding quality, and stabilizes Post-LN (which is what I am using).

* __Token-shift__: https://github.com/BlinkDL/RWKV-LM#token-shift-time-shift-mixing (applicable to all transformers), especially helpful for char-level models.

* __Head-QK__: https://github.com/BlinkDL/RWKV-LM#the-head-qk-trick-learning-to-copy-and-avoid-tokens (applicable to all transformers). Note: it's helpful, but I disabled it in the Pile model to keep it 100% RNN.

* __Extra R-gate in the FFN__ (applicable to all transformers). I am also using reluSquared from Primer.

* __Better initilization__: I init most of the matrices to ZERO (see RWKV_Init in https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v2-RNN/src/model.py).

* __You can transfer some parameters__ from a small model to a large model (note: I sort & smooth them too), for faster and better convergence (see https://www.reddit.com/r/MachineLearning/comments/umq908/r_rwkvv2rnn_a_parallelizable_rnn_with/).

* __My CUDA kernel__: https://github.com/BlinkDL/RWKV-CUDA to speedup training.

- - -

# pseudocode 
https://github.com/BlinkDL/RWKV-LM#the-pseudocode-execution-from-top-to-bottom

## rwkv v2
- Là RNN nhưng có thể được huấn luyện như GPT transformer
- Chỉ cần {x_t, a_t, b_t} của vị trí t để tính ra vector của t+1
- Nó nhanh hơn và tiết kiệm VRAM hơn GPT 100 lần

![](files/rwkv-05.jpg)
- Ma trận trọng số K: K.shape = (C,C) init to zero
- Ma trận trọng số V: V.shape = (C,C)
- Ma trận trọng số R: R.shape = (C,C) init to zero
- x_t là đầu vào của tầng n tại vị trí t, x_t.shape = (C)
- Token-shift Ts init to (1,1,1,...,0,0,0), Ts.shape = (C) (C//2 values đầu là 1, C//2 values sau là 0)
- `z_t = Ts*x_{t-1} + (1 - Ts)*x_t` trộn x_t với token x_{t-1} 
- `k_t =     exp(K @ z_t)`
- `v_t =        (V @ z_t)`
- `r_t = sigmoid(R @ z_t)`
- Tích lũy theo thời gian của kv và k
  - W: vector hệ số phân rã của từng channel, W.shape = (C)
  - X: vector hệ số self-attn của từng channel, X.shape = (C)
- `a_0 = W*0 + k_0*v_0`
- `b_0 = W*0 + k_0`
- `c_t = a_{t-1} + X*k_t * v_t` tử số (numerator)
- `d_t = b_{t_1} + X*k_t`      mẫu số (denominator)
- `a_t = W*a_{t-1} + k_t*v_t`
- `b_t = W*b_{t-1} + k_t
- `y_t = r_t * c_t / d_t`
- P: ouput projection, p.shape = (C, C), init to zero
- Self-attn output: `out_t = x_t + P @ (y_t)`

Notes:
- Các hệ số a,b,c,d làm việc cùng nhau để xây dựng một đường cong phân rã theo thời gian 
[X, 1, W, W^2, W^3, ...]
- a và b là EMAs của kv và k
- c và d là a và b kết hợp với "self-attn"
- kv / k là cơ chế ghi nhớ. Token với giá trị k cao sẽ được ghi nhớ lâu hơn nếu W gần với 1 trong channel đang xét.
- R-gate là quan trọng với hiệu năng.
  - k = sức mạnh thông tin của token đang xét, sẽ được chuyển tiếp tới các tokens trong tương lai.
  - r = liệu có áp dụng thông tin vào token đang xét hay không

## The GPT mode - FFN block
The FFN block has three tricks comparing with the usual GPT:

1. My time_mix trick.
2. The [sqReLU](./sqrelu.md) from the Primer paper.
3. An extra receptance-gate (similar to the receptance-gate in ATT block).

```py
# Mix x with the previous timestep to produce xk, xr
xx = self.time_shift(x)
xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

# The usual FFN operation
k = self.key(xk)
k = torch.square(torch.relu(k)) # from the Primer paper
kv = self.value(k)

# Apply an extra receptance-gate to kv
rkv = torch.sigmoid(self.receptance(xr)) * kv
return rkv
```