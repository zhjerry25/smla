冻结设计：**固定坐标刷新、MLA 私有 reader、每层完整刷新 RoPE Key**。省略标准 Transformer 的细节。

### 数学定义

设模型有 \(L\) 层，content latent 宽度为 \(c\)，固定刷新宽度为 \(w\)，RoPE 宽度为 \(r\)。\(x_t\) 为位置 \(t\) 的 token，\(h_{\ell,t}\) 为第 \(\ell\) 层输入。

由 token embedding 一次性产生初始 latent：

\[
[u_{1,t};s_t]
=
W_{\mathrm{init}}N_{\mathrm{init}}(E[x_t]),
\qquad
u_{1,t}\in\mathbb R^w,\quad
s_t\in\mathbb R^{c-w}.
\]

其中 \(s_t\) 在所有层保持不变；后续层只覆盖同一组动态坐标：

\[
n_{\ell,t}=N_\ell(h_{\ell,t}),\qquad
u_{\ell,t}=W^U_\ell n_{\ell,t}\quad(\ell\ge2),
\]

\[
\boxed{m_{\ell,t}=[u_{\ell,t};s_t]\in\mathbb R^c.}
\]

每层用独立 reader，**先对完整 latent 做 RMSNorm，再投影**：

\[
\begin{bmatrix}
k^C_{\ell,t}\\v_{\ell,t}
\end{bmatrix}
=
W^{KV}_\ell\,N^{KV}_\ell(m_{\ell,t}).
\]

位置 Key 每层独立生成并完整旋转：

\[
k^R_{\ell,t}=R_tW^R_\ell n_{\ell,t}.
\]

每个 attention head 使用标准 MLA query：

\[
q_{\ell,t}=[q^C_{\ell,t};R_tq^R_{\ell,t}],
\qquad
k_{\ell,t}=[k^C_{\ell,t};k^R_{\ell,t}],
\]

\[
a_{\ell,tj}
=
\operatorname{softmax}_{j\le t}
\left(
\frac{q_{\ell,t}^{\top}k_{\ell,j}}
{\sqrt{d_h+r}}
\right),
\qquad
o_{\ell,t}=\sum_{j\le t}a_{\ell,tj}v_{\ell,j}.
\]

随后接标准输出投影、残差和 FFN。

### 简要伪代码

```python
H = Embedding(tokens)
U, S = split(InitProj(InitNorm(H)), [w, c - w])

for layer in layers:
    X = layer.attn_norm(H)

    if layer.index > 0:
        U = layer.write(X)              # 覆盖固定的 w 个坐标

    M = concat(U, S)
    Kc, V = split(layer.kv_up(layer.kv_norm(M)))
    Kr = RoPE(layer.rope_key(X), positions)
    Q = layer.mla_query(X, positions)

    H = H + layer.out(CausalAttention(Q, concat(Kc, Kr), V))
    H = H + layer.ffn(layer.ffn_norm(H))

logits = LMHead(FinalNorm(H))
```

这里的 `concat` 和展开 K/V 是数学参考实现；实际 decode 可以直接读取分块缓存。

### 共享词表与缓存

因为 \(s_t\) 只依赖 token ID，冻结权重后可预计算：

\[
T_s[v]
=
\operatorname{StaticSlice}
\left(W_{\mathrm{init}}N_{\mathrm{init}}(E[v])\right),
\qquad s_t=T_s[x_t].
\]

因此，每 token 缓存从

\[
\underbrace{(c-w)+Lw+Lr}_{\text{静态底座一次，动态与位置部分逐层}}
\]

变为

\[
\boxed{L(w+r)\ \text{个标量}+\text{token ID},}
\]

另有模型侧共享词表 \(T_s\in\mathbb R^{V\times(c-w)}\)。这一步保持模型计算的数学定义不变。