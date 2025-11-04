
# LLM Inference Prep Guide
https://swamp-magnolia-c4c.notion.site/LLM-Inference-Prep-Guide-1f736157c38280278fd5dc6724106e7e?pvs=143

[LLM Fundamentals](https://www.notion.so/LLM-Fundamentals-1f736157c38280c49170fd4e7a655e3a?pvs=21)

[Pytorch Coding Round](https://www.notion.so/Pytorch-Coding-Round-1f736157c382802f8461caf634f32969?pvs=21)

[ML Inference System Design](https://www.notion.so/ML-Inference-System-Design-20f36157c3828072a3dece5fbcb9ae71?pvs=21)

— by VS Chandra Mourya (https://www.youtube.com/@vsmourya)


# LLM Fundamentals

**High-Level Overview**

**Training Optimizations**

**Reducing Latency**

1.	**Mixed Precision Training**: Reduces memory usage and speeds computation.

2.	**Optimizer Enhancements (e.g., AdamW, LAMB)**: Accelerates convergence.

3.	**FlashAttention**: Optimizes attention computation for speed and memory.

4.	**Cocktail SGD**: Reduces network overhead in distributed training.

5.	**Sub-Quadratic Architectures (e.g., Striped Hyena)**: Lowers computational complexity.

**Higher Throughput**

1.	**Gradient Accumulation**: Enables large batch sizes on limited GPU memory.

2.	**Data Parallelism**: Splits data across GPUs for faster training.

3.	**Model Parallelism**: Distributes model layers across GPUs.

4.	**Pipeline Parallelism**: Pipelines model layers across GPUs for efficiency.

5.	**Gradient Checkpointing**: Trades computation for memory savings.

6.	**LoRA Optimization**: Efficient fine-tuning for large models.

---

**Inference Optimizations**

**Reducing Latency**

1.	**Layer Fusion**: Combines operations into single kernels.

2.	**Speculative Decoding**: Uses draft models to predict tokens faster.

3.	**SplitRPC**: Splits control/data paths for reduced latency.

4.	**FlashAttention-3**: Enhances inference speed for long sequences.

5.	**Custom CUDA Kernels**: Optimizes specific ops (e.g., softmax).

**Higher Throughput**

1.	**Quantization**: Reduces precision (e.g., FP16, INT8) for faster inference.

2.	**Dynamic Batching**: Groups requests for GPU efficiency.

3.	**Continuous Batching**: Processes requests as they arrive.

4.	**Caching (e.g., KV Cache)**: Reuses computed values.

5.	**Knowledge Distillation**: Trains smaller, faster models.

## High-Level Overview

### Training Optimizations

1. **Mixed Precision Training**: Reduces memory usage and speeds computation.
2. **Gradient Accumulation**: Enables large batch sizes on limited GPU memory.
3. **Data Parallelism**: Splits data across GPUs for faster training.
4. **Model Parallelism**: Distributes model layers across GPUs.
5. **Pipeline Parallelism**: Pipelines model layers across GPUs for efficiency.
6. **Gradient Checkpointing**: Trades computation for memory savings.
7. **Optimizer Enhancements (e.g., AdamW, LAMB)**: Accelerates convergence.
8. **FlashAttention**: Optimizes attention computation for speed and memory.
9. **Cocktail SGD**: Reduces network overhead in distributed training.
10. **Sub-Quadratic Architectures (e.g., Striped Hyena)**: Lowers computational complexity.
11. **LoRA optimization** 

### Inference Optimizations

1. **Quantization**: Reduces precision (e.g., FP16, INT8) for faster inference.
2. **Layer Fusion**: Combines operations into single kernels.
3. **Dynamic Batching**: Groups requests for GPU efficiency.
4. **Speculative Decoding**: Uses draft models to predict tokens faster.
5. **Continuous Batching**: Processes requests as they arrive.
6. **Caching (e.g., KV Cache)**: Reuses computed values.
7. **Knowledge Distillation**: Trains smaller, faster models.
8. **Custom CUDA Kernels**: Optimizes specific ops (e.g., softmax).
9. **SplitRPC**: Splits control/data paths for reduced latency.
10. **FlashAttention-3**: Enhances inference speed for long sequences.

---

## 1. Core Neural Network & Deep Learning Fundamentals

- **Forward Pass vs. Backward Pass (Backpropagation)**
    - Understanding how gradients are computed (chain rule, layer-by-layer propagation).
    - Difference between training (forward + backward) and inference (forward-only).
    - Relevance for inference: parameters are fixed, but knowledge of backprop cements how models are trained.
- **Common Activation & Loss Functions**
    - ReLU, Sigmoid, Softmax, etc.
    - Role of loss functions in model training (though not directly used in inference).

---

## 2. GPU Programming & CUDA Foundations

- **GPU Architecture (NVIDIA Focus)**
    - Organization of threads into warps, blocks, and grids.
    - How warps operate in lock-step and the impact of warp divergence on performance.
- **CUDA Memory Hierarchy**
    - Global memory vs. shared memory vs. registers vs. caches.
    - Techniques for **memory coalescing** and shared-memory tiling to reduce global memory bandwidth usage.
    - Asynchronous data transfers and concurrent kernel execution.
- **CUDA Kernel Optimization for Inference**
    - Kernel fusion to reduce memory round-trips.
    - Minimizing memory transfers by carefully orchestrating host–device communication.
    - Keeping GPUs saturated (maximizing occupancy, avoiding warp divergence, etc.).

---

## 3. Transformer & Large Language Model (LLM) Architecture

- **Transformer Basics**
    - Encoder vs. decoder vs. decoder-only structures.
    - Self-attention mechanism and **scaled dot-product attention**.
    - Multi-head attention rationale (parallel heads, capturing different representations).
- **Attention Mechanism Details**
    - Queries, keys, values: how they are computed and combined.
    - O(n²) complexity in naive self-attention and how this impacts inference speed.
    - Key-value caching in autoregressive decoding to avoid recomputing past tokens’ attention.
- **LLM Training Pipeline & RLHF**
    - High-level overview of pre-training, fine-tuning, and RLHF.
    - Why LLMs are typically pretrained on massive corpora and then specialized or aligned.

---

## 4. Model Compression & Parameter-Efficient Techniques

- **Quantization**
    - Floating-point (FP16, BF16) vs. integer (INT8/INT4) representations.
    - Post-training quantization (PTQ) vs. quantization-aware training (QAT).
    - Trade-offs between accuracy, memory savings, and speed.
- **Knowledge Distillation**
    - Concept: training a smaller “student” model to mimic a larger “teacher” model.
    - How this can drastically reduce model size and inference cost.
- **Pruning (Structured & Unstructured)**
    - Removing redundant weights or neurons for sparser models.
    - Relevance for LLMs (e.g., SparseGPT).
    - Hardware considerations: sparse kernels are not always fully efficient on GPUs.
- **LoRA (Low-Rank Adaptation)**
    - Freezing the original model and training low-rank updates for fine-tuning.
    - Why LoRA adds negligible overhead in inference (low-rank matrices can be merged or applied with minimal extra cost).

---

## 5. LLM Inference Frameworks & Optimization Techniques

- **High-Throughput Inference Strategies**
    - Challenges: large memory footprint, sequential token generation, concurrent requests.
    - Batching approaches: static vs. continuous/dynamic batching.
- **vLLM (PagedAttention & Continuous Batching)**
    - PagedAttention for more efficient key/value caching.
    - Continuous batching to merge incoming requests on the fly for better GPU utilization.
- **NVIDIA TensorRT & TensorRT-LLM**
    - Overview of TensorRT graph optimizations, kernel fusion, quantization support.
    - TensorRT-LLM specifics: support for 8-bit & 4-bit quantization, multi-GPU scaling, fast attention kernels (FlashAttention-style).
    - Building optimized inference “engines” from trained LLMs.
- **Other Inference Frameworks (awareness)**
    - HF Text Generation Inference (TGI), DeepSpeed-Inference, FasterTransformer, Triton Inference Server.
    - High-level idea that all aim for minimized latency and maximized throughput via specialized optimizations.

---

## 6. Decoding & Speculative Decoding

- **Standard Decoding Methods**
    - Greedy, beam search, top-k, nucleus sampling—trade-offs for speed, diversity, and quality.
- **Speculative Decoding**
    - Core concept: using a smaller “draft” model to predict multiple tokens at once, then verifying with the larger “target” model.
    - ~2–3× speedup in throughput with zero quality loss if the draft model aligns well with the main model.
    - Implementation details in NVIDIA TensorRT-LLM and Google’s research (2022).

---

## 7. Systems Integration & End-to-End Inference Knowledge

- **Putting It All Together**
    - Memory management, scheduling (continuous batching), model optimization (quantization, LoRA, etc.), and decoding strategies (speculative decoding) to achieve high-performing LLM inference.
    - Trade-offs: speed vs. accuracy, throughput vs. latency, memory footprint vs. user concurrency.
- **Interview Q&A Readiness**
    - Be prepared to explain the “why” behind each optimization (e.g., “Why quantize?”, “Why use LoRA instead of full fine-tuning?”, “Why does GPU shared memory speed things up?”, “How does speculative decoding preserve exact distribution?”).
    - Cite relevant research or frameworks to demonstrate cutting-edge familiarity.

---

### Final Note

Mastering these topics—from the low-level (CUDA, memory hierarchy) to the high-level (Transformer architecture, inference frameworks, decoding strategies)—will position you to speak confidently about modern LLM inference pipelines. These skills collectively show that you can optimize and serve large models effectively in a production environment.

**Video references:** 

https://youtu.be/9tvJ_GYJA-o?si=dE2gwQm2bCfmGn7R

https://youtu.be/wjZofJX0v4M?si=ZS7RHTt0Q8pihsJc

https://youtu.be/KuXjwB4LzSA?si=9KPrv2GFHJ1d3UYo

https://youtu.be/eMlx5fFNoYc?si=UrCt9Xri3YuIfHUW

https://youtu.be/9-Jl0dxWQs8?si=nbg8_RbTRfIBY1AE

https://youtu.be/q8SA3rM6ckI?si=XVA7dlNItqc5Q8MH

https://youtu.be/7xTGNNLPyMI?si=oBpLaDegRkAOjZIu

https://youtu.be/UcwDgsMgTu4?si=P1uBXLf7SLsTrOYT

https://youtu.be/cXpTDKjjKZE?si=-JetWKIvWH1A9iQS

## Keywords to dig into:

Time for first token

Grouped query attention

Multi-query attention

Multi-latent attention

DeepSeek innovation

Groups all the KVs from different heads into one latent vector which is expanded in inference time

Quantization 

Paged Attention

Multi-head latent attention

Continuous batching

Sliding Window Attention

Cache warming

KV Cache preloading

Prefetch input data

CUDA Graphs

CUDA Streams

3 Things that matter in building LLMs:

Latency (**s/tokens**)

Throughput (**Query/s**)

Cost

IMP: Size of your **Model** and the **KV Cache** … in Memory …. limits your batch size




# Pytorch Coding Round

### Pytorch Questions

[Practice 1](https://www.notion.so/Practice-1-1f736157c382807fab2af0d1f32ea733?pvs=21)

[Practice 2](https://www.notion.so/Practice-2-20f36157c382805b9f22ee3c983ac33b?pvs=21)

LLM specific

[LLM specific questions](https://www.notion.so/LLM-specific-questions-20f36157c3828090baa0e5faa3abfd24?pvs=21)

- GPT2 Code (Imp to Understand)
    
    ```python
    import torch
    import torch.nn as nn
    
    class CausalSelfAttention(nn.Module):
        """
        Multi-Head Causal Self-Attention:
          - Splits embedding into multiple heads
          - Computes scaled dot-product attention
          - Applies a causal mask so tokens cannot attend to future positions
        """
        def __init__(self, embed_size, num_heads, dropout=0.1):
            super().__init__()
            assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
            self.embed_size = embed_size
            self.num_heads = num_heads
            self.head_dim = embed_size // num_heads
    
            # Transform from embed -> queries, keys, values
            self.q_proj = nn.Linear(embed_size, embed_size)
            self.k_proj = nn.Linear(embed_size, embed_size)
            self.v_proj = nn.Linear(embed_size, embed_size)
    
            # Output projection back to embed dimension
            self.out_proj = nn.Linear(embed_size, embed_size)
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            """
            x: (batch_size, seq_len, embed_size)
            """
            bsz, seq_len, _ = x.size()
    
            # Compute Q, K, V
            q = self.q_proj(x)  # (bsz, seq_len, embed_size)
            k = self.k_proj(x)
            v = self.v_proj(x)
    
            # Reshape to (bsz, seq_len, num_heads, head_dim) then transpose to
            # (bsz, num_heads, seq_len, head_dim) for multi-head attention
            q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
            # Scale queries
            q = q / (self.head_dim ** 0.5)
    
            # Compute attention scores: (bsz, num_heads, seq_len, seq_len)
            att_scores = torch.matmul(q, k.transpose(-2, -1))
    
            # Causal mask: restrict attention to current + previous positions
            causal_mask = torch.ones((seq_len, seq_len), device=x.device).tril()
            # shape (seq_len, seq_len), 1.0 in lower-triangular, 0.0 elsewhere
            # We expand to [1, 1, seq_len, seq_len] for broadcasting
            att_scores = att_scores.masked_fill(causal_mask == 0, float('-inf'))
    
            # Attention weights
            att_weights = torch.softmax(att_scores, dim=-1)
            att_weights = self.dropout(att_weights)
    
            # Weighted sum of values
            out = torch.matmul(att_weights, v)  # (bsz, num_heads, seq_len, head_dim)
    
            # Recombine heads
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_size)
    
            # Final linear projection
            out = self.out_proj(out)
            return out
    
    class FeedForward(nn.Module):
        """
        Position-wise Feed Forward layer, typically:
          - Linear (embed_size) -> (4*embed_size)
          - Activation (GELU or ReLU)
          - Linear (4*embed_size) -> (embed_size)
        """
        def __init__(self, embed_size, expansion_factor=4, dropout=0.1):
            super().__init__()
            inner_dim = expansion_factor * embed_size
            self.net = nn.Sequential(
                nn.Linear(embed_size, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, embed_size),
                nn.Dropout(dropout),
            )
    
        def forward(self, x):
            return self.net(x)
    
    class GPTBlock(nn.Module):
        """
        Single block of GPT-style transformer:
          - LayerNorm
          - Causal Self-Attention + skip
          - LayerNorm
          - FeedForward + skip
        """
        def __init__(self, embed_size, num_heads, expansion_factor=4, dropout=0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_size)
            self.attn = CausalSelfAttention(embed_size, num_heads, dropout)
            self.ln2 = nn.LayerNorm(embed_size)
            self.ff = FeedForward(embed_size, expansion_factor, dropout)
    
        def forward(self, x):
            # Causal Self-Attention sub-layer
            attn_out = self.attn(self.ln1(x))  # apply LN first
            x = x + attn_out  # residual connection
    
            # Feed Forward sub-layer
            ff_out = self.ff(self.ln2(x))
            x = x + ff_out  # residual connection
    
            return x
    
    class GPTModel(nn.Module):
        """
        Decoder-Only Transformer, GPT-style.
    
        Typical GPT-3 scale settings (for reference, not shown in code default):
          - vocab_size ~ 50k
          - embed_size ~ 12,288
          - num_heads ~ 96
          - n_layers ~ 96
          - block_size (context length) ~ 2048
        """
        def __init__(
            self,
            vocab_size,
            block_size,       # maximum sequence length
            embed_size=768,   # smaller default for demonstration
            num_heads=12,     # smaller default
            num_layers=12,    # smaller default
            expansion_factor=4,
            dropout=0.1,
            pad_id=0
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.block_size = block_size
            self.embed_size = embed_size
            self.pad_id = pad_id
    
            # Token + Position embeddings
            self.token_emb = nn.Embedding(vocab_size, embed_size)
            self.pos_emb = nn.Embedding(block_size, embed_size)
    
            # Transformer blocks
            self.blocks = nn.ModuleList([
                GPTBlock(embed_size, num_heads, expansion_factor, dropout)
                for _ in range(num_layers)
            ])
    
            # Final layer norm
            self.ln_f = nn.LayerNorm(embed_size)
    
            # Output head (projection to vocab)
            self.head = nn.Linear(embed_size, vocab_size, bias=False)
    
        def forward(self, idx):
            """
            idx: LongTensor of shape (batch_size, sequence_length) with token IDs
            Returns: logits => (batch_size, sequence_length, vocab_size)
            """
            bsz, seq_len = idx.size()
            assert seq_len <= self.block_size, "Sequence too long!"
    
            # Create token + positional embeddings
            pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
            pos = pos.unsqueeze(0)  # shape (1, seq_len)
            token_embeddings = self.token_emb(idx)           # (bsz, seq_len, embed_size)
            position_embeddings = self.pos_emb(pos)          # (1, seq_len, embed_size)
            x = token_embeddings + position_embeddings       # (bsz, seq_len, embed_size)
    
            # Pass through each GPT block
            for block in self.blocks:
                x = block(x)
    
            # Final layer norm + linear head
            x = self.ln_f(x)
            logits = self.head(x)  # (bsz, seq_len, vocab_size)
            return logits
    
    if __name__ == "__main__":
        # Example usage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Hyperparams for demonstration (much smaller than GPT-3 scale)
        vocab_size = 50257  # typical GPT-2/3 Byte-Pair Encoding
        block_size = 128    # max context length
        embed_size = 768
        num_heads = 12
        num_layers = 6
        batch_size = 2
        seq_len = 64
    
        model = GPTModel(
            vocab_size=vocab_size,
            block_size=block_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_layers=num_layers,
            expansion_factor=4,
            dropout=0.1
        ).to(device)
    
        # Create random input tokens
        x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    
        # Forward pass
        logits = model(x)
        print("Logits shape:", logits.shape)  # (2, 64, 50257)
    
    ```
    
- DDP Code (Distributed Data Parallelism )
    
    
    ```python
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    
    def get_dummy_data(batch_size=8, input_dim=10, num_samples=64):
        data = torch.randn(num_samples, input_dim)
        labels = torch.randint(0, 2, (num_samples,))
        dataset = TensorDataset(data, labels)
        return dataset
    
    class SimpleNet(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=16, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        
        def forward(self, x):
            return self.net(x)
    
    def ddp_worker(rank, world_size):
        """
        rank: which process is this?
        world_size: total number of processes
        """
        print(f"[Process {rank}] Initializing...")
    
        # 1) Initialize the process group
        dist.init_process_group(
            backend='nccl',            # for GPU communication
            init_method='tcp://localhost:12355',  # or some other URL
            world_size=world_size,
            rank=rank
        )
        
        # 2) Set device for this process
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)  # ensure the device is pinned to the rank
    		
        # 3) Create model, move to device
        model = SimpleNet().to(device)
        ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    		
        # 4) Prepare data with distributed sampler
        dataset = get_dummy_data(num_samples=64)
        
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
        # 5) Standard training setup
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
    
        # 6) Train
        ddp_model.train()
        for epoch in range(2):
            sampler.set_epoch(epoch)  # ensures each epoch shuffles uniquely for each rank
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
    2
                optimizer.zero_grad()
                outputs = ddp_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
    
            if rank == 0:  # Usually only rank=0 prints
                print(f"[Rank {rank}] Epoch {epoch}, Loss: {loss.item():.4f}")
    
        # 7) Clean up
        dist.destroy_process_group()
    
    def run_ddp_example(world_size):
        """
        Launch N processes, each running ddp_worker
        """
        mp.spawn(ddp_worker, args=(world_size,), nprocs=world_size, join=True)
    
    if __name__ == "__main__":
        # Suppose you have 2 GPUs
        world_size = 2
        run_ddp_example(world_size)
    ```

    # Pytorch Coding Round

### Pytorch Questions

[Practice 1](https://www.notion.so/Practice-1-1f736157c382807fab2af0d1f32ea733?pvs=21)

[Practice 2](https://www.notion.so/Practice-2-20f36157c382805b9f22ee3c983ac33b?pvs=21)

LLM specific

[LLM specific questions](https://www.notion.so/LLM-specific-questions-20f36157c3828090baa0e5faa3abfd24?pvs=21)

- GPT2 Code (Imp to Understand)
    
    ```python
    import torch
    import torch.nn as nn
    
    class CausalSelfAttention(nn.Module):
        """
        Multi-Head Causal Self-Attention:
          - Splits embedding into multiple heads
          - Computes scaled dot-product attention
          - Applies a causal mask so tokens cannot attend to future positions
        """
        def __init__(self, embed_size, num_heads, dropout=0.1):
            super().__init__()
            assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
            self.embed_size = embed_size
            self.num_heads = num_heads
            self.head_dim = embed_size // num_heads
    
            # Transform from embed -> queries, keys, values
            self.q_proj = nn.Linear(embed_size, embed_size)
            self.k_proj = nn.Linear(embed_size, embed_size)
            self.v_proj = nn.Linear(embed_size, embed_size)
    
            # Output projection back to embed dimension
            self.out_proj = nn.Linear(embed_size, embed_size)
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            """
            x: (batch_size, seq_len, embed_size)
            """
            bsz, seq_len, _ = x.size()
    
            # Compute Q, K, V
            q = self.q_proj(x)  # (bsz, seq_len, embed_size)
            k = self.k_proj(x)
            v = self.v_proj(x)
    
            # Reshape to (bsz, seq_len, num_heads, head_dim) then transpose to
            # (bsz, num_heads, seq_len, head_dim) for multi-head attention
            q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
            # Scale queries
            q = q / (self.head_dim ** 0.5)
    
            # Compute attention scores: (bsz, num_heads, seq_len, seq_len)
            att_scores = torch.matmul(q, k.transpose(-2, -1))
    
            # Causal mask: restrict attention to current + previous positions
            causal_mask = torch.ones((seq_len, seq_len), device=x.device).tril()
            # shape (seq_len, seq_len), 1.0 in lower-triangular, 0.0 elsewhere
            # We expand to [1, 1, seq_len, seq_len] for broadcasting
            att_scores = att_scores.masked_fill(causal_mask == 0, float('-inf'))
    
            # Attention weights
            att_weights = torch.softmax(att_scores, dim=-1)
            att_weights = self.dropout(att_weights)
    
            # Weighted sum of values
            out = torch.matmul(att_weights, v)  # (bsz, num_heads, seq_len, head_dim)
    
            # Recombine heads
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_size)
    
            # Final linear projection
            out = self.out_proj(out)
            return out
    
    class FeedForward(nn.Module):
        """
        Position-wise Feed Forward layer, typically:
          - Linear (embed_size) -> (4*embed_size)
          - Activation (GELU or ReLU)
          - Linear (4*embed_size) -> (embed_size)
        """
        def __init__(self, embed_size, expansion_factor=4, dropout=0.1):
            super().__init__()
            inner_dim = expansion_factor * embed_size
            self.net = nn.Sequential(
                nn.Linear(embed_size, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, embed_size),
                nn.Dropout(dropout),
            )
    
        def forward(self, x):
            return self.net(x)
    
    class GPTBlock(nn.Module):
        """
        Single block of GPT-style transformer:
          - LayerNorm
          - Causal Self-Attention + skip
          - LayerNorm
          - FeedForward + skip
        """
        def __init__(self, embed_size, num_heads, expansion_factor=4, dropout=0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_size)
            self.attn = CausalSelfAttention(embed_size, num_heads, dropout)
            self.ln2 = nn.LayerNorm(embed_size)
            self.ff = FeedForward(embed_size, expansion_factor, dropout)
    
        def forward(self, x):
            # Causal Self-Attention sub-layer
            attn_out = self.attn(self.ln1(x))  # apply LN first
            x = x + attn_out  # residual connection
    
            # Feed Forward sub-layer
            ff_out = self.ff(self.ln2(x))
            x = x + ff_out  # residual connection
    
            return x
    
    class GPTModel(nn.Module):
        """
        Decoder-Only Transformer, GPT-style.
    
        Typical GPT-3 scale settings (for reference, not shown in code default):
          - vocab_size ~ 50k
          - embed_size ~ 12,288
          - num_heads ~ 96
          - n_layers ~ 96
          - block_size (context length) ~ 2048
        """
        def __init__(
            self,
            vocab_size,
            block_size,       # maximum sequence length
            embed_size=768,   # smaller default for demonstration
            num_heads=12,     # smaller default
            num_layers=12,    # smaller default
            expansion_factor=4,
            dropout=0.1,
            pad_id=0
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.block_size = block_size
            self.embed_size = embed_size
            self.pad_id = pad_id
    
            # Token + Position embeddings
            self.token_emb = nn.Embedding(vocab_size, embed_size)
            self.pos_emb = nn.Embedding(block_size, embed_size)
    
            # Transformer blocks
            self.blocks = nn.ModuleList([
                GPTBlock(embed_size, num_heads, expansion_factor, dropout)
                for _ in range(num_layers)
            ])
    
            # Final layer norm
            self.ln_f = nn.LayerNorm(embed_size)
    
            # Output head (projection to vocab)
            self.head = nn.Linear(embed_size, vocab_size, bias=False)
    
        def forward(self, idx):
            """
            idx: LongTensor of shape (batch_size, sequence_length) with token IDs
            Returns: logits => (batch_size, sequence_length, vocab_size)
            """
            bsz, seq_len = idx.size()
            assert seq_len <= self.block_size, "Sequence too long!"
    
            # Create token + positional embeddings
            pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
            pos = pos.unsqueeze(0)  # shape (1, seq_len)
            token_embeddings = self.token_emb(idx)           # (bsz, seq_len, embed_size)
            position_embeddings = self.pos_emb(pos)          # (1, seq_len, embed_size)
            x = token_embeddings + position_embeddings       # (bsz, seq_len, embed_size)
    
            # Pass through each GPT block
            for block in self.blocks:
                x = block(x)
    
            # Final layer norm + linear head
            x = self.ln_f(x)
            logits = self.head(x)  # (bsz, seq_len, vocab_size)
            return logits
    
    if __name__ == "__main__":
        # Example usage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Hyperparams for demonstration (much smaller than GPT-3 scale)
        vocab_size = 50257  # typical GPT-2/3 Byte-Pair Encoding
        block_size = 128    # max context length
        embed_size = 768
        num_heads = 12
        num_layers = 6
        batch_size = 2
        seq_len = 64
    
        model = GPTModel(
            vocab_size=vocab_size,
            block_size=block_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_layers=num_layers,
            expansion_factor=4,
            dropout=0.1
        ).to(device)
    
        # Create random input tokens
        x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    
        # Forward pass
        logits = model(x)
        print("Logits shape:", logits.shape)  # (2, 64, 50257)
    
    ```
    
- DDP Code (Distributed Data Parallelism )
    
    
    ```python
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    
    def get_dummy_data(batch_size=8, input_dim=10, num_samples=64):
        data = torch.randn(num_samples, input_dim)
        labels = torch.randint(0, 2, (num_samples,))
        dataset = TensorDataset(data, labels)
        return dataset
    
    class SimpleNet(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=16, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        
        def forward(self, x):
            return self.net(x)
    
    def ddp_worker(rank, world_size):
        """
        rank: which process is this?
        world_size: total number of processes
        """
        print(f"[Process {rank}] Initializing...")
    
        # 1) Initialize the process group
        dist.init_process_group(
            backend='nccl',            # for GPU communication
            init_method='tcp://localhost:12355',  # or some other URL
            world_size=world_size,
            rank=rank
        )
        
        # 2) Set device for this process
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)  # ensure the device is pinned to the rank
    		
        # 3) Create model, move to device
        model = SimpleNet().to(device)
        ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    		
        # 4) Prepare data with distributed sampler
        dataset = get_dummy_data(num_samples=64)
        
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
        # 5) Standard training setup
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
    
        # 6) Train
        ddp_model.train()
        for epoch in range(2):
            sampler.set_epoch(epoch)  # ensures each epoch shuffles uniquely for each rank
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
    2
                optimizer.zero_grad()
                outputs = ddp_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
    
            if rank == 0:  # Usually only rank=0 prints
                print(f"[Rank {rank}] Epoch {epoch}, Loss: {loss.item():.4f}")
    
        # 7) Clean up
        dist.destroy_process_group()
    
    def run_ddp_example(world_size):
        """
        Launch N processes, each running ddp_worker
        """
        mp.spawn(ddp_worker, args=(world_size,), nprocs=world_size, join=True)
    
    if __name__ == "__main__":
        # Suppose you have 2 GPUs
        world_size = 2
        run_ddp_example(world_size)
    ```
    


## AFTER YOU ARE DONE WITH ABOVE QUESTIONS:

For an interview focusing on creative problem-solving and being clever with PyTorch, you can expect questions that test your ability to:

```
1.	Optimize Model Training & Memory Usage
•	How would you efficiently train a large model on limited GPU memory?
•	How do mixed precision training and gradient checkpointing work?
•	Implement a custom torch.autograd.Function to save memory.

2.	Custom Implementations & PyTorch Internals
•	Implement a custom activation function with PyTorch.
•	How does torch.nn.Module work internally?
•	Explain how PyTorch’s autograd computes gradients.

3.	Efficient Tensor Operations
•	Optimize a given PyTorch operation to minimize GPU memory and maximize speed.
•	Implement a function that computes a rolling window mean using efficient tensor operations.
•	Why should we prefer torch.einsum over explicit loops?

4.	Parallelism & Multi-GPU Training
•	Implement a simple data parallel training loop.
•	What is the difference between torch.nn.DataParallel and torch.nn.parallel.DistributedDataParallel?
•	How do you handle synchronization issues in multi-GPU training?

5.	Custom Loss Functions & Gradients
•	Implement a custom loss function that requires second-order gradients.
•	How do you stop gradients from flowing through part of the computation graph?

6.	Debugging & Profiling Performance Issues
•	How would you debug a PyTorch model that is training extremely slowly?
•	Use torch.profiler to identify bottlenecks in a model’s training loop.

7.	Reinforcement Learning / Optimization-Specific Questions
•	Implement a basic reinforcement learning policy network in PyTorch.
•	How would you use PyTorch for differentiable optimization tasks?

```

# Concepts

**1. PyTorch & Basic Usage**

- `torch.no_grad()` for inference
- `model.eval()` mode vs. `model.train()` mode
- Device management (CPU/GPU transfers, `tensor.to(device)`)
- Basic `nn.Module` creation and forward passes
- Parameter counting and memory footprint estimation
- Profiling with `torch.profiler` or `torch.autograd.profiler`

**2. Tensor Operations & Data Handling**

- Manual implementation of softmax, embedding lookups, basic activation functions
- Padding sequences, attention masks
- Micro-batching / sub-batching to handle GPU memory constraints
- Vectorized operations vs. Python loops (performance considerations)

**3. Automatic Differentiation & Gradients**

- Disabling gradient computation for inference
- Custom autograd Functions (forward/backward definitions)
- Understanding why `.item()` calls cause synchronization

**4. Transformer & LLM Fundamentals (Coding-Level)**

- Scaled dot-product attention (queries, keys, values)
- Multi-head attention (splitting tensors into heads, re-concat)
- Transformer decoder block structure (residuals, layer norms, feed-forward)
- Positional encodings (sinusoidal or learned)
- Weight tying (input embeddings and output projections)

**5. Decoding Methods**

- Greedy decoding loops
- Beam search & how to manage multiple hypotheses
- Top-k / nucleus (top-p) sampling (logits filtering & sampling with `torch.multinomial`)
- Speculative decoding (two-model approach: draft vs. verify)
- Handling end-of-sequence tokens and stopping conditions

**6. KV Caching & Autoregressive Generation**

- Storing/reusing past key-values for subsequent tokens
- Managing model outputs and partial sequences across multiple steps
- Minimizing redundant computation in loops

**7. Inference Optimization Techniques**

- Mixed precision inference (e.g., `torch.cuda.amp.autocast()`)
- Quantization: dynamic quantization (on CPU) vs. static quantization (or QAT)
- Pruning and the effect of unstructured vs. structured sparsity
- LoRA (low-rank adaptation) for parameter-efficient fine-tuning
- TorchScript (`torch.jit.trace` vs. `torch.jit.script`)
- PyTorch 2.x compile (`torch.compile`) and partial compilation
- Avoiding CPU synchronization (minimizing `.item()` calls, asynchronous GPU ops)

**8. Parallel & Distributed Strategies**

- Data parallelism (splitting a batch across multiple GPUs)
- Model parallelism (splitting layers across different GPUs)
- Micro-batching vs. bigger batch sizes
- Overlapping data transfer & compute via CUDA streams
- CPU threading vs. multiprocessing for data preprocessing

**9. Low-Level GPU / Kernel Topics**

- Writing custom CUDA kernels with Triton (`@triton.jit`, `tl.load`, `tl.store`)
- Understanding how to configure launch blocks, grids, block sizes
- Managing warp divergence, coalesced memory access, pinned memory
- Understanding GPU concurrency (streams, asynchronous transfers)

**10. Model Serving & Deployment Frameworks**

- Exporting PyTorch to ONNX (e.g., `torch.onnx.export`) and building a TensorRT engine
- Using TensorRT Python API or Torch-TensorRT
- Dynamic batching configurations (`config.pbtxt` in Triton Inference Server)
- Hugging Face TGI server usage (simple REST or gRPC calls)
- vLLM’s continuous batching logic (paged KV cache, dynamic scheduling)
- Hugging Face Optimum usage (exporting to ONNX Runtime, `BetterTransformer`, etc.)

**11. Code Structure & Validation**

- Comparing outputs (e.g., original vs. optimized model) for numerical drift
- Managing memory leaks (retain_graph issues, using `.detach()`, or `torch.no_grad()`)
- Handling control flow that can break tracing (e.g., `torch.jit.script` needed)
- Logging or storing outputs carefully without building large computation graphs

---

## **MUST WATCH**

https://www.youtube.com/watch?v=3XUG7cjte2U

https://www.youtube.com/watch?v=gXDsVcY8TXQ&t=127s

https://youtu.be/LDG6fc8jUS4?si=O4IO7HjUfsOhCiDf

https://www.youtube.com/watch?v=-K3bZYHYHEA&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj&index=1

https://www.youtube.com/watch?v=toUSzwR0EV8

https://www.youtube.com/watch?v=LuieZTc-hvU

https://www.youtube.com/watch?v=HQeKwCsnH4k&list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT&index=2

## **TIME UNTE**

https://www.youtube.com/watch?v=tC01FRB0M7w

https://www.youtube.com/watch?v=IpHjDoW4ffw

https://youtu.be/C9QSpl5nmrY?si=wqjyr32LpP12LbXF




# LLM specific questions

Below are **25** basic (yet thorough) *coding-focused* questions that test fundamental PyTorch skills relevant to building and running LLMs. 
They range from creating and manipulating tensors, to implementing small transformer components, to applying sampling methods. 
Each question should prompt you to write working code (in a live environment or whiteboard style), ensuring you can demonstrate good coding practices in PyTorch for LLM use cases.

---

1. **Tensor Creation & Manipulation**
    - **Question**: Write a small function to:
        1. Create a 2D PyTorch tensor (e.g. shape `[3,4]`) of random floats.
        2. Print its shape and data type.
        3. Move it to a GPU if available.
        4. Reshape it to `[2,6]`.
    - *Tests ability to create, reshape, and manage devices for tensors.*
2. **Embedding Lookup**
    - **Question**: Suppose you have a vocabulary size of 10,000 and an embedding dimension of 768. Create an `nn.Embedding` for this vocabulary. Then:
        1. Generate a batch of token indices (e.g., shape `[batch_size=4, seq_len=5]`).
        2. Pass these indices through the embedding to get the corresponding embeddings.
    - *Tests creation and usage of embedding layers, along with batch dimension handling.*
3. **Forward Pass Through a Simple Network**
    - **Question**: Define a small `nn.Module` that includes:
        1. An `nn.Embedding` layer.
        2. A single `nn.Linear` layer mapping from embedding dimension to a “hidden” dimension of your choice.
        3. A forward method that takes token indices, embeds them, and produces a final output tensor.
    - *Tests understanding of custom modules, forward methods, and dimension handling.*
4. **Positional Encoding**
    - **Question**: Write a function that takes input embeddings `(batch, seq_len, embedding_dim)` and adds sinusoidal positional encodings of shape `(seq_len, embedding_dim)` to them. Show how you would:
        1. Generate the sinusoidal encodings (using `sin` and `cos`).
        2. Broadcast-add them to the batch of embeddings.
    - *Tests how to handle shape broadcasting and incorporate positional information for LLMs.*
5. **Basic Autoregressive Decoding Loop (Greedy)**
    - **Question**: Assume you have a function `model.forward(input_ids)` that returns logits over your vocabulary. Write a greedy decoding loop that:
        1. Starts with a prompt (list of token IDs).
        2. Iteratively obtains the next token by taking `argmax` of the logits at each step.
        3. Continues until you reach a special `<EOS>` token or a maximum length.
    - *Tests ability to implement the simplest decode strategy by coding a loop with a model call each iteration.*
6. **Top-k Sampling Decoding**
    - **Question**: Modify the above decoding loop to implement top-k sampling instead of greedy:
        1. Use `torch.topk` on the logits to keep only the top-k tokens.
        2. Sample from the resulting distribution using `torch.multinomial`.
    - *Tests handling probability distributions and random sampling in PyTorch for more varied text generation.*
7. **Nucleus (Top-p) Sampling**
    - **Question**: Write a function to implement top-p (nucleus) sampling in one step of decoding:
        1. Sort token probabilities by descending order.
        2. Select tokens until their cumulative probability ≥ p.
        3. Sample from the truncated distribution.
    - *Tests dynamic selection of a token set based on cumulative probability.*
8. **Mini “Attention” Mechanism**
    - **Question**: Implement a simple scaled dot-product attention from scratch. Given `Q, K, V` of shape `(batch, seq_len, dim)`, compute:
        
        $$
        Attention(Q,K,V)=softmax(QK⊤d)V\text{Attention}(Q, K, V) = \mathrm{softmax}\Big(\frac{Q K^\top}{\sqrt{d}}\Big) V
        $$
        
        - *Tests basic matrix multiplications, shape alignment, and understanding of attention in LLMs.*
9. **Layer Normalization**
    - **Question**: Write your own PyTorch module that implements `LayerNorm` manually (i.e., do not use `nn.LayerNorm`). Show how you’d:
        1. Compute mean and variance across the last dimension.
        2. Subtract mean, divide by std, multiply by a learnable `gamma`, and add a learnable `beta`.
    - *Tests knowledge of normalization steps and custom parameter usage.*
10. **Masking in Attention**
- **Question**: Extend your scaled dot-product attention code to support an attention mask (e.g., a boolean mask of shape `(batch, seq_len, seq_len)`). Any position where the mask is `False` should be assigned a very negative value (like `1e9`) before the softmax.
- *Tests the typical approach for ignoring future tokens or padded tokens in attention calculation.*
1. **KV Caching**
- **Question**: Show how to implement a basic “past key-value” cache for an autoregressive model. Suppose your model returns `(logits, new_k, new_v)`, and you want to store `(k, v)` from all previous timesteps to avoid recomputing them. Demonstrate how you’d:
    1. Initialize empty lists or tensors for the cache.
    2. Append `new_k, new_v` at each time step.
    3. Pass the entire cached `(k, v)` to the attention mechanism.
- *Tests your understanding of how LLMs speed up inference by caching previous computations.*
1. **Dynamic Padding / Batching**
- **Question**: Suppose you have a list of sequences (lists of token IDs) of varying lengths. Write a function `collate_fn(batch)` that:
    1. Finds the longest sequence in the batch.
    2. Pads all sequences to that length.
    3. Stacks them into a single PyTorch tensor `(batch_size, max_len)`.
- *Tests handling variable-length inputs in an LLM setting.*
1. **Mixed Precision Inference**
- **Question**: Write a code snippet using `torch.cuda.amp.autocast()` that performs a forward pass in half precision:
    1. Create a small model (or load an existing one).
    2. Run the forward pass in the `amp` context manager.
    3. Print the dtype of the output to confirm it’s half precision.
- *Tests knowledge of half precision usage for faster GPU-based LLM inference.*
1. **Basic TorchScript Scripting**
- **Question**: Given a small PyTorch module for text classification, show how to script it using `torch.jit.script(model)`. Then do a forward pass with a sample input on the scripted module.
- *Tests ability to compile a model with TorchScript for potential inference optimizations.*
1. **Beam Search**
- **Question**: Implement beam search for an LLM. At each step:
    1. Keep track of the top *beam_size* partial sequences.
    2. Expand each partial sequence by possible next tokens.
    3. Prune down to the top *beam_size* expansions based on cumulative log probability.
- *Tests more advanced decoding strategy relevant to many LLM tasks.*
1. **Temperature Scaling**
- **Question**: Write a decode function that, at each step, applies a temperature factor τ to the logits: `logits = logits / tau`, and then does a softmax sampling. Demonstrate that setting `tau < 1.0` makes generation more deterministic (less random), and setting `tau > 1.0` yields more varied text.
- *Tests ability to handle sampling diversity through temperature scaling.*
1. **Check Model Parameter Counts**
- **Question**: Write code to:
    1. Count the total parameters in a model (sum of `p.numel()` for all parameters).
    2. Separate them into trainable vs. non-trainable parameters.
    3. Print the result in a user-friendly format (e.g., “Total parameters: X, Trainable: Y, Frozen: Z”).
- *Tests basic parameter inspection and that you can confirm correct freezing or updating of certain layers in an LLM scenario.*
1. **Implement a Simple GPT-like Block**
- **Question**: Build an `nn.Module` named `GPTBlock` that contains:
    1. A self-attention sublayer (multi-head or single-head).
    2. A feed-forward sublayer.
    3. Residual connections & layer norms.
    4. A `forward` method that expects `(x, mask=None)`.
- *Tests modular design of LLM building blocks, reflecting GPT-like architecture basics.*
1. **Training Loop vs. Inference Loop**
- **Question**: Write code that sets up a training loop for a toy language model (e.g., next-token prediction on random data). Then create a separate function or block for *inference* that:
    1. Disables gradient tracking (`no_grad`).
    2. Uses the model in eval mode.
    3. Performs next-token prediction.
- *Tests clarity about switching between training and inference in real code, which is crucial for LLM usage.*
1. **Use Hugging Face Transformers for a Basic LLM**
- **Question**: Demonstrate loading a small GPT-2 model (`'gpt2'`) via `AutoModelForCausalLM.from_pretrained('gpt2')`. Then manually:
    1. Tokenize an input prompt with `AutoTokenizer`.
    2. Convert the tokenizer outputs to tensors.
    3. Run the model in a loop to generate tokens (greedy or top-k).
- *Tests direct coding with HF Transformers library for an LLM scenario, ensuring familiarity with model & tokenizer usage.*
1. **Manual Gradient-Freezing**
- **Question**: Suppose you only want to fine-tune the last two layers of a model. Write code to:
    1. Freeze all model parameters by default.
    2. Unfreeze only the final two layers.
    3. Confirm that only those last two layers’ parameters have `requires_grad=True`.
- *Tests knowledge of partial fine-tuning, which is common in LLM training/inference setups (LoRA, etc.).*
1. **Storing Intermediate Outputs**
- **Question**: In a forward method, suppose you want to store the attention maps (the `softmax` results) for analysis. Write code that:
    1. Saves each head’s attention map to a Python dictionary or list.
    2. Ensures you aren’t inadvertently storing a full computation graph. (Hint: use `.detach()` or `with torch.no_grad()` appropriately.)
- *Tests ability to debug or visualize attention while ensuring no memory leaks or graph references remain.*
1. **Continuous Batching Concept**
- **Question**: Sketch out code for a simple server loop that collects incoming requests (prompts), forms a batch from whichever requests arrived in the last X milliseconds or up to a max batch size, runs them all at once, and then returns the results. You don’t have to implement a full server—just show the pseudo-code for forming the batch and calling `model(batch)`.
- *Tests dynamic batching approach used in real LLM-serving frameworks (vLLM / TGI style).*
1. **Profiling Inference**
- **Question**: Using `torch.profiler`, show how to:
    1. Wrap a model’s forward pass in `torch.profiler.profile(...)`.
    2. Print or view the events, focusing on CPU vs. GPU time per operation.
    3. Identify which layer is taking the most time.
- *Tests performance analysis, essential in real LLM inference to find bottlenecks.*
1. **Memory Footprint & GPU Cleanup**
- **Question**: Suppose you have run some large batches and GPU memory is near capacity. Show code that:
    1. Deletes any intermediate tensors you no longer need.
    2. Calls `torch.cuda.empty_cache()` if necessary (with disclaimers about what it does).
    3. Verifies GPU memory usage with something like `torch.cuda.memory_allocated()` or `nvidia-smi` checks in Python.
- *Tests awareness of GPU memory management, which is critical in LLM inference serving for large contexts.*

---

**How to Use These Questions**

Each prompt is meant to be solved *live*—ideally with actual PyTorch code that you can execute or walk through. These exercises collectively cover the foundational coding concepts for building,
fine-tuning, and serving LLMs in PyTorch. By practicing them, you’ll gain familiarity with everything from basic tensor operations and embedding layers to advanced tasks like caching key-value pairs 
for faster generation and implementing custom attention.


# Practice 1

---

### **Foundations & Basic PyTorch Operations**

1. **Basic Tensor Creation:**
    
    *Question:* Write a PyTorch code snippet to create a 3×3 tensor of random numbers and multiply it by 2.
    
    *Why:* To ensure you understand tensor creation, basic arithmetic, and using built‑in functions.
    
2. **Tensor from Python Data Structures:**
    
    *Question:* How do you create a tensor from a list of lists and compute its element‑wise square?
    
    *Why:* It practices converting Python lists to tensors and applying element‑wise operations.
    
3. **Tensor Indexing & Slicing:**
    
    *Question:* Write code to slice a given 4×4 tensor to extract the middle 2×2 sub-tensor.
    
    *Why:* It reinforces your ability to index and manipulate tensor subsets.
    
4. **Broadcasting in PyTorch:**
    
    *Question:* Demonstrate broadcasting by adding a 1×3 tensor to a 4×3 tensor.
    
    *Why:* To understand how PyTorch automatically expands dimensions during operations.
    
5. **Matrix Multiplication:**
    
    *Question:* Write a snippet that computes the matrix product of two tensors using `torch.matmul`.
    
    *Why:* To become comfortable with linear algebra operations that are foundational in deep learning.
    
6. **Understanding Data Types:**
    
    *Question:* How do you change a tensor’s data type (e.g., from float32 to int64) and why might this be important?
    
    *Why:* It’s critical to know how data types affect model computations and performance.
    

---

### **Autograd & Custom Gradients**

1. **Gradient Computation Basics:**
    
    *Question:* Write a code snippet to compute the gradient of a simple scalar function (e.g., f(x)=x²) using `torch.autograd`.
    
    *Why:* To understand how PyTorch tracks operations and computes gradients automatically.
    
2. **Disabling Gradients:**
    
    *Question:* How would you disable gradient calculations during inference using PyTorch? Write an example.
    
    *Why:* This practice is essential for saving memory and speeding up inference.
    
3. **Custom Autograd Function:**
    
    *Question:* Create a custom PyTorch autograd Function for a simple operation (for example, a custom square function) that implements both forward and backward passes.
    
    *Why:* To deepen your understanding of the autograd system and custom gradient computation.
    
4. **In-place vs. Out-of-place Operations:**
    
    *Question:* Explain the difference between in-place and out-of-place tensor operations and demonstrate with a code example.
    
    *Why:* To learn how in-place operations can affect gradient tracking and memory usage.
    

---

### **Building Neural Network Modules**

1. **Simple Linear Model:**
    
    *Question:* Implement a linear regression model using `nn.Module` in PyTorch.
    
    *Why:* To practice building a custom model and understand module structure.
    
2. **Feedforward Neural Network:**
    
    *Question:* Code a two-layer MLP for a simple classification task and explain each component.
    
    *Why:* It builds your skills in structuring multi-layer networks.
    
3. **Activation Functions:**
    
    *Question:* Write a custom PyTorch module that applies ReLU or GELU activation and explain why non-linearities are essential.
    
    *Why:* To appreciate the role of activation functions in deep networks.
    
4. **Building a Convolutional Layer:**
    
    *Question:* Create a custom convolutional layer using PyTorch’s functional API.
    
    *Why:* To understand low-level implementation details of convolution, a building block for many models.
    
5. **Dropout Implementation:**
    
    *Question:* Write code to add dropout in a neural network module and explain its role during training vs. inference.
    
    *Why:* Dropout is crucial for regularization, and you should know how to integrate it properly.
    
6. **Batch Normalization:**
    
    *Question:* Implement a simple network that uses BatchNorm, and explain how it improves training stability.
    
    *Why:* Batch normalization is a widely used technique to accelerate convergence.
    

---

### **Training Loop & Model Optimization**

1. **Custom Training Loop:**
    
    *Question:* Write a complete training loop (forward pass, loss computation, backpropagation, and optimizer step) for a simple neural network.
    
    *Why:* To master the end-to-end process of training models in PyTorch.
    
2. **Using GPU Acceleration:**
    
    *Question:* How do you modify your training loop to leverage GPU acceleration in PyTorch? Provide a code example.
    
    *Why:* To ensure you can transfer models and data to GPU for faster computation.
    
3. **Learning Rate Scheduling:**
    
    *Question:* Write code to implement a learning rate scheduler using `torch.optim.lr_scheduler` and explain its benefit.
    
    *Why:* Adaptive learning rates are important for effective training.
    
4. **Gradient Clipping:**
    
    *Question:* How do you apply gradient clipping in a training loop? Write a code snippet.
    
    *Why:* It prevents exploding gradients, which is especially useful in training deep or recurrent networks.
    
5. **Saving and Loading Models:**
    
    *Question:* Write functions to save a model’s state and then load it back.
    
    *Why:* To understand model serialization and checkpointing for production deployment.
    
6. **Custom Loss Function:**
    
    *Question:* Create and implement a custom loss function in PyTorch (e.g., a weighted mean squared error).
    
    *Why:* Building custom loss functions deepens your understanding of how optimization objectives affect training.
    
7. **Early Stopping Mechanism:**
    
    *Question:* Write code to implement early stopping in a training loop.
    
    *Why:* Early stopping is a practical strategy to prevent overfitting and save resources.
    
8. **Model Evaluation:**
    
    *Question:* Code a validation loop that computes accuracy and loss on a validation set, and discuss its integration in training.
    
    *Why:* To ensure you can evaluate your models effectively during training.
    
9. **Data Augmentation & Custom DataLoader:**
    
    *Question:* Write a custom PyTorch Dataset class and DataLoader that applies data augmentation on-the-fly.
    
    *Why:* Data loading and augmentation are key to building robust models.
    

---

### **Deep Dive into Transformers & LLMs**

1. **Implementing Positional Encoding:**
    
    *Question:* Write a PyTorch module to compute sinusoidal positional encodings for sequence data.
    
    *Why:* Positional encoding is vital for Transformers to capture sequence order.
    
2. **Scaled Dot-Product Attention:**
    
    *Question:* Code the scaled dot-product attention mechanism and explain its components (queries, keys, values, scaling).
    
    *Why:* Attention mechanisms are at the heart of Transformers.
    
3. **Multi-head Attention Module:**
    
    *Question:* Implement a multi-head attention module in PyTorch, including splitting and concatenating heads.
    
    *Why:* Multi-head attention enhances the model’s ability to capture diverse features.
    
4. **Transformer Encoder Block:**
    
    *Question:* Build a single Transformer encoder block that includes multi-head attention, layer normalization, and a feedforward network.
    
    *Why:* It’s a core building block for LLMs and advanced NLP models.
    
5. **Masked Self-Attention:**
    
    *Question:* Code a masked self-attention layer for auto-regressive generation, explaining how and why the mask is applied.
    
    *Why:* To understand how Transformers handle sequential data during generation.
    
6. **Feedforward Network in Transformer:**
    
    *Question:* Implement the position-wise feedforward network used in Transformer layers.
    
    *Why:* It complements the attention mechanism and is essential for learning non-linear transformations.
    
7. **Layer Normalization from Scratch:**
    
    *Question:* Write your own PyTorch implementation of layer normalization and compare it to `nn.LayerNorm`.
    
    *Why:* Understanding normalization techniques helps optimize model training.
    
8. **Custom Transformer Decoder Layer:**
    
    *Question:* Build a Transformer decoder layer, incorporating masked attention and encoder-decoder attention mechanisms.
    
    *Why:* Decoders are critical for sequence-to-sequence tasks like translation.
    
9. **Implementing a Simple Transformer Model:**
    
    *Question:* Assemble a full Transformer model for a language modeling task, including embeddings, positional encoding, encoder/decoder stacks, and a linear output layer.
    
    *Why:* To integrate all components into a functioning model from scratch.
    
10. **Handling Variable-length Sequences:**
    
    *Question:* Write code to implement padding and create attention masks for variable-length sequences in a Transformer model.
    
    *Why:* Proper masking is key to processing batches with varying sequence lengths.
    
11. **Custom Activation Functions:**
    
    *Question:* Implement the GELU activation function in PyTorch without using built-in functions.
    
    *Why:* To deepen your understanding of non-linear activations and their numerical implementations.
    

---

### **Advanced Topics & Production-level Coding**

1. **Gradient Accumulation:**
    
    *Question:* Write a training loop that uses gradient accumulation to simulate larger batch sizes on limited GPU memory.
    
    *Why:* It helps when hardware constraints limit batch size.
    
2. **Mixed-Precision Training:**
    
    *Question:* Modify your training loop to use PyTorch’s AMP for mixed-precision training.
    
    *Why:* Mixed-precision can significantly speed up training while conserving memory.
    
3. **Custom Collate Function:**
    
    *Question:* Write a custom collate function for a DataLoader that dynamically pads sequences for a batch.
    
    *Why:* To efficiently handle variable-length data during batching.
    
4. **Integrating TensorBoard:**
    
    *Question:* Write a snippet that logs loss and accuracy during training using TensorBoard.
    
    *Why:* Monitoring training metrics is critical for debugging and optimizing performance.
    
5. **Implementing Beam Search:**
    
    *Question:* Code a basic beam search algorithm for sequence generation using a Transformer model.
    
    *Why:* Beam search is a common technique to improve generation quality in LLMs.
    
6. **Caching in Auto-Regressive Generation:**
    
    *Question:* Implement a caching mechanism to store and reuse key/value pairs during Transformer inference.
    
    *Why:* This optimization is crucial for efficient text generation.
    
7. **Visualization of Attention Weights:**
    
    *Question:* Write a function that extracts and visualizes attention weights from a Transformer layer.
    
    *Why:* Visualization helps you understand and debug model behavior.
    
8. **Model Parallelism Basics:**
    
    *Question:* Explain and demonstrate how to split a model’s layers across multiple GPUs using PyTorch’s distributed features.
    
    *Why:* To handle large models that exceed the memory of a single GPU.
    
9. **Implementing a Custom Optimizer/Modifier:**
    
    *Question:* Write code that customizes an existing optimizer (e.g., by modifying learning rates per layer) or implements a simple custom optimizer.
    
    *Why:* To explore and understand optimization strategies beyond standard implementations.
    
10. **Integrating PyTorch Lightning:**
    
    *Question:* Refactor one of your training scripts using PyTorch Lightning and discuss the benefits.
    
    *Why:* Lightning helps to simplify and structure training code for scalability and reproducibility.
    
11. **Custom Callbacks in Lightning:**
    
    *Question:* Create a custom PyTorch Lightning callback for early stopping or dynamic learning rate adjustments.
    
    *Why:* Custom callbacks further automate training management and improve model performance.
    
12. **Distributed Data Parallel (DDP):**
    
    *Question:* Write a sample script that trains a model using PyTorch’s Distributed Data Parallel (DDP) on multiple GPUs.
    
    *Why:* Understanding DDP is crucial for scaling training on multi-GPU systems.
    
13. **Profiling GPU Memory Usage:**
    
    *Question:* Develop a small utility that profiles and logs GPU memory usage during model training/inference.
    
    *Why:* Profiling helps you optimize memory consumption and debug performance issues.
    
14. **End-to-End LLM Implementation:**
    
    *Question:* Implement a Transformer-based language model from scratch—including training, evaluation, and inference scripts—that can be fine-tuned on a sample dataset.
    
    *Why:* This capstone challenge integrates all components (tensor operations, autograd, custom modules, advanced optimization, and distributed training) to simulate a production-level LLM development process.
    

---

Happy coding and good luck with your preparation! Kummesey!









# ML Inference System Design

# Skim and Scan:

[High level conceptual notes](https://www.notion.so/High-level-conceptual-notes-20f36157c38280dfa5bfdd4df6bcec3a?pvs=21)

[Concepts to Master](https://www.notion.so/Concepts-to-Master-20f36157c38280e981fae8a3cd255c28?pvs=21)

# Study in Detail:

[**Parallelism Techniques for Large-Scale LLM Inference**](https://www.notion.so/Parallelism-Techniques-for-Large-Scale-LLM-Inference-20f36157c38280a7946bc29285a042d4?pvs=21)

# End Practice:

[**10 Rigorous LLM System Design Interview Questions**](https://www.notion.so/10-Rigorous-LLM-System-Design-Interview-Questions-20f36157c38280e2869ee5ba4074037b?pvs=21)




# High level conceptual notes

## 1. Model Serving (Local & Global)

- **Local Serving**
    - Single data center or a single node handling inferences.
    - Lower latency (no cross-region hops), simpler infrastructure.
    - Works best if all your users/traffic are geographically close.
- **Global Serving**
    - Multiple data centers or cloud regions, each with replicas of the model.
    - Load balancer directs requests based on proximity or resource availability.
    - Minimizes latency for globally distributed users, but cost and complexity go up.
    - Must consider cross-region load balancing, network overhead, and possible replication of large model artifacts.
- **Key Considerations**
    - **Latency**: Real-time user experience requires local-ish compute.
    - **Concurrency**: More replicas → handle more requests but higher overall cost.
    - **Resilience**: Multi-region can handle outages, but orchestration gets complex.
    - **Cost**: GPU hours in multiple regions can be pricey; you might scale down in regions with low usage.

---

## 2. Continuous Batching

- **Core Idea**
    - Autoregressive decoding merges multiple requests at the same token step into one forward pass.
    - Greatly boosts GPU utilization by combining small workloads into a single big kernel launch.
- **Benefits**
    - **Throughput**: Fewer, larger GPU kernels instead of many small ones.
    - **Latency**: Small added “batch wait” (a few ms) can be worth the big throughput gain.
    - **Easy Integration**: Tools like vLLM, Hugging Face TGI do this automatically.
- **Mechanics**
    - Orchestrator checks which sessions are at decode step *n*, merges them, runs one forward pass for that token → merges again for *n+1*, etc.
    - Works even if requests arrive asynchronously or have different prompt lengths.
- **Trade-Off**
    - If concurrency is very low (fewer requests), your batch size might be 1. Gains may be smaller.
    - Must carefully handle user experience so we don’t stall them too long while batching.

---

## 3. Parallelism Strategies (Pros, Cons, Uniqueness)

### A. **Data Parallelism**

- **Concept**: Replicate the entire model on each GPU (or node).
- **Pros**
    - Simple: Each GPU runs a full copy of the model, no cross-GPU sync for forward passes.
    - Scales throughput linearly for many concurrent requests.
- **Cons**
    - Doesn’t reduce per-request latency. One large inference still runs on one GPU.
    - Memory duplication for all replicas → might be expensive if model is huge.
- **Use Case**
    - High concurrency (many user requests), and the model fits on one GPU.

### B. **Tensor (Model) Parallelism**

- **Concept**: Split each weight matrix (or attention heads) across multiple GPUs.
- **Pros**
    - Enables serving models larger than a single GPU’s memory.
    - Can reduce latency if compute is heavy and interconnect is fast.
- **Cons**
    - Requires high-speed GPU interconnects (NVLink/InfiniBand). Communication overhead can bottleneck.
    - Complex to implement (Megatron-LM, etc.). Diminishing returns with too many GPUs.
- **Use Case**
    - Ultra-large model that doesn’t fit on one GPU. HPC or specialized clusters.

### C. **Pipeline Parallelism**

- **Concept**: Each GPU holds a consecutive chunk of layers; pass activations from one stage to the next in sequence.
- **Pros**
    - Also allows serving bigger models than one GPU can hold.
    - Increases throughput if you have many micro-batches in flight (assembly-line).
- **Cons**
    - Single-request latency can be higher (sequential pipeline stages + inter-stage comm).
    - Requires careful load balancing among pipeline stages.
- **Use Case**
    - Very deep models (lots of layers). Good for batch or multi-request concurrency.

### D. **Expert Parallelism (MoE)**

- **Concept**: Many “expert” sub-models distributed across GPUs; gating network routes tokens to specific experts.
- **Pros**
    - Sparse activation → can hold massive total parameters with limited per-token compute.
    - Scales model capacity almost arbitrarily (more experts = bigger model).
- **Cons**
    - Complex routing, possible load imbalance if many tokens pick the same expert.
    - High communication overhead when scattering tokens to different experts.
- **Use Case**
    - Extremely large models with specialized “experts” (multi-language, multi-domain).

---

## 4. KV Caching & APC (Automatic Prefix Caching)

- **KV Cache**
    - Stores hidden states (Key & Value tensors) from past tokens to speed up subsequent attention steps.
    - Essential for autoregressive LLMs (GPT-like).
- **Benefits**
    - Avoids recomputing entire sequence for every new token – big latency and throughput boost.
    - Must manage memory (KV can be large if many tokens or many concurrent requests).
- **APC (Automatic Prefix Caching)**
    - If multiple prompts share the same prefix, reuse the same computed KV chunk.
    - Greatly speeds up repeated patterns (e.g., same system prompts or repeated instructions).
- **Implementation Details**
    - **PagedAttention**: KV blocks stored in “pages” for dynamic allocation and offload.
    - Some frameworks automatically detect prefix overlaps.

---

## 5. Eviction Policies & Offloading Strategies

- **Why Evict?**
    - GPU memory is precious. Large or idle sessions can hog KV space.
    - Eviction frees memory for new or active sessions.
- **Common Policies**
    - **LRU (Least Recently Used)**: Discard the oldest or least-accessed session’s KV.
    - **Time-Based**: If a session is idle for X seconds, remove or move it to CPU.
    - **Priority**: Premium sessions never evict; low-priority sessions evict first.
- **Offloading Approaches**
    - **KV Offload**: Move old tokens’ KV to CPU pinned memory or disk. Reload if needed.
    - **Partial Summarization**: Summarize older context, reduce token count (soft eviction).
    - **FlexGen** (advanced): Offloads *model weights* as well, loading layers on-demand.
- **Trade-Off**
    - Reloading from CPU or disk can spike latency for revived sessions.
    - Summarization saves GPU memory but might lose detailed context.

---

## 6. Scaling Inference Algorithms

Here are some acronyms and features to keep in mind:

- **GQA (Grouped Query Attention) / MQA (Multi-Query Attention)**
    - Variants of multi-head attention with fewer key/value heads to reduce memory usage.
    - Can help scale to longer contexts or reduce overhead.
- **MLA (Multi-Loader Attention?)**
    - Not a standard acronym in mainstream usage; might refer to specialized attention or multi-level attention.
    - Key idea: optimizing how attention states are loaded or partitioned.
- **FlashAttention**
    - Fused kernel that calculates attention in one pass using GPU shared memory.
    - Dramatically reduces memory reads/writes, lowering latency for large seq lengths.
- **Speculative Decoding**
    - Use a smaller “draft model” to predict multiple tokens at once, then verify with the large model.
    - Achieves 2–3x speedups if the draft model’s predictions are usually correct.
- **Quantization**
    - 8-bit or 4-bit weights (INT8/FP8) to reduce memory footprint and speed up matmul.
    - Slight hit to model accuracy but huge gains in throughput.

---

## 7. Ways to Optimize for **Latency**

Think of three angles: **System-Level**, **Model-Level**, **Hardware-Level**.

### **System-Level**

- **Low-Latency Batching Windows**
    - Keep batch windows (waiting time) short so tokens appear quickly.
- **Local Serving**
    - Deploy replicas close to user region to cut network RTT.
- **High-Speed Interconnect**
    - Use NVLink or InfiniBand for multi-GPU setups to reduce communication overhead.
- **Pipeline Stage Minimization**
    - Too many pipeline stages across nodes → high hop latency.

### **Model-Level**

- **Reduced Sequence Length**
    - Summarize or chunk context if possible to shorten sequence.
- **FlashAttention**
    - Minimizes attention overhead at large sequence lengths.
- **Quantization**
    - Fewer bits → faster compute → lower latency (with caution on accuracy).
- **Speculative Decoding**
    - Big latency win if the small draft model’s guesses are good.

### **Hardware-Level**

- **GPU Generation**
    - Modern GPUs (A100, H100) have better memory bandwidth + tensor cores.
- **Sufficient GPU Memory**
    - Avoid constant offloading to CPU/disk, which kills latency.
- **Efficient CUDA Kernels**
    - Fused ops reduce overhead (FlashAttention, fused MLP, etc.).

---

## 8. Ways to Optimize for **Throughput**

Again, consider **System**, **Model**, and **Hardware**.

### **System-Level**

- **Continuous Batching**
    - Merge multiple requests per decode step → higher total tokens/sec.
- **Autoscaling / Data Parallel**
    - Multiple replicas handle more requests in parallel.
- **Eviction Policies**
    - Free up memory from idle sessions to serve more active requests concurrently.
- **Load Balancer**
    - Distribute requests so no single node is overloaded while others idle.

### **Model-Level**

- **Tensor Parallel**
    - Split big layers across GPUs → handle bigger batches concurrently (if interconnect is fast).
- **Pipeline Parallel**
    - Keep multiple micro-batches in flight like an assembly line.
- **Quantization**
    - Smaller data → bigger batch fits in GPU memory → more tokens per second.
- **MoE (Expert Parallel)**
    - Sparse activation: can handle large batch if routing is balanced.

### **Hardware-Level**

- **Scaling Up GPU Count**
    - More GPUs (with enough bandwidth) → more total throughput.
- **High-Bandwidth Networking**
    - Critical if your model is sharded (tensor or pipeline).
- **Faster Disks / Storage**
    - If offloading to disk (FlexGen), faster NVMe or SSD read speeds matter.

---

### One-Liner Reminder

1. **Local vs. Global** → Where you serve from. Concurrency vs. region & cost trade-offs.
2. **Continuous Batching** → Merge decode steps for throughput with minimal latency penalty.
3. **Parallelism** → Data (throughput), Tensor (big model), Pipeline (layers), MoE (experts).
4. **KV & APC** → Cache previous tokens and share repeated prefixes = speed.
5. **Eviction & Offload** → LRU/time-based to manage GPU memory. Summarize or store old KV.
6. **Scaling Algos** → FlashAttention, Quantization, Speculative Decoding = big speedups.
7. **Latency** → Optimize system, model, hardware. Batching windows, local replicas, high-end GPUs.
8. **Throughput** → Data parallel replicas, continuous batching, quantization, and HPC interconnect.

Keep these bullet points in mind, and you’ll have a strong mental map of how to handle high-performance, scalable LLM inference. Good luck!


# Concepts to Master

Includes all the essential concepts mentioned … continuous batching, parallelism strategies, caching, offloading/eviction policies—and link to the most relevant open-source frameworks (vLLM, TGI, TensorRT-LLM, etc.), whitepapers, videos, and documentation.

# System Design Study Guide: LLM Inference & Training Infrastructure

This guide is a comprehensive prep toolkit for a Machine Learning Engineer – Inference role (such as at Together AI). It focuses on the system-level architectural patterns for **large-scale LLM inference system design** and the supporting **LLM training infrastructure**.

## 1. Continuous Batching for LLM Inference

**Summary:** Continuous batching (also called **dynamic batching** or **iteration-level scheduling**) is a technique to maximize GPU utilization by p ([TensorRT: Optimizing Model Inference for Maximum Performance | by Kishore C S | Medium](https://medium.com/@kish.imss/tensorrt-optimizing-model-inference-for-maximum-performance-8be266f78ec0#:~:text=1,optimizes%20the%20computation%20graph%2C%20removing))requests at the token level rather than one-request-at-a-time. In static or request-level batching, all requests in a batch must finish before new ones start, leaving the GPU underutilized when shorter sequences finish early. Continuous batching instead fills those “gaps” immediately with new incoming requests, processing many requests in an interleaved fashion token-by-token. This yields much higher throughput (often an order of magnitude or more) at minimal latency cost, especially under real-world multi-user loads. Key challenges include scheduling policies (when to add new requests vs. wait), handling timeouts and sequence padding efficiently, and ensuring memory for attention key/value caches is managed as sequences of different lengths are mixed. Modern LLM inference servers like Hugging Face’s **Text Generation Inference (TGI)** and UC Berkeley’s **vLLM** implement continuous batching to achieve significantly better throughput and cost-efficiency compared to naive batching.

**Resources:**

- **ORCA Paper (OSDI 2022)** – Introduced *iteration-level scheduling* for transformer inference (the idea behind continuous batching). ORCA shows that once a sequence finishes, a new one can be inserted in its place, increasing utilization. Achieved 36× throughput improvement on GPT-3 175B vs. static batching.
- **Anyscale Blog on Continuous Batching (2023)** – Great overview with benchmarks. Shows up to *23× higher throughput* and lower latency using continuous batching (via vLLM) vs. conventional methods of static and request-based batching, then how continuous batching works in systems like Ray Serve and HF TGI.
- **Baseten Blog – Continuous vs Dynamic Batching** – Explains the differences between static, dynamic, and continuous batching in simple terms. Recommends continuous batching for LLMs (token-level scheduling) to eliminate idle time waiting on the longest sequence. Provides analogies (like a bus filling freed seats mid-route) and mentions frameworks: TGI and vLLM offer continuous batching, while NVIDIA TensorRT-LLM uses a similar “in-flight batching” approach.
- **LLM Inference Optimization (Medium)** – Discusses continuous batching and selective batching in ORCA and vLLM. Also touches on mem ([TensorRT: Optimizing Model Inference for Maximum Performance | by Kishore C S | Medium](https://medium.com/@kish.imss/tensorrt-optimizing-model-inference-for-maximum-performance-8be266f78ec0#:~:text=1,optimizes%20the%20computation%20graph%2C%20removing)) ([TensorRT: Optimizing Model Inference for Maximum Performance | by Kishore C S | Medium](https://medium.com/@kish.imss/tensorrt-optimizing-model-inference-for-maximum-performance-8be266f78ec0#:~:text=2,a%20more%20efficient%20inference%20pipeline))y iteration-level processing (like vLLM’s PagedAttention for efficient KV cache management).

## 2. Parallelism Strategies for LLMs (Model & Data Parallelism)

**Summary:** Large LLMs often require splitting work across multiple GPUs or machines. **Model parallelism** refers to partitioning a single model’s execution across GPUs. This comes in flavors: **tensor parallelism (TP)** splits the *tensors within each layer* (e.g. a weight matrix is divided among GPUs), while **pipeline parallelism (PP)** assigns different layers (or groups of layers) to different GPUs, passing intermediate activations along a pipeline. Many large-model training setups (e.g. NVIDIA Megatron-LM) use a combination of TP + PP to fit and accelerate models across dozens of GPUs. **Data parallelism (DP)**, on the other hand, replicates the full model on each GPU and splits different input data among them – this is more common in training (synchronizing gradients) but offers limited benefit for single-query inference. For inference serving, DP can be used to handle more requests in parallel (throughput scaling) by running multiple model instances. **GPU multi-streaming** refers to using CUDA streams to execute multiple inference kernels concurrently on one GPU (when one model alone doesn’t fully saturate it). This can increase utilization for smaller models, though for large LLMs single-stream usually keeps the GPU busy (and context-switching overhead or memory bandwidth can bottleneck multi-stream performance). In practice, high-throughput LLM inference might use *batching + single-stream* for each model, and scale out to multiple GPUs via model parallelism or multiple replicas rather than concurrent streams on one device. The key is understanding the trade-offs: tensor/model parallelism adds communication overhead (especially across nodes), pipeline parallelism adds latency due to fill/drain of the pipeline, and data parallelism is memory-intensive (multiple copies of weights) – so efficient combinations and hardware-aware tuning (NVLink, InfiniBand usage) are required for multi-GPU/multi-node LLM serving.

**Resources:**

- **NVIDIA NeMo Guide – Parallelism** – Official definitions of parallel strategies. TP “distributes the parameter tensor of an individual layer across GPUs” (e.g. splitting a large fully-connected layer’s weights), while PP “assigns consecutive layers to different GPUs”. Also covers **sequence parallelism** and **optimizer/gradient sharding** in training.
- **Hugging Face Transformers – Model Parallelism Docs** – Describes how pipeline parellism are used to spread a model like GPT-2 or GPT-3 across GPUs. Noted that pipeline parallelism addresses GPU idle time by using micro-batches to overlap computation.
- **DeterminedAI Blog – Tensor Parallelism** – Tutorial-style explanation of TP, including how intermediate results are combined from multiple devices. Useful for intuition on how splitting a matrix multiplication across GPUs works in practice.
- **Victor Leung (2025) – NVIDIA Inference Optimizations** – Discusses that as LLMs grow, **tensor parallelism be ([How Pytorch 2.0 Accelerates Deep Learning with Operator Fusion ...](https://medium.com/data-science/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26#:~:text=How%20Pytorch%202,First))ial** even if a model fits on one GPU, because TP can double memory bandwidth and compute by using 2 GPUs, thus improving latency (with some communication cost). Recommends NVLink/HGX systems for ef ([TensorRT: Optimizing Model Inference for Maximum Performance | by Kishore C S | Medium](https://medium.com/@kish.imss/tensorrt-optimizing-model-inference-for-maximum-performance-8be266f78ec0#:~:text=1,optimizes%20the%20computation%20graph%2C%20removing)) ([TensorRT: Optimizing Model Inference for Maximum Performance | by Kishore C S | Medium](https://medium.com/@kish.imss/tensorrt-optimizing-model-inference-for-maximum-performance-8be266f78ec0#:~:text=2,a%20more%20efficient%20inference%20pipeline))
- **Stack Overflow: CUDA Streams & Concurrency** – Notes that by default, frameworks execute one stream per GPU, so multiple requests get serialized. Using custom CUDA streams can allow overlapping operations, but for large LLM kernels the benefit might be limited (as one inference already heavily uses the GPU). If serving many small queries, frameworks like Triton can spawn multiple model execution instances on one GPU (using streams) to increase throughput.

## 3. Caching Strategies (KV Cache Reuse & Persistence)

**Summary:** **Attention key-value (KV) caching** is crucial to LLM inference. Transformers generate key and value tensors for each token at each layer; caching those from prior tokens means each new token only computes attention for the new token rather than recomputing all tokens from scratch. This makes autoregressive generation scale *linearly* instead of quadratically with sequence length. However, KV caches consume a lot of memory – growing linearly with sequence length, batch size, and number of layers/heads. Effective caching strategies are needed to balance speed and memory:

- **KV Cache Reuse across requests:** If multiple requests share an identical prefix, we can compute that prefix’s KV cache once and reuse it for all – skipping those tokens’ computation for subsequent requests. This is *Automatic Prefix Caching (APC)* as implemented in vLLM. For example, if user queries often start with the same prompt, the server can detect it and reuse the cached keys/values instead of recomputing the prefix every time.
- **Persistent KV caching / Warm cache:** In a long-running service, one might persist popular prefixes’ caches (in memory or even on disk) to “warm start” new queries that contain those prefixes. This reduces tail latency for repeated prompts at the cost of memory. vLLM’s APC allows skipping shared parts of prompts entirely.
- **Grouped Query/Key Attention (GQA):** This is an architectural tweak to reduce cache size. Multi-Query Attention (MQA) means using a single shared key/value per all heads (or a small group of heads) instead of separate per head. Models like Falcon, LLaMA-70B, Mistral use this to cut KV cache size (e.g. if 16 heads share one KV, that’s 16× less KV memory). GQA trades some model capacity for much smaller caches.
- **Cache eviction policies:** Since GPU memory for KV cache is limited, one must evict old caches when they are no longer needed. A simple policy is **Least Recently Used (LRU)** – evict caches from requests that finished longest ago. More advanced strategies involve priorities (e.g. keep caches of more likely-to-repeat prompts).
- **On-disk or CPU cache extension:** Tools like **LMDB** or custom cache stores can persist KV tensors when GPU memory is full, for later reuse. This is tricky due to bandwidth limits but can be worthwhile for frequently repeating contexts.

**Resources:**

- **vLLM Blog – PagedAttention & Prefix Caching** – Introduces *PagedAttention*, which allocates KV memory in fixed-size pages to reduce fragmentation and allow dynamic growth. Also covers **Automatic Prefix Caching** in vLLM: new queries that share prefixes with running queries can reuse those cached keys/values to skip computation. This yields significant throughput improvements for workloads with repeated prefixes.
- **NVIDIA Tech Blog (2025) – TensorRT-LLM Cache Optimizations** – Discusses how **KV cache grows** with model size, batch, and context, straining memory. Highlights TensorRT-LLM’s support for **paged KV cache**, **quantized KV cache** (storing KV in INT8/FP8 to halve memory), **circular buffers**, etc.. Also describes an LRU eviction by default and a new priority-based eviction API for custom retention of certain sequences’ cache. Good for understanding practical GPU cache management.
- **“LLM Inference Series #4: KV Caching” (Medium, 2024)** – Deep dive into how large KV caches can get, and strategies to cope. Quantifying memory use per token. Explains **KV cache quantization**: e.g. FlexGen compressing KV to 4-bit, or SmoothQuant/LLM.int8() approaches that quantize activations so the KV cache uses fewer bytes. Also details multi-query attention (GQA) and lists which models use it (PaLM, Falcon, etc.).
- **“LM Cache” Project** – (Blog) Describes an external KV cache store that can hold “hundreds of times” more cached contexts by storing them off-GPU, to be retrieved when a repeated prompt comes. Although external, it illustrates persistent caching beyond GPU memory for inference.
- **Hugging Face TGI Documentation – Caching** – Notes that TGI v3 introduced efficient cache management and chunked processing. TGI also supports prefix-batching ( ([AI Gateway for unified LLM inferencing](https://www.truefoundry.com/ai-gateway#:~:text=Performance%20%26%20Reliability))) and streaming of cache to CPU if needed.

## 4. Eviction and Offloading Policies (Memory Management for LLMs)

**Summary:** Given the limited memory, systems need strategies to **offload** parts of the model or its activations to CPU or disk, and to **evict** less-used data from fast memory. For inference, this often means:

- **Offloading model weights:** If the model is too large for GPU memory, some weights can reside in CPU RAM or even NVMe and be brought in when needed. Approaches like **FlexGen ( ([Microservices: Fault Tolerance Mechanisms | by Anu Abraham](https://medium.com/@anu.abraham37/microservices-fault-tolerance-mechanisms-e3b9fc6ce744#:~:text=Microservices%3A%20Fault%20Tolerance%20Mechanisms%20,unavailable%2C%20the%20system%20can))his by partitioning the model across GPU, CPU, and disk with an I/O-aware schedule. FlexGen compresses weights and KV to 4-bit and uses a **“zigzag” scheduling** to overlap data transfer with compute, achieving high throughput even with minimal GPU memory.
- **Offloading activations/KV:** During generation, not all KV cache can stay on GPU if context is very long or batch is large. Systems may offload older layers’ KV to CPU and fetch back for attention computation on new tokens. This requires careful scheduling to avoid stalls. Tools like **DeepSpeed** and **Accelerate** provide hooks to offload activations to CPU between layers for memory savings.
- **Eviction policies:** When memory is full, decide what to evict. **LRU (Least Recently Used)** is a common default – evict the oldest finished sequence’s KV cache (assuming it’s least likely to be reused). Some scenarios might fav ([[FEATURE] AI model/provider pool, or fallback models #874 - GitHub](https://github.com/langchain4j/langchain4j/issues/874#:~:text=,fallback%20if%20one%20model%2Fprovider%20errors))st Recently Used)** eviction (e.g. if recent contexts won’t repeat soon, but older ones might in a Q&A loop). More advanced is letting the application assign priorities – e.g. keep caches for premium users or frequently repeating prompts longer (TensorRT-LLM’s priority eviction API).
- **FlexGen-style scheduling:** FlexGen formalizes offloading as an optimization problem: how much of weights/KV to keep in each tier (GPU, CPU, disk) to maximize throughput given bandwidth constraints. It introduced **“Double Buffering”** and **“Overlap”** so that while GPU processes one layer, the next layer’s weights are being transferred from CPU/disk in parallel. This way, it hides a lot of the I/O latency.
- **Memory fragmentation & compaction:** Long-running inference servers may fragment GPU memory (especially if context lengths vary). Strategies like pooling memory or using fixed “blocks” (as in vLLM’s PagedAttention) help avoid fragmentation so that offloaded data can be brought in contiguous free spaces.

**Resources:**

- **FlexGen Paper (2023)** – *High-throughput Generative Inference on a Single GPU.* Shows how a 175B model can run on one 16GB GPU by offloading majority of data. It quantizes weights+KV to 4-bit and uses CPU RAM + SSD as extensions of GPU memory. The paper details its scheduling algorithm that decides which layers’ weights (and how much of KV) stay on GPU vs CPU vs disk. It achieved ~1 token/sec on GPT-NeoX-20B with just 4GB GPU memory by carefully overlapping I/O and compute.
- **DeepSpeed ZeRO-Infinity** – Microsoft’s ZeRO-Infinity (stage 3) is another offloading approach for *training* that also benefits inference of gigantic models. It automatically partitions model states across GPU, CPU, and NVMe, and can swap memory pages in/out as needed. Their blog/paper shows near-linear scalability in model size by leveraging CPU memory effectively.
- **NVIDIA TensorRT-LLM** – Provides an *Executor* that can offload completed sequences’ KV cache to host memory when not in use, and bring them back if needed for, say, a user editing a prompt and continuing generation. By default it evicts LRU blocks, but the new API allows marking certain sequences to keep (for better *cache hit* on reuse).
- **vLLM PagedAttention** – Although aimed at fragmentation, it also implicitly helps offloading: since KV is allocated in pageable chunks, the system could evict unused pages of KV caches. Their blog notes under 4% memory wastage with paging vs. up to 20–30% in traditional contiguous allocation. This kind of design makes offloading more granular (page by page).
- **AWS Blog on Multi-Node Inference (2024)** – Describes a multi-node deployment of a 405B model using Triton and TensorRT-LLM. They emphasize that **sharding** the model across nodes requires robust failover: if one node’s GPU fails or is slow, the system needs to retry that shard’s work on another node or have redundancy, else the whole inference stalls. This touches fault tolerance (next topics) but also offloading in a sense – e.g. if a node drops out, others may have to load that part of the model on the fly.

## 5. LLM Serving Architectures & Inference Frameworks

**Summary:** Several specialized serving systems have emerged to handle large-scale LLM inference, each with features like optimized CUDA kernels, batching, and distributed support:

- **vLLM (UC Berkeley)** – An open-source LLM serving engine built around *PagedAttention* and continuous batching. It achieves extremely high throughput by managing GPU memory for KV caches efficiently and scheduling at token-level. The team reports up to **24× faster** than naive HuggingFace Transformers and ~3× throughput of HF’s TGI on certain workloads. vLLM supports dynamic batching, streaming, and can distribute across multiple GPUs. Great for latency-sensitive applications due to its innovative memory management.
- **Hugging Face Text Generation Inference (TGI)** – A production-grade inference server (in Rust + Python) for generative models. It supports continuous batching, multi-GPU tensor-parallelism, and provides HTTP endpoints. TGI integrates **FlashAttention** and other fused ops for speed. It’s highly used in industry for deploying LLaMA, GPT-J, etc., and can serve thousands of requests with streaming. The design focuses on performance and features like token streaming, safe scaling, etc. (It’s behind HuggingFace’s Hosted Inference API).
- **NVIDIA TensorRT-LLM** – Part of NVIDIA’s TensorRT ecosystem tailored for LLMs. It compiles Transformer models to optimized TensorRT engines, using tactics like fused kernels, quantization, and support for new precisions (FP8). TensorRT-LLM provides Python APIs to build and execute these engines and includes features like in-flight batching (similar to continuous batching) and KV c ([Building Production-Ready LLM Inferencing Pipeline: A Step-by-Step Guide | by Chirav Dave | Feb, 2025 | Medium](https://medium.com/@davechirav/building-production-ready-llm-inferencing-pipeline-a-step-by-step-guide-6622fc510a3b#:~:text=Step%203%3A%20Setting%20Up%20the,Inference%20Server)) ([Building Production-Ready LLM Inferencing Pipeline: A Step-by-Step Guide | by Chirav Dave | Feb, 2025 | Medium](https://medium.com/@davechirav/building-production-ready-llm-inferencing-pipeline-a-step-by-step-guide-6622fc510a3b#:~:text=Step%204%3A%20API%20Server%20Deployment))It’s optimized for NVIDIA GPUs (Hopper, Ampere, etc.) and often achieves lower latency per token by using kernels handcrafted for Transformer block ([Building Production-Ready LLM Inferencing Pipeline: A Step-by-Step Guide | by Chirav Dave | Feb, 2025 | Medium](https://medium.com/@davechirav/building-production-ready-llm-inferencing-pipeline-a-step-by-step-guide-6622fc510a3b#:~:text=With%20the%20inference%20engine%20in,This%20server%20is%20responsible%20for))A Triton Inference Server** – An inference serving framework that can host multiple models (of different types) and allows for dynamic batching and concurrent execution o ([Building Production-Ready LLM Inferencing Pipeline: A Step-by-Step Guide | by Chirav Dave | Feb, 2025 | Medium](https://medium.com/@davechirav/building-production-ready-llm-inferencing-pipeline-a-step-by-step-guide-6622fc510a3b#:~:text=SOTA%20Inference%20Server%20Options%3A))L389】. Triton is framework-agnostic (supports TensorRT engines, TorchScript, ONNX, etc.) and commonly used to deploy models at scale (with HTTP/GRPC endpoints, model repository, auto-scaling). For LLMs, Triton can work in conjunction with TensorRT-LLM or FasterTransformer backends to serve compiled models. It handles scheduling, queuing, and metrics. Triton’s advantage is in multi-model scenarios and ease of deployment on Kubernetes.
- ([Architect scalable and cost-effective LLM & RAG inference pipelines](https://decodingml.substack.com/p/architect-scalable-and-cost-effective#:~:text=pipelines%20decodingml,business%20logic%20into%20two%20layers))ransformer (NVIDIA)** – A library of highly optimized GPU kernels for Transformer inference (supports GPT-2/3, BERT, etc.). It provides C++ implementations that can be integrated into serving systems (both Triton and HuggingFace TGI have options to use FasterTransformer). It features batch beam search, multi-GPU support, and efficient sampling methods, which can greatly speed up generation throughput.
- **DeepSpeed-Inference (Microsoft)** – Part of DeepSpeed, offers optimized kernels (e.g. quantized int8 ops via **DeepSpeed Turbo**), concurrency scheduling, and supports very large models with ZeRO-offloading. It can serve models with tens of billions of parameters on a single node by spilling to CPU.
    
    In practice, these frameworks often complement each other (e.g., TGI now supports using TensorRT-LLM and vLLM as *backends*). The choice depends on the use case: vLLM and TGI are easy to use for generic models and handle batching for you; TensorRT-LLM and FasterTransformer give maximal performance if you can compile the model (but require NVIDIA GPUs); Triton is ideal for scaling many models or when you need a robust microservice architecture.
    

**Resources:**

- **vLLM Paper/Blog (2023)** – *“Easy, Fast, and Cheap LLM Serving with PagedAttention.”* Describes vLLM’s architecture and its novel memory management. Reports that vLLM delivers **2–3.5× higher throughput than HF TGI** under various conditions. Explains how it avoids 60–80% memory waste seen in naive allocation by using OS-like paging for KV cache. A must-read to understand continuous batching and memory efficiency in LLM serving.
- **HuggingFace TGI Documentation** – Highlights TGI features: “high-performance text generation for popular LLMs” with *continuous batching*, *tensor parallelism*, *streaming*, *flash attention*, *paged attention* integration, and *quantization* support. Also covers monitoring, reliability (it’s production-hardened). Checkout the **“Internal Architecture”** and **“v3 update, caching and chunking”** sections for insight into how TGI handles concurrent requests.
- **NVIDIA Triton Inference Server Overview (AWS blog)** – Introduces Triton as an open-source serving solution that simplifies deployment of AI models at scale, supporting multiple frameworks and GPUs/CPUs. Emp ([[2106.09685] LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685#:~:text=example%20,trainable%20parameters%2C%20a%20higher%20training)) Triton allows **parallel execution of models** on a single GPU and dynamic batching to boost throughput. It’s widely used in industry for its reliability and easy scaling.
- **NVIDIA TensorRT-LLM Blog/Release Notes** – Discusses how TensorRT-LLM achieves low latency: using kernels optimized for transformer layers, support for FP8 (on Hopper GPUs) t ([[2106.09685] LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685#:~:text=example%20,trainable%20parameters%2C%20a%20higher%20training))ed vs FP16, and combining techniques like multi-streaming, in-flight batching, and priority-based scheduling. NVIDIA’s developer blog on TensorRT-LLM (Jan 2025) also details the KV cache reuse and eviction features (see Topic 3 & 4).
- **GitHub – Text Generation Inference** – The README of TGI’s repo outlines usage and mentions it can serve BLOOM-176B with tensor parallel on 8 GPUs. It’s a good reference for how to deploy (Docker images, config options) and includes performance tips.
- **Serving at Scale Talks** – e.g. *“The Evolution of Multi-GPU Inference in vLLM”* (Ray Summit 2024 video) – discusses distributed serving with Ray + vLLM, and *“Mastering LLM Inference on HuggingFace”* (Nov 2023 webinar) – covers best practices with TGI.

## 6. Scaling LLM Inference to Multi-GPU & Multi-Node

**Summary:** When one GPU isn’t enough (either for model size or throughput), we scale out. **Multi-GPU inference** can be achieved via *model sharding* (each GPU holds part of the model) or *data parallel replicas* (each GPU has a full copy and handles different requests). For single large LLMs that don’t fit in one GPU’s memory, **model parallelism** is mandatory: e.g. split the model’s laye () ()e parallel) or split each layer’s parameters (tensor parallel) – see Topic 2. This requires high-speed interconnect (NVLink, PCIe4/5, or InfiniBand between nodes) to avoid bottlenecks when GPUs communicate activations or gradients. Libraries like Megatron-LM, DeepSpeed, or FasterTransformer h ([Quantization Aware Training (QAT) vs. Post-Training ... - Medium](https://medium.com/better-ml/quantization-aware-training-qat-vs-post-training-quantization-ptq-cd3244f43d9a#:~:text=Quantization%20Aware%20Training%20%28QAT%29%20vs,may%20result%20in%20some)) ([Improving INT8 Accuracy Using Quantization Aware Training and ...](https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/#:~:text=Improving%20INT8%20Accuracy%20Using%20Quantization,deployment%2C%20without%20compromising%20on%20accuracy))ails of partitioning and synchronization. **Multi-node inference** extends this across servers: you might shard a 175B model across 2–4 nodes, each with multiple GPUs. Frameworks such as Ray, MPI, or PyTorch RPC can coordinate the forward pass across nodes. For example, if a model is pipeline-parallel across 8 GPUs on 2 nodes, the sequence of layers must pass outputs over the network at pipeline boundaries. Ensuring minimal latency overhead is key – technologies like NCCL and GPUDirect RDMA help by providing efficient GPU-to-GPU communication across nodes.

To **scale throughput**, one can also deploy *multiple replicas* of the model across many GPUs and use a load balancer or scheduler to distribute incoming requests. This is horizontal scaling – e.g., 10 replicas of a 7B model on 10 GPUs to handle 10× the QPS (queries/sec). Usually, a combination is used: vertical scaling (shard the model on N GPUs to fit it) and horizontal scaling (M replicas of those N GPUs to handle load).

Important considerations:

- **Synchronization**: If using data parallel (for batching across GPUs), ensure all GPUs finish computing a token around the same time to avoid stragglers (usually okay in inference since no backprop). Model-parallel inference must synchronize at each layer or micro-batch boundary.
- **Distributed scheduling**: Systems like **Alpa** or **Tensor Parallel (ColossalAI)** automate splitting the model and running it in a distributed fashion. HuggingFace’s Accelerate can also automatically shard a model across multiple GPUs/nodes for inference, handling moving data to the correct device.
- **Multi-node orchestration**: Commonly done with Kubernetes or Slurm for static clusters, or frameworks like **Ray Serve** which can treat a cluster of GPUs a () (). Ray’s **Serving** layer can manage a pool of model replicas across nodes and route requests (with support for batching).
- **Bandwidth vs compute trade-off**: Multi-node inference of LLMs can be bandwidth-bound if huge tensors (like 10s of GB of weights or activations) must be sent over network frequently. The design often tries to *minim ([[2106.09685] LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685#:~:text=example%20,trainable%20parameters%2C%20a%20higher%20training))ation* rounds – e.g., using larger pipeline stages (so less frequent but larger transfers) or using tensor parallel only within a node and pipeline parallel across nodes to leverage intra-node high bandwidth.

**Resources:**

- **vLLM Distributed Inference Docs** – Notes that vLLM supports *tensor parallelism across GPUs and pipeline parallelism across nodes*. It currently uses Megatron-LM’s tensor-parallel algorithm. It outlines how to launch vLLM on multiple GPUs or machines, showing that setting `tensor_parallel_size=4` on 4 GPUs will shard the model automatically. Helpful for practical steps.
- **AWS HPC Blog: Multi-Node LLM Inference (2024)** – Demonstrates serving a 405B Llama model on two AWS p5 instances (each with 8×A100 GPUs) using **TensorRT-LLM + Triton**. They shard the model across t () ()netes with a custom scheduler (LeaderWorkerSet) to coordinate. This post gives insight into multi-node deployment issues like ensuring all shards load the correct weights, handling node failures, etc.
- **Google Cloud GKE Guide (2023)** – “Serve LLMs like DeepSeek 670B on GKE” – details a Kubernetes setup where the model is partitioned across multiple pods, and a serving client orchestrates the inference. Emphasizes using high-bandwidth networking (placed on same Switch etc.) and shows how latency scales with number of nodes.
- **Petals Project** – While not a typical enterprise use-case, Petals is a decentralized inference of a 176B model over the internet. It’s relevant in that it treats each volunteer node as hosting part of the model (like a giant multi-node). They had to implement robust fault tolerance (if a node drops, route to another hosting same layer) and request scheduling. Reading Petals’ documentation/paper can provide insight into multi-node scheduling under unreliable conditions – which is analogous to handling node failures in a cluster setting.
- **NVIDIA NeMo Megatron** – Mega-scale models (e.g. GPT-3, MT-NLG) are trained and inferenced with thousands of GPUs. NVIDIA’s Megatron-LM and NeMo have recipes for using *infiniband-connected nodes* with ring-allreduce for TP/PP communications. Their documentation on inference describes how to generate text with a model sharded across nodes, using an inference pipeline parallel approach.
- **Horovod/DeepSpeed-Inference** – Horovod (Uber) is mainly for training, but DeepSpeed-Inference extends beyond one node by using RDMA or TCP to scatter model partitions. The DeepSpeed system can automatically partition a model across GPUs (even across nodes if properly set up) and do inference – its InferenceEngine handles syncing. DeepSpeed’s docs or OSDI paper (Somerville et al.) could be insightful for design patterns.

## 7. Throughput vs. Latency Trade-offs (Batching & Token Generation)

**Summary:** There is an inherent trade-off between achieving high **throughput** (tokens generated per second, or QPS for entire requests) and low **latency** (response time for a single request). **Batching** is the primary lever here: processing multiple requests together greatly increases GPU utilization and throughput, but each request may wait a bit for others, adding latency. For example, batching 32 requests might 10× the throughput of single requests, but any given request could see a delay (waiting for the batch to fill or the next batch cycle). Continuous batching (Topic 1) mitigates this by not waiting for full batches, but still, larger effective batch sizes = more tokens per iteration = more latency per token. Another aspect is **token-level latency vs end-to-end latency**: LLMs stream token by token. If we aim for lowest time to first token (TTFT), we might run each request immediately (batch size 1) – but GPU is underutilized. To maximize throughput, we batch many tokens for parallel processing, which increases each token’s latency slightly.

In system design, one often sets an SLA (e.g. p95 latency of 2s for a response) and then maximize throughput under that constraint. Techniques:

- **Micro-batching**: Instead o ([Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#:~:text=Behind%20the%20scenes%2C%20Accelerate%20added,to%20the%20model%2C%20so%20that))h for the entire prompt generation, some servers dynamically adjust batch size per iteration to balance latency. For instance, if only a few requests are active, batch them together; if many are queued, batch more at once. This can keep latency low when load is light, and throughput high when load is heavy.
- **Max Tokens per step**: Systems can limit how many new tokens to generate in one GPU pass to avoid long latency spikes. E.g., generate 5 tokens at a time in a batch, then deliver and repeat, rather than 50 at once.
- **Concurrency vs. latency**: If we allow concurrency (multiple model instances), one can achieve low latency for some by runnin ([Understanding SafeTensors: A Secure Alternative to Pickle for ML Models - DEV Community](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71#:~:text=Beyond%20security%2C%20safetensors%20brings%20significant,reduced%20memory%20usage%20during%20loading))s on one GPU, while another GPU does a big batch for throughput. Coordination is needed to not overload.
- **Latency SLO based scheduling**: Some advanced approaches (like Microsoft’s *Autobatch* or newer schedulers) will group requests with similar latency requirements together. Real-time chatbot requests migh ([Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#:~:text=,and%20cleaned%20up%20just%20after))aller batches for snappiness, whereas large offline jobs (generating long text) can be heavily batched.

Fundamentally, **as you relax latency requirements, you can significantly boost throughput and reduce cost per query**. For example, one source notes that at concurrency 250 (i.e. lots of parallel requests waiting to batch), throughput can be 50× higher than at concurrency 1, while latency increases by only 5×【50†L57-L ([SageMaker FastFileMode, dataset streaming and memory mapping](https://discuss.huggingface.co/t/sagemaker-fastfilemode-dataset-streaming-and-memory-mapping/70464#:~:text=SageMaker%20FastFileMode%2C%20dataset%20streaming%20and,streaming%20less%20performant%20than))ght latency sacrifices yield huge throughput gains. The challenge is to find that sweet spot and perhaps provide **graceful degradation**: under overload, the system might automatically batch more (accept higher latency) to handle the load, rather than crash or queue indefinitely.

**Resources:**

- **Databricks Blog (2023) – LLM Inference Performance** – Includes a throughput-vs-latency curve (Figure 7) for MPT-7B at various batch sizes. It shows how latency grows when batch size increases, but throughput also jumps. They discuss using such curves to pick a batch size that meets a target latency. Also notes that for large models like 70B, cost-per-token is only efficient at high batch sizes – *“good cost/performance at large batch sizes… however, large batch → larger KV → more GPUs requ ([Understanding SafeTensors: A Secure Alternative to Pickle for ML Models - DEV Community](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71#:~:text=Performance%20Improvements)) ([Understanding SafeTensors: A Secure Alternative to Pickle for ML Models - DEV Community](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71#:~:text=Beyond%20security%2C%20safetensors%20brings%20significant,reduced%20memory%20usage%20during%20loading))-L319】. This highlights the multi-dimensional trade: latency vs throughput vs memory/cost.
- **Victor Leung’s Guide (2025)** – Summarizes the latency-throughput trade-off clearly: *“These two metrics are inversely related: improving one often comes at the expense of the other.”* Gives the example that 50× throughput gain only caused a 5× latency increase by increasing batch concurrency. Emphasizes adjusting for use-case (chatbot vs batch processing).
- **Sarathi (OSDI 2024)** – A research scheduler that dynamically decides batch sizes per iteration to balance this trade-off. The paper “Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi” shows that fixed-size batching is suboptimal and proposes a control algorithm to meet latency targets while batching as much as possible. It’s advanced reading, but demonstrates state-of-the-art in this space.
- **AWS Inferentia2 Benchmark Blog** – When introducing TGI on AWS Inferentia2, they noted how increasing batch sizes yielded higher throughput until latency violated certain thresholds. It provides practical numbers on how latency grows (they often keep latency within 1–2 seconds for chat apps by limiting batch size).
- **Anyscale (Continuous batching)** – The Anyscale blog (from Topic 1) also frames it nicely: continuous batching *improves both throughput and latency under load*. That seems counterintuitive but the key is “under load” – because without batching, latency would blow up due to queueing when many requests arrive. By batching, they reduced p50 latency *in heavy load scenarios* while massively increasing throughput. This suggests that beyond a point, not batching leads to even worse latency (as GPUs get backlogged). So intelligent batching actually *wins on both* up to a certain concurrency level. They provide data on p50 latency reduction with continuous batching.
- **Modal Blog – “Boost throughput with dynamic batching”** – A short piece explaining how even non-LLM workloads benefit from batching and how to implement a simple dynamic batch with timeouts. Although not specific to LLM, it helps in understanding how to set a maximum wait time so latency doesn’t increase unbounded.

## 8. Mixed Precision, Quantization, and Model Compression for Serving

**Summary:** **Mixed precision** refers to using lower-precision number formats (FP16, BF16, or even FP8) instead  ([Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#:~:text=device_map%20%3D%20%7B%20,)) ([Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#:~:text=Behind%20the%20scenes%2C%20Accelerate%20added,to%20the%20model%2C%20so%20that))d computations. This is almost standard now ([Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#:~:text=,and%20cleaned%20up%20just%20after)) ([Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#:~:text=This%20only%20supports%20the%20inference,GPU%20memory%20with%20intermediate%20activations))ory usage and can double throughput by utilizing tensor core hardware, with negligible accuracy loss for inference. Newer GPUs (NVIDIA H100) support **FP8** precision, which furt ([Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#:~:text=,and%20cleaned%20up%20just%20after))ory relative to FP16 – NVIDIA reports that FP8 can *double* throughput vs FP16 while maintaining accuracy via their Transformer Engine (which dynamically scales to prevent overflow). In serving, running LLMs in FP16 or BF16 is ([Understanding SafeTensors: A Secure Alternative to Pickle for ML Models - DEV Community](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71#:~:text=Performance%20Improvements)) ([Understanding SafeTensors: A Secure Alternative to Pickle for ML Models - DEV Community](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71#:~:text=Beyond%20security%2C%20safetensors%20brings%20significant,reduced%20memory%20usage%20during%20loading))ten preferred on newer hardware for its range and speed).

**Quantization** goes beyond float vs float – it uses integer representations, often **8-bit or 4-bit** for weights (and sometimes activations). This can dramatically reduce memory (a 4-bit model is 8× smaller than FP32) and can increase speed if optimized int8 ker ([Understanding SafeTensors: A Secure Alternative to Pickle for ML Models - DEV Community](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71#:~:text=Beyond%20security%2C%20safetensors%20brings%20significant,reduced%20memory%20usage%20during%20loading)) ([Understanding SafeTensors: A Secure Alternative to Pickle for ML Models - DEV Community](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71#:~:text=The%20design%20allows%20for%20partial,huge%20models%20under%20memory%20constraints))bit multiply-accumulate can be very fast on certain hardware). However, quantization can impact model quality if not done carefully. Two main types:

- **Post-Training Quantization (PTQ)**: Quantize a pre-trained model’s weights (and maybe activations) without additional training. Methods like GPTQ (for 4-bit) have shown you can quantize LLMs to 4-bit with minimal loss for inference. PTQ is easy and fast but for very low bits or if high accuracy is needed, it might degrade performance.
- **Quantization-Aware Training (QAT)**: Fine-tune the model with quantization in the loop so it learns to adapt to the lower precision. QAT typically yields better accuracy at low precision (e.g. enabling int8 with virtually no loss, or int4 with minimal loss), but requires additional training data and compute. For inference-serving, pure PTQ is more common due to the expense of QAT on huge models.

**Model compression** also includes **pruning** (removing redundant weights) and **distillation** (training a smaller model to mimic a larger one). Distillation can produce a much smaller model that’s faster to serve, at the cost of some accuracy. For example, distilling a 13B LLM down to 2B can massively reduce serving costs, though the smaller model may not perform as well. Another technique is **LoRA** or adapters (see Topic 13) which don’t compress the base model but make fine-tuning lighter.

For serving, the key strategies are:

- Use FP16/BF16 for all deployments (almost given for modern hardware) – 2x speed, 2x memory savings, no significant drawbacks【53† ([Stream - Hugging Face](https://huggingface.co/docs/datasets/en/stream#:~:text=Dataset%20streaming%20lets%20you%20work,you%20iterate%20over%20the%20dataset)) ([Stream - Hugging Face](https://huggingface.co/docs/datasets/v2.0.0/en/stream#:~:text=Stream%20,progressively%20as%20you%20iterate))t8 quantization if supported by the hardware/software stack (e.g. NVIDIA TensorRT, Intel DeepSparse). Many LLMs can run in int8 with minimal quality drop using techniques like **LLM.int8() (Dettmers)** which only quantize certain matrices and us ([Know your dataset](https://huggingface.co/docs/datasets/en/access#:~:text=There%20are%20two%20types%20of,for%20it%20to%20download%20completely))ers. This gives ~1.3–1.5× speedup and memory cut by 2×.
- Investigate 4-bit quantization for *very* large models or GPU-limited scenarios. Tools like **bitsandbytes** allow running models in 4-bit on GPU (with custom kernels). Combined with sparse attention, one could even attempt 3-bit or 2-bit in research contexts. Quality can suffer, so often 4-bit is paired with fine-tuning (see QLoRA in Topic 13).
- If extreme latency or cost-efficiency is needed, consider distilling the model. E.g., serve a distilled 6B model instead of the original 13B to hal ([Know your dataset](https://huggingface.co/docs/datasets/en/access#:~:text=There%20are%20two%20types%20of,for%20it%20to%20download%20completely))e smaller model might be 95% as good on target tasks for a fraction of cost. Some companies do this for generative models that will be served at scale, using the big model as teacher during development, then serving a distilled student.

**Resources:**

- **PyTorch 2.0 & Transformer Engine** – PyTorch 2 introduced `torch.compile` and improvements for FP16/BF16. NVIDIA’s Transformer Engine (used in Megatron) automates mixed precision and supports FP8 on Hopper GPUs. A blog example: *“FP8 halves storage and doubles speed compared to FP16, with minimal accuracy impact due to dynamic scaling”*.
- **Hugging Face Guide – Mixed Precision** – The HF Accelerate docs show how to load models in FP16 or BF16, and note that one should prefer FP16 on GPUs without native BF16, and BF16 on those with (since BF16 has a larger exponent and can be a bit more stable for very large activations). Also mentions using `autocast` for mixed precision context.
- **NVIDIA Developer Blog – QAT** – *“Achieving FP32 Accuracy for INT8 Inference Using QAT”*. Walks through quantizing a model to int8 and fine-tuning it to regain accuracy. Good to understand how QAT can make int8 essentially as accurate as FP32 for CNNs – similar ideas apply to transformers.
- **QLoRA Paper (Dettmers et al. 2023)** – While QLoRA is about fine-tuning (see Topic 13), it introduces a new quantization (NF4) that’s optimized for minimal loss. They show 4-bit quantized LLaMA-65B *with LoRA finetune* can match full 16-bit model performance. This implies that serving a 4-bit model is viable with the right approach. It’s a key reference for state-of-the-art quantization techniques.
- **HF Blogs on Distillation** – Hugging Face has blog posts or forum entries on distilling large transformers (e.g., distilling GPT-2 to smaller, or BERT to TinyBERT). They outline the process and trade-offs. One relevant example is Stanford’s Alpaca: they took OpenAI’s text-davinci (175B) outputs to train a 7B model – essentially distillation via generated data, which is an approach to compress capabilities for easier serving.
- **BitsandBytes Library** – This is the open-source library enabling 8-bit optimizers and 4-bit quantization for models. The documentation for bitsandbytes (by Tim Dettmers) explains how their int8 matrix multiplication works with minimal accuracy loss (by doing per-row quantization and unquantizing high-magnitude outlier rows). For a deeper understanding of how quantization is implemented efficiently, this is a great resource.

## 9. CUDA & Triton Kernel Optimizations (Fused Ops, Warp Management)

**Summary:** At the systems level, a huge part of LLM serving speed comes from optimized GPU kernels. **Fusing operations** means executing multiple computations in one kernel launch, reducing memory reads/writes and overhead. For example, in transformer forward pass, instead of separate kernels for matrix multiply, bias add, layernorm, etc., a fused kernel might do them all together on each tile of data. This improves throughput especially for *memory-bound* operations (like elementwise adds, softmax) by avoiding writing intermediate results to global memory. Modern libraries (PyTorch 2.x, TensorRT, JAX) automatically fuse many ops either via JIT or through libraries like FlashAttention (which fuses attention softmax + masking + matmul in SRAM).

**Triton** is a language by OpenAI for writing custom GPU kernels in Python that compile to PTX. It enables relatively easier development of fused kernels tailored to your model. For instance, a fused **softmax kernel** in Triton can outperform PyTorch’s standard softmax for certain matrix sizes by doing the reduction and exponentiation in one pass. Developers have written Triton kernels for things like layernorm, GELU, attention, etc., achieving speedups.

**Warp management** refers to writing kernels that effectively utilize the GPU’s 32-thread warps. Techniques include using **warp shuffles** and **shared memory** to let threads exchange data without going to global memory, and ensuring threads in a warp follow the same execution path (avoiding divergence). For example, a ([Big data? Datasets to the rescue! - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/en/chapter5/4#:~:text=Big%20data%3F%20Datasets%20to%20the,the%20entries%20in%20a%20corpus)) ([Know your dataset](https://huggingface.co/docs/datasets/en/access#:~:text=There%20are%20two%20types%20of,for%20it%20to%20download%20completely))ray) can be done efficiently with warp-level primitives that sum 32 values in a warp using registers (shuffle XOR operations), rather than e ([Know your dataset](https://huggingface.co/docs/datasets/en/access#:~:text=There%20are%20two%20types%20of,for%20it%20to%20download%20completely))ting to memory and reading again. Optimizing at this level gets close to device peak performance.

In context of LLMs:

- **FlashAttention** is a fused kernel that performs attention by tiling the computation to avoid explicit huge intermediate matrices, and uses on-chip memory to store partial results. It significantly speeds up attention (especially for long sequences) by being both fused and by managing reads/writes carefully (it avoids a lot of memory access) – making attention computation compute-bound rather than memory-bound. Many frameworks integrate FlashAttn now.
- **Fused MLP kernels**: The two linear layers and GELU in a Transformer MLP block can be fused. NVIDIA’s FasterTransformer and others do this. PyTorch 2’s Inductor will fuse elementwise ops into the GEMM when possible.
- **Tensor Cores**: Ensuring matrices are sized or padded to use 16x16 or 8x8 tensor core operations can give huge speedups. Kernel optimization includes arranging memory in **LDG (coalesced global loads)** and using **shared memory tiling** to feed tensor cores efficiently. This is often done by libraries (cublasLt has APIs to fuse bias and activation into gemm too).
- **Kernel autotuning**: For some ops, different launch configurations (block size, etc.) yield different performance. Systems like TensorRT and TVM will autotune or choose the best kernel. But in custom dev, one might manually tune occupancy (number of warps per block, etc.) to maximize use of GPU resources. For example, a kernel might use many registers per thread to reduce memory accesses, but that limits number of threads resident – finding the balance is an art.

In summary, expertise in CUDA kernel optimization can squeeze out additional performance in serving (especially for model architectures or hardware not well-covered by existing libraries). However, frameworks and compilers (Inductor, XLA, TensorRT) often handle a lot of this – knowing their principles helps in understanding why certain deployment gives better latency.

**Resources:**

- **Triton Language Tutorials** – The Triton docs have great tutorials like *“Fused Softmax”*. It walks through writing a kernel that computes softmax + normalization in one pass across each row. It explains benefits of kernel fusion for bandwidth-bound ops (like softmax) and shows how Triton code is structured.
- **CUDA Warp Primitives Blog (Nvidia 2018)** – Introduces warp-level instructions (like `__shfl_down_sync`) that allow threads in a warp to directly exchange values. The blog illustrates how to use these to write faster reductions and scans. It’s a foundational read for “warp management” – ensuring you maximize instruction parallelism within warps and use the fact that a warp executes in lockstep.
- **PyTorch 2.0 Inductor – Operator Fusion** – A Medium post *“How PyTorch 2.0 fuses kernels with torch.compile”* describes how Inductor takes an FX graph and generates fused loops in C++/Triton. It specifically discusses fusing elementwise ops and how it can generate CUDA**Resources (continued):**
- **PyTorch 2.x Inductor & Dynamo** – Under the hood of `torch.compile`: TorchDynamo captures the Python code into an IR, and TorchInductor generates fused GPU code, often via Triton for GPUs. The official PyTorch 2.0 announcement describes how Inductor uses Triton to generate fast kernels for multiple accelerators. This can fuse many small ops in Transformer forward passes automatically, yielding large speedups without manual kernel coding.
- **NVIDIA Medium – TensorRT and Kernel Auto-Tuning** – Explains how TensorRT during model conversion will perform *layer fusion*, *precision calibration*, *kernel auto-tuning*, and *graph optimizations*. For example, it fuses activation layers into matmul, chooses the best GEMM algorithm for the GPU, and removes redundant ops. It gives insight into how an automated compiler improves performance similarly to manual optimizations.
- **Kapil’s Talk on Fusing Kernels (PyTorch)** – Covers using `torch.compile` (with AOTAutograd and Triton) to fuse back-to-back operations. Real-world example: fuse layernorm + dropout + linear operations. Also touches on how to ensure memory alignment for vectorized loads.
- **OpenAI Triton GitHub** – The source and docs contain many examples of custom fused kernels (like FlashAttention’s Triton implementation, LayerNorm, etc.). Studying these can show advanced warp techniques, e.g., using shared memory to store a tile of matrix, then using warp-level ops to reduce for softmax.

## 10. Model Compilation & Graph Optimization for Inference (TorchDynamo, TensorRT, etc.)

**Summary:** Model compilation involves transforming the high-level model (PyTorch, TensorFlow graph) into a more optimized, lower-level representation before execution. For LLM inference, compilers aim to *reduce overhead and improve runtime*: by fusing operations, eliminating redundant computations, and leveraging target-specific libraries. Key tools:

- **TorchDynamo + TorchInductor (PyTorch 2.x)**: Together exposed via `torch.compile`, these capture a PyTorch model and compile it to an optimized code path. TorchDynamo hooks into Python to trace the model execution into an FX graph (even with dynamic shapes), then Inductor lowers it to efficient code (C++ or Triton for GPU). The result is that much of the interpreter overhead is removed and many ops get fused. Users saw up to ~2x speedups on transformer models with PyTorch 2.0’s compiler. It supports dynamic shapes and is still improving.
- **NVIDIA TensorRT**: A well-established C++ compiler/runtime for neural nets, especially on NVIDIA GPUs. For transformers, you typically convert the model (via ONNX or FX) into a TensorRT engine. The engine will have fused kernels and use lower precision (FP16/INT8) as configured. TensorRT does heavy graph optimization: *layer fusion, constant folding, precision lowering, kernel selection*. It will, for example, fuse consecutive matrix multiplies or a matmul followed by activation, choose the fastest kernel implementations for each layer, and remove any ops that are no-ops. The outcome is a binary blob that runs the model end-to-end very efficiently. TensorRT-LLM (as mentioned) extends this with support for generation and KV cache.
- **ONNX Runtime / TVM**: ONNX Runtime has an optimization engine for inference and can also use NVIDIA’s TensorRT as an execution provider. Apache TVM is another deep learning compiler that can optimize models for various hardware; it can auto-tune kernels specifically for your model and device. Some folks have used TVM to compile transformer models to GPUs or even CPUs and seen decent speedups, though it may require writing some custom relay passes for complex patterns.
- **XLA (Accelerated Linear Algebra)**: XLA is the compiler backend for TensorFlow (and JAX). It can also compile PyTorch models via the XLA bridge. It performs many graph optimizations and can generate very optimized code for TPUs and GPUs. While XLA was more aimed at training, it also benefits inference by fusing pointwise ops, etc. PyTorch/XLA isn’t commonly used just for inference (outside TPU), but JAX with XLA is – e.g., running a model in JAX and compiling it will result in a single fused executable that is often quite fast.
- **Graph optimization**: Beyond kernel fusion, compilers also optimize the computation graph: e.g., eliminate dead branches, *common subexpression elimination* (if the same calculation is done multiple times, compute once and reuse), and *reordering ops* when safe to improve memory access patterns. For LLMs, one important optimization is *operator reordering*: moving dropout and layernorm computations to places where they can be fused or folded (some compilers can fold dropout during inference as it’s no-op then).
- **Specialized graph optimizers**: Hugging Face’s `optimum` library and Intel’s Neural Compressor provide facilities to take a transformer model and apply transformations like merging consecutive linear layers or splitting large matmuls for better caching. These are narrower in scope compared to full compilers but can yield some gains.

In production, using a compiler like TensorRT can dramatically reduce inference latency (often 30-50% speedup for large models) but comes with complexity (long conversion times, less flexibility for dynamic input sizes, etc.). PyTorch 2.0’s approach is more seamless but might not yet hit the absolute performance of hand-tuned TensorRT engines. The trend is moving toward these compilers as models and hardware become more complex.

**Resources:**

- **PyTorch 2.0 Announcement (Dev Blog)** – Introduces `torch.compile` and the technologies (TorchDynamo, AOTAutograd, PrimTorch, TorchInductor). It explains how TorchDynamo captures an entire model, and Inductor generates code using OpenAI Triton for GPUs. They demonstrated ~1.5x speedups on HuggingFace Transformer models with just `torch.compile(model)`. A good high-level understanding of how dynamic graphs can be optimized ahead-of-time.
- **PyImageSearch – “What’s behind PyTorch 2.0: TorchDynamo and TorchInductor”** – A tutorial-style explanation of these components. Shows an example of a simple model before and after compilation, and discusses how Inductor fuses element-wise ops into the GEMMs.
- **TensorRT Official Docs** – NVIDIA’s docs (and Medium posts) outline the steps TensorRT takes: it lists optimization phases like layer fusion, precision calibration, kernel auto-tuning, and graph pruning. For instance, see *“Optimizing BERT with TensorRT”* where they show fusing MatMul+Add+Gelu, etc. Also, the **TensorRT Developer Guide** (Graph Optimizations section) is a detailed reference for all graph passes.
- **ONNX Runtime Transformers** – ONNX Runtime has a “Transformers optimization toolkit” (ORT Open Source) that performs many transformer-specific graph rewrites (like pre-fusing layernorm patterns, merging attention subgraphs). Microsoft’s blogs on ORT mention huge speedups for BERT and GPT-2 using these optimizations.
- **TVM Conference Talks** – There are talks/papers on using TVM to optimize GPT-2 and other models. TVM uses auto-tuning: it will try many variants of a kernel (e.g. different tiling sizes) on your hardware to find the fastest. One case study: optimizing a 2-layer transformer with TVM achieved close to cuDNN performance for those layers. It’s complex, but the **TVM 2022 paper “Octomizer”** demonstrates end-to-end optimizations on large models.
- **Hugging Face Optimum** – This library integrates with compilers like ONNX Runtime, OpenVINO, TensorRT, and provides scripts to quantize and compile HF models. The documentation and examples (like compiling a BERT to TensorRT) highlight real-world speedups and how to use them.

## 11. Fault Tolerance and Graceful Degradation in Model Serving

**Summary:** In production, an LLM serving system must be robust to failures – both system failures (node crashes, GPU OOMs) and model errors (e.g. a particular input triggers an error). **Fault tolerance** means the service should continue operating (perhaps in degraded mode) even if some components fail. **Graceful degradation** means if the system is overloaded or parts are unavailable, it should still provide a best-effort response rather than total failure. Key strategies:

- **Redundancy**: Run multiple instances of the model (possibly on different servers or GPUs) so that if one instance goes down, a load balancer can route requests to another. This is often done behind an API endpoint – e.g., 2 replicas of the model in Kubernetes, so if one pod dies, traffic goes to the other.
- **Health checks and auto-restart**: The serving system should monitor each model worker. If a process is unresponsive or crashes, it’s restarted. Frameworks like Kubernetes do this natively. At a higher level, the system might temporarily stop sending requests to a troubled instance (circuit breaker pattern).
- **Timeouts and fallback responses**: If generating a response exceeds a certain time or fails, the system can fallback to a simpler method. For example, if an LLM call fails, maybe return a pre-canned apology or a result from a smaller backup model. In critical applications, you might have a cascade: try the big model, if it fails or is too slow, use a smaller model or a rule-based system to ensure the user gets something.
- **Graceful GPU memory handling**: One common failure in LLM serving is OOM (out-of-memory) on GPU if too large a batch or context is requested. A graceful strategy could catch this and either: trim the input (with a warning), offload some context to CPU (slower but avoids hard crash), or route the request to a different instance with more memory (if available). The goal is to avoid process termination.
- **Transactional token generation**: During streaming generation, if the model server dies mid-stream, the client might be left hanging. To mitigate, some systems periodically flush partial results and have the client able to reconnect or another server take over. This is hard for stateful processes, but at least ensuring partial progress is sent out reduces impact.
- **Load shedding**: In extreme overload, rather than timing out every request (which users experience as failure), a graceful degradation is to reject or defer some requests early (shed load) so that those which are processed can still succeed. For instance, if QPS is 10× what the system can handle, it might immediately return an error or “try later” response to some fraction of requests – this is better than accepting all and timing out/failing all.
- **Logging and retry**: The system should log failures and possibly automatically retry a request if it’s safe to do so. For example, if a GPU execution fails due to a transient CUDA error, the orchestrator can retry that request on another GPU. As long as the request is idempotent (common in inference), this can turn a failure into just a slight latency hit.
- **Multi-region and DR (Disaster Recovery)**: At a higher level, for global services, have multiple data center regions. If one region goes down (power outage, etc.), traffic is routed to a backup region. This often entails having models deployed in both and a global load balancer (like Cloudflare, AWS Route53) switching traffic. It’s costly but important for mission-critical apps.

**Resources:**

- **TrueFoundry AI Gateway** – Advertises *“intelligent load balancing, failover, and automatic retries ensure seamless uptime and fault tolerance”*. Although a product blurb, it highlights industry practices: they mention automatic retries (likely if a model provider fails, they try another) and load balancing across a pool of models for high availability.
- **Microservice Fault Tolerance (MicroProfile)** – General patterns like Circuit Breaker and Fallback are documented (e.g., the MicroProfile Fault Tolerance spec, or Martin Fowler’s patterns). For example, a **fallback mechanism** could be returning a default answer or calling a simpler service if the main model call fails. In an LLM context, a fallback might be a smaller model or a template response.
- **OpenAI API status** – Not a formal resource, but observing how OpenAI’s API behaves: they have rate limits (to shed load), they return specific error codes when overloaded (429 or 503), encouraging client to retry after some time. This prevents the system from being overwhelmed and is a form of graceful degradation (the service says “too busy now” instead of just timing out).
- **Paper: “Adaptive Fault Tolerance for LLMs in cloud”** – Possibly the arXiv [61†L0-L8], though I haven’t read it, seems directly relevant. It likely discusses techniques to enhance reliability of LLM services in cloud environments. It might cover dynamic resource allocation upon failures.
- **GitHub Issue – fallback models** – HuggingFace transformers or others have had feature requests like *“support a pool of models with automatic fallback if one fails”*. Community discussions (e.g., linked GitHub or forum threads) sometimes outline how practitioners handle model provider outages by routing to another.
- **Netflix Chaos Engineering** – While about microservices, the philosophy is applicable: inject failures (kill a model instance randomly) to test if your system correctly reroutes and recovers. Ensuring that in a cluster of N model servers, N-1 can carry the load if one goes down (maybe at slightly higher latency) is a design goal – often achieved via over-provisioning and fast failover.
- **Kubernetes HPA/Cluster Autoscaling** – Documentation on auto-scaling can be considered: if load increases beyond capacity (threatening latency SLA), a *graceful* approach is to spin up more instances (if on cloud). This is more *scaling* than fault tolerance, but it prevents failure by being proactive.

## 12. End-to-End LLM Inference Pipeline Architecture

**Summary:** An end-to-end pipeline encompasses all stages from receiving a user’s request to delivering the generated text. A typical LLM serving pipeline might look like: **Client → API Gateway → Inference Service → Post-processing → Client**. Key components/stages:

- **Request Ingestion (Front-end)**: Often an API server (REST or gRPC) that clients connect to. This layer handles user authentication, rate limiting, and quickly queues or forwards the request to the backend workers. In some setups, this is a lightweight HTTP server (e.g. FastAPI or Node.js) that puts the request into a task queue for the model server. In others (like TGI), the HTTP server is built-in and directly interacts with the model.
- **Pre-processing**: The input might be raw text; it needs to be tokenized into IDs. This can happen either on the API server or the model server. Many systems do it on the model server to keep all ML logic in one place. Batch preparation happens here: multiple requests’ tokens might be padded and combined into a batch tensor. Also, things like converting to the right dtype, moving to GPU, etc., occur.
- **Inference core**: This is the actual model forward passes. For a text generation request, it usually involves an iterative loop generating token by token (unless using special decoders). The core might interact with caching layers (for past KV) and manage the generation stopping criteria (max length or end-of-sequence token). This part is compute-heavy and runs on GPU (or TPU). It may be distributed across GPUs as discussed. The pipeline here might also involve *multiple stages* if using pipeline parallelism (each stage receiving activations from previous). In that case, the pipeline is a series of forward passes across devices.
- **Post-processing**: After generation, the output token IDs are converted back to text (detokenization). Also, any final formatting (maybe combining with the prompt for context, etc.) is done. If the application has additional steps (like stuffing the answer into a larger response JSON, adding reference tags, etc.), that happens here. If streaming, this actually interleaves with the inference core – tokens are post-processed and sent to client incrementally.
- **Return to Client**: The response (full text or stream of tokens) goes back through the API gateway to the user. The pipeline should ensure correct ordering (if using async or multithreading, keep track of which response corresponds to which request). Logging of the request/response can also be considered part of pipeline (for audit or analytics).
- **Monitoring and Logging**: End-to-end pipeline includes recording latency of each stage, any errors, and possibly the content (with PII considerations) for quality monitoring.
- **Supporting Systems**: For a complete pipeline, additional pieces like a *vector database* or external knowledge store might be consulted (for Retrieval-Augmented Generation). In that case, the pipeline might first do a vector search (embedding the query, searching DB) before the model inference, and feed the results into the prompt. Those steps add complexity: e.g., “Client -> vector DB -> combine context -> model -> response”. Each sub-step (embedding model, DB lookup) has its own serving considerations. But the core remains orchestrating these sequentially or in parallel and merging results.

Crucially, the pipeline should be **scalable and decoupled**: the API layer can scale independently from the model workers. Often a message queue (like RabbitMQ, Redis, or just an async queue in memory) is used between API and inference workers to buffer requests and allow batching. Systems like Ray Serve or Kubernetes-based microservices help manage this – Ray Serve, for instance, can automatically batch calls to a deployment and has backpressure if the queue grows.

**Resources:**

- **Chirag Dave’s Medium (2025) – Production LLM Pipeline** – A step-by-step guide that maps out an end-to-end pipeline. It describes: Step 1 Fine-tuning, Step 2 Optimization, Step 3 Inference Server setup, Step 4 API Server deployment. In Step 4, they detail the API server’s role: exposing endpoints, validating/preprocessing input, and batching requests for the inference backend. This matches real-world architecture: a separate API gateway that feeds the inference engine. It also enumerates inference server options (vLLM, TensorRT-LLM) in Step 3, which shows how the pipeline components split (API vs Model server).
- **Decoding ML Substack – “LLM & RAG inference pipelines”** – This likely discusses a microservice approach: where you have one service for retrieval (RAG) and another for generation, etc., and how to compose them. It stresses separating business logic from ML logic (e.g., one layer for orchestration that calls an LLM microservice). Good for understanding modular pipeline design.
- **Hugging Face Inference Endpoints** – Their docs show that an incoming request goes through a load balancer to a dedicated container running the model. In a sense, the pipeline is trivial (direct), but the interesting part is how they handle streaming: the container streams tokens back through an API Gateway. Looking at HF’s text-generation-inference repository, the architecture section explains how the request thread handles token generation and yields partial outputs via server-sent events. This is a concrete example of end-to-end flow for streaming.
- **Ray Serve** – Ray Serve’s documentation is a resource on how to build an inference pipeline (ingress -> router -> worker). It natively supports batching. Ray Serve examples for LLM (some blog posts exist) illustrate sending requests to a central queue where they get batched. The *“Serve Multi-Model Pipeline”* tutorial shows how to chain deployments (e.g., one deployment for retrieval, one for generation).
- **KServe / TorchServe** – These are model serving frameworks that allow defining pre/post-processing and the model execution separately. For instance, TorchServe lets you define a “handler” where you can override preprocess and postprocess. Studying a TorchServe handler for a transformer model can give insight: e.g., it will load a tokenizer in preprocess, do `model.generate` in inference, then decode in postprocess. This is essentially the pipeline in one process, but logically separated.
- **System Design Articles** – General system design blogs on building scalable chat services (not specific to LLM) cover components like API gateway, scaling web servers vs workers, using task queues – which all apply here. An example is Slack’s backend for messaging: conceptually similar pattern (ingest -> queue -> worker -> respond), which is useful to draw analogies.

## 13. Parameter-Efficient Training: LoRA, QAT, etc.

**Summary:** Parameter-efficient training methods enable fine-tuning or adjusting LLMs without updating all ~Billion weights, which is useful both for *training infrastructure* (less compute needed) and sometimes for *serving* (smaller changes can be applied on the fly). Key techniques:

- **LoRA (Low-Rank Adaptation)**: As introduced by Hu et al. (2021), LoRA freezes the original model weights and learns small low-rank matrices that approximate the weight updates. For each large weight $W \in \mathbb{R}^{d \times k}$, LoRA learns $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ such that the effective weight becomes $W + \Delta W$ with $\Delta W = A B$ (where $r \ll d,k$). This drastically reduces trainable params (r is maybe 4 to 128 instead of thousands), e.g., LoRA on GPT-3 175B reduces trainable params by ~10,000×. The beauty for system design: these LoRA matrices (often only ~0.1% of original model size) can be stored and applied at runtime to modify the model’s behavior, without having to deploy a whole new model. For serving, one can keep a base model and multiple LoRA “modules” for different tasks, loading and merging them as needed (merging LoRA weights into base on the fly is efficient). This supports quick personalization or multi-tenancy of one model for different domains. LoRA adds **no extra inference latency** if merged (it’s just new weights), and if kept separate, it’s just an extra matmul which is negligible.
- **Quantization-Aware Fine-Tuning**: (Quant-Aware Training, QAT) – This involves training the model (or fine-tuning) with simulated low precision (int8 or int4) so that the model learns weights that will work well when quantized. Instead of training all weights, one might combine QAT with LoRA: e.g., do LoRA fine-tuning while the model is in 8-bit mode. The result is a LoRA that, when applied to an 8-bit model, yields good accuracy. This is effectively what **QLoRA** does: it keeps the base model weights in 4-bit precision, and optimizes LoRA adapters in 16-bit, achieving full 16-bit performance after training. Parameter-efficient since only LoRA params (low-rank matrices) are learned, and memory-efficient by quantizing everything else. QAT generally ensures that when you later serve the model quantized, it behaves as if it was trained that way, often recovering accuracy lost by naive quantization.
- **Adapters and Prefix Tuning**: Apart from LoRA, there are **Adapter layers** (Houlsby et al. 2019) – small feed-forward MLPs inserted in each Transformer block, trained while freezing the main model. These add some latency (extra layers), but small. **Prefix Tuning** (Li & Liang 2021) keeps model weights fixed and instead optimizes a set of *virtual tokens* or prefix vectors that are prepended to the input at each layer. This doesn’t change model arch, just extends the context with learnable embeddings. It’s parameter-efficient (only a few thousand prefix tokens learned) and at inference you simply concatenate the learned prefix to the prompt (so overhead is negligible, just a slightly longer prompt). This is great for domain adaptation without modifying the model weights at all.
- **RLHF and reward models**: While not exactly “parameter-efficient” in the sense of fewer trainable params, RLHF fine-tuning typically is done on a model that might be partially frozen or with low-rank updates to avoid catastrophic forgetting. OpenAI’s InstructGPT, for example, fine-tuned GPT-3 with human feedback; one could incorporate LoRA during RLHF to only adjust certain directions.
- **Why it matters for system design**: If you need to support many fine-tuned variants of a base model (for different clients or tasks), parameter-efficient methods let you do that without deploying N full copies of the model. You can deploy one 20B base and have N small adapter files, loading an adapter on demand per request. For example, a multi-tenant service might keep user-specific LoRA diffs and apply them per user’s requests – far more feasible than N separate 20B models. Also, training infrastructure benefits: fine-tuning a 65B with LoRA on a single GPU (as QLoRA showed) is possible, whereas full fine-tuning would require 8+ GPUs and a lot more memory.

**Resources:**

- **LoRA Original Paper (Hu et al. 2021)** – *“LoRA: Low-Rank Adaptation of LLMs.”* Explains the method and reports that LoRA fine-tuned GPT-3 175B on tasks like SST-2 (classification) matched full fine-tuning quality with only 0.1% of parameters trained. It also points out no additional inference latency and the ability to *merge the LoRA weights into the original model for deployment* (which essentially applies the low-rank update to the weight matrix permanently).
- **LoRA Implementation (GitHub loralib)** – Many practical notes on how LoRA is applied (usually to query/key/value weight matrices in Transformers, which are big dense layers). Also HuggingFace’s PEFT library documentation – it supports LoRA on HuggingFace transformers and shows examples combining LoRA with bit quantization for efficient fine-tuning.
- **QLoRA Paper (Dettmers et al. 2023)** – *“QLoRA: Efficient Finetuning of Quantized LLMs.”* A major result enabling a 65B model fine-tune on a single GPU by quantizing to 4-bit and using LoRA (rank 64). They achieved 99% of ChatGPT performance on a benchmark with 24 hours of training. This is a paradigm shift for training infrastructure – it means instead of needing an A100 cluster to fine-tune a large model, one can do it on a beefy single machine. For serving, the benefit is that the resulting model remains 4-bit weights + LoRA – which is super memory-efficient to host. The paper also introduces NF4 quantization and double quant that might interest those building training systems.
- **Adapter Fusion (Pfeiffer et al. 2020)** – A technique where multiple learned adapters (for different tasks) can be combined. In a serving scenario, this could allow merging knowledge from multiple fine-tunings. Not widely used yet for LLMs, but conceptually important: parameter-efficient pieces that can be composed at inference time.
- **Hugging Face PEFT Library** – Documentation and blogs (e.g., “Fine-tuning 30B models on Colab with LoRA”). They also have tutorials for **Prefix Tuning** and **P-Tuning** (learned prompt embeddings) which are even lighter-weight than LoRA. Good to know trade-offs: prefix tuning doesn’t change model weights at all (so very safe to deploy), but might be slightly less efficient in quality per parameter compared to LoRA.
- **System distillation** – There’s an angle of using these methods for *system-level adaptation*. E.g., LoRA can be used to fine-tune a model to reduce toxic output without full retraining (just train a LoRA on a dataset of toxic vs non-toxic responses). OpenAI has hinted at use of such techniques for alignment. For an interview, mentioning that you’re aware of how RLHF or bias mitigation could be done via small parameter updates (instead of retraining everything) shows understanding of maintaining models over time.

## 14. Checkpointing, Sharding, and Model Loading at Scale (Training & Inference)

**Summary:** Large models require careful strategies for saving/loading and distributing weights across devices:

- **Checkpointing (Training)**: During training, checkpointing means periodically saving the model state to disk (to recover from failures or pause). For multi-GPU training (DDP, FSDP, etc.), each GPU might only have *sharded* weights (e.g., under ZeRO, each GPU holds a fraction of each layer). Two approaches: **Full checkpoints** (gather all shards to one file – expensive for very large models), or **sharded checkpoints** (each rank saves its shard). FSDP/ZeRO often do sharded checkpoints by default. For fault tolerance, one might checkpoint every N iterations – but this can be many GBs. An emerging idea is **continuous checkpointing** where layers are written to disk as soon as updated (like a write-ahead log), but that’s more research. The main trade-off: how often to checkpoint (frequency) vs training speed and storage IO.
- **Sharding (Inference)**: If a model is too large for one node’s memory, it must be sharded. **Model parallelism** at inference time means splitting weights across GPUs (within a node or across nodes). To load a sharded model, one can either have a single checkpoint with all weights and load slices on each GPU (needs coordination), or save separate shard files for each device. HuggingFace Accelerate’s `device_map` approach can automatically load parts of the model on different devices. For multi-node, usually one launches the process on each host and uses a distributed loader (e.g., using PyTorch’s `load_checkpoint` that knows how to read global checkpoint and scatter to ranks). The **model parallel initialization** is critical – ensuring each GPU gets the correct slice of each tensor. Many frameworks handle this (Megatron-LM’s loader, DeepSpeed’s engine, etc.).
- **Lazy Loading & Memory Mapping**: When models are huge (tens of GB), loading can take minutes. Memory-mapped file formats like **Safetensors** allow loading “lazily” – not actually reading the weights into RAM until needed, and possibly directly using disk-paged memory. Safetensors is a safe (no code execution) and fast format that supports partial loading. For example, if using CPU offloading, one might memory-map the weights and only transfer to GPU when layer executes (Accelerate does this with `load_checkpoint_and_dispatch` where disk-stored weights are paged in on-the-fly). This can drastically reduce peak memory during load and enable loading models larger than RAM (by keeping most on disk until use). The drawback is potential latency hits when a page is loaded mid-inference, but if using high-bandwidth SSD and good scheduling (prefetch next layers’ weights), it can work well. Amazon’s **FastFileMode** for SageMaker (mentioned in HF forums) also memory-maps to start inference quickly without full load.
- **Parallel Loading**: To utilize multiple IO threads or nodes, frameworks often load different layers in parallel. PyTorch’s `torch.load` can be made multi-process, or one can split the checkpoint into multiple files and have each rank read its file concurrently. The HuggingFace `petals` project (distributed inference) had to coordinate loading 100B model across many machines – they used concurrent downloading of shards. In multi-node HPC training, often each rank reads only its shard from a shared storage, avoiding one node reading everything then broadcasting (which would be slower).
- **Format considerations**: Traditional PyTorch checkpoints are pickle-based (not safe and not memory efficient). `Safetensors` has become popular for large models – it’s zero-copy (memory mapped) and secure. Also, some use **HDF5** or **NPY** for weights. The key is to choose a format that allows efficient partial loading.
- **Resuming training (Hot restart)**: Checkpointing includes not just weights but optimizer states, RNG states. For massive models, optimizer state (like Adam moments) can be 2× the model size. Zero/Partitioned optimizers shard these too; saving them is heavy. Some systems do “optimizer offload checkpoint” – storing optimizer state to slower storage, or not at all (e.g., DeepSpeed ZeRO-Infinity can drop CPU optimizer states to reduce checkpoint size). For inference, this is less relevant, but for training infrastructure it’s crucial to be able to resume from a checkpoint reliably – meaning all shards align perfectly to continue training. There have been cases where a distributed checkpoint is corrupt or mismatched and cannot resume – robust checkpointing code (with versioning, hashes) is needed for multi-petabyte model states.
- **Shard management at inference**: Suppose you have a model sharded on 4 GPUs. If one GPU goes down, how to recover? Ideally you’d have a checkpoint so another GPU or node can spin up with that shard’s weights. In practice, redundant deployment (the same model replicated in another set of GPUs) is the solution, since live re-loading 40GB shard on the fly is slow. But one could maintain a “hot spare” instance ready to take over.

**Resources:**

- **DeepSpeed Checkpointing Docs** – Explain ZeRO’s model state sharding. E.g., ZeRO Stage 3 saves each rank’s shard separately by default, and provides utilities to gather a full checkpoint if needed. The DeepSpeed wiki on Checkpointing discusses how to save and load efficiently in sharded training. It’s instructive to see how they name files (`mp_rank_*.pt` etc. for model parallel shards).
- **PyTorch FSDP (Fully Sharded Data Parallel)** – PyTorch’s native FSDP has a checkpointing feature where you can choose to save `state_dict()` in full (auto-gathered) or in sharded form. The official tutorial or blog *“DeepSpeed ZeRO vs PyTorch FSDP”* might cover checkpointing. Important detail: saving full weights can OOM, so FSDP can instead save shards and include metadata so that loading knows how to map them.
- **HuggingFace Accelerate** – They have a function `load_checkpoint_and_dispatch` which can load a giant model across multiple GPUs or CPU+GPU from a given checkpoint with `device_map`. The Accelerate docs “Big Model Inference” show examples of sharded loading (even from Hub, where weights are fragmented). They also illustrate offloading: putting some layers on CPU or disk with hooks to load on-the-fly. This is a highly relevant read for inference loading strategies.
- **Safetensors format** – The Medium post *“Safetensors: fast, safe tensor serialization”* explains that it avoids pickle and supports memory mapping for zero-copy load. It specifically notes 3x faster load for BERT thanks to eliminating Python overhead and using mmap. Many large model repos now provide `.safetensors` checkpoints which users can load much faster (and safer). The Dev.to link provided reiterates these benefits and how safetensors ensures alignment for vectorized reads, etc..
- **Case Study – GPT-3 Loading**: Although not public, some info from EleutherAI’s GPT-NeoX or Microsoft’s MT-NLG suggests that loading a 100B+ model took tens of minutes from disk. They likely had to partition the checkpoint into parts to parallelize. If any blog or paper (“DeepSpeed Megatron inference”) touches on how to reduce spin-up time (maybe by quantizing weights on disk or by pipeline parallel preload), that’s useful.
- **Model parallel libraries** – Megatron-LM (NVIDIA) can save and load models in model-parallel shards. Their README might mention the procedure to load a model across N GPUs (giving an example command). It’s usually: each rank reads its partition of each layer from the checkpoint.
- **Lightning Fabric & Hivemind** – There are tools for collaborative training (Hivemind) that chunk models and share pieces – some documentation might cover how they checkpoint, but this is niche.

## 15. Dataset Streaming, Memory-Mapped Data, and Training Data Systems

**Summary:** LLM training typically uses massive datasets (multi-terabyte text corpora). Efficiently feeding this data to the GPUs is a challenge. Solutions involve streaming data directly from disk or remote storage, and using memory mapping or smart caching to avoid loading everything into RAM at once. Key aspects:

- **Streaming Datasets**: Instead of downloading an entire dataset to local storage, you stream it – read example by example (or chunk by chunk) on the fly. HuggingFace Datasets supports `streaming=True`, which will sequentially fetch data (from Hub or a URL) as you iterate. This is crucial for web-scale data: you can start training immediately and not worry about disk space. It does mean your training pipeline must tolerate slower IO and potentially lack of random access (since streaming yields an IterableDataset). Typically, streaming is combined with shuffling buffers (to simulate random order).
- **Memory Mapping (mmapping)**: If your dataset is stored in a file (e.g., in Arrow format as used by HF Datasets), memory mapping allows loading data examples on-demand without reading the entire file into memory. For instance, HuggingFace datasets uses Apache Arrow under the hood, which memory-maps the data file. Thus, you can have a 100GB dataset file and when you iterate, it will load each batch from disk as needed, using the operating system’s page cache to manage memory. This greatly reduces memory usage – you can handle datasets larger than RAM. It also leverages the OS for caching: recent or frequently accessed portions might stay in RAM automatically.
- **Sharding the dataset**: In distributed training across nodes, each node should get a unique portion of data each epoch. Data loaders typically use an “epoch counter + seed” to offset/shard the data. With streaming, this can be tricky, but frameworks solve it by splitting by data index or using modulo arithmetic to assign examples to ranks. Proper sharding prevents redundancy and ensures full dataset coverage with multiple workers reading in parallel.
- **Throughput considerations**: The data pipeline must keep GPUs busy. If IO is too slow, GPUs starve. Solutions:
    - Use multiple data loader worker threads/processes to prefetch data. For example, PyTorch DataLoader with `num_workers>0` will spawn workers to read and process data (e.g., tokenizing text) in parallel, filling a queue. In large-scale setups, separate nodes might handle data preparation.
    - Use fast storage – NVMe SSDs locally, or memory-mapped network storage (like AWS FSx for Lustre, which can stream at GB/s scale). Some even copy data to local SSD from a data lake as a first step.
    - **NVIDIA DALI** (Data Loading Library): This is a library that accelerates data preprocessing by using GPU or optimized CPU code. It’s more for images (JPEG decode, augmentations) but can be used for text (e.g., it could theoretically handle tokenization on GPU). For LLM training, DALI is less common, but if doing things like audio or video, it’s very useful.
    - **Tokenization pipeline**: One bottleneck is tokenizing raw text (esp. for Python-based tokenizers). Many projects preprocess the entire dataset into token IDs once and save that (so training just reads token IDs). For example, The Pile (an 800GB text corpus) is often distributed as pre-tokenized binary files (maybe in HF Arrow or TFRecord format). This shifts the cost to a preprocessing step but makes training IO much lighter (just reading int arrays). If not pretokenized, one might use fast C++ tokenizers (like HuggingFace’s FastTokenizer in Rust) to speed it up on the fly.
- **Dataset storage format**: Common formats for big data: **Apache Arrow** (used by HF Datasets, memory-mappable, columnar), **TFRecord** (TensorFlow’s binary record format, supports streaming reads and can be sharded), **WebDataset (tar archives)** – WebDataset stores data in tar files (potentially with compression) and lets you stream read and shard by splitting tar into chunks. E.g., you could pack thousands of text samples per file and stream those. Each has trade-offs; Arrow is great for random access and memory map, TFRecord is simple and works well with sequential access + prefetching, WebDataset is convenient for web-scale (just HTTP range requests into tar).
- **Scaling data systems**: If training on a cluster, consider using a distributed filesystem (like Lustre, NFS, or Hadoop HDFS) to have one copy of data accessible by all nodes. Otherwise, you replicate data to each node’s local disk (which costs time/storage). Many HPC setups rely on a fast shared filesystem and count on caching (each node might cache frequently accessed chunks locally). Cloud solutions sometimes use object storage (S3) and stream from it, possibly caching to local disk.
- **Online data augmentation or sampling**: For LLMs, not much augmentation besides maybe shuffling or filtering. But note: some pipelines might filter on the fly (e.g., discard long lines, or do temperature sampling from mixture of sources). These operations should be efficiently implemented (vectorized or in C++ if possible) as they happen for each sample.
- **Monitoring and reproducibility**: When streaming data especially, ensuring you see the full dataset and can reproduce training can be tricky (if the source changes or if streaming order is nondeterministic). It’s common to fix random seeds and perhaps save the shuffled index order per epoch to be able to restart consistently.

**Resources:**

- **Hugging Face Datasets Course – Big Data** – Describes how HF Datasets uses memory mapping and streaming to handle big datasets. Notably: *“treats datasets as memory-mapped files, and for really big datasets that don’t fit disk, use IterableDataset to stream without downloading completely”*. This highlights the two modes: `Dataset` (memory-mapped, random access) vs `IterableDataset` (streaming).
- **WebDataset (github)** – Documentation explains how you can efficiently stream from tar files with compression, and it handles sharding across workers by splitting by file. Many large-scale training efforts (esp. multimodal) have used WebDataset. There are blog posts by the creator (John Hebert) discussing throughput achieved (like 10GB/s reads, if using multiple threads).
- **NVIDIA DALI** – NVIDIA blog “Accelerating data preparation for DL with DALI” shows how using DALI can remove CPU bottlenecks. For NLP, DALI is not widely used, but it could be; if asked, one can mention DALI primarily helps with image/video but conceptually a similar approach (offloading data prep to C++/GPU) could apply to text (e.g., RaggedTensors, etc.).
- **TensorFlow input pipelines** – Even though we are focusing on PyTorch likely, TensorFlow’s tf.data is an advanced streaming dataset API. Their Performance Guide (TF Data performance) covers prefetching, parallel interleave (to read from multiple files concurrently), caching on disk vs memory, etc. Many principles generalize to any pipeline: e.g., always prefetch a few batches (`DataLoader(..., prefetch_factor=...)` in PyTorch), use asynchronous reads.
- **OpenWebText & The Pile** – These are example large datasets used for LLMs. The Pile was provided as 800GB of text in 30+ files. They recommended using their indexing for random access or streaming sequentially. Reading any documentation on The Pile can reveal how they expected users to load it (I recall they provided an index for direct seek to each document for random access if needed). This is real-world example of handling a multi-source massive dataset.
- **Scaling Laws (not data pipeline)** – Just a note: sometimes interviews might ask how to handle *continual training data streaming* (like training on freshly generated data). The pipeline would then need to incorporate new data on the fly, perhaps by merging streams or periodically updating the dataset. Ensuring the system can incorporate new data without full retraining from scratch is more of a machine learning question, but the system design side is how to flow that data in reliably (maybe via a Kafka stream or something feeding into training).

---

Each of these topics equips you to discuss a crucial aspect of large-scale LLM systems. By understanding and citing these concepts – from how continuous batching boosts throughput, to how LoRA fine-tuning can update a model with minimal overhead – you can demonstrate a well-rounded mastery of LLM inference and training design. This knowledge will allow you to intelligently reason about system design decisions in an interview setting,
covering both the software (algorithmic) optimizations and the hardware/resource considerations.





# **Parallelism Techniques for Large-Scale LLM Inference**

## **Data Parallelism**

*Data parallelism replicates the entire model on each GPU (each colored box) to process different input data in parallel.*

**How it works:** Data parallelism involves deploying multiple copies of the model on different GPUs or nodes. Each model replica handles a different portion of the incoming requests or batch independently, much like having multiple instances of the same microservice handling separate users . All GPUs have the full model loaded, so this technique does **not** split the model itself; instead it splits the *data*. For example, if you have 4 identical GPUs each with a copy of a 7B parameter model, you can send different input sequences to each GPU concurrently and get 4 times the throughput (in ideal conditions). Importantly, this does **not** help if a single inference is too large to fit on one GPU’s memory – data parallelism assumes the model *does* fit in one device’s memory .

- **Benefits:**
    - **Linear throughput scaling:** Increases overall serving throughput almost linearly with the number of replicas, since each GPU can handle a separate request in parallel . This is great for serving many independent queries simultaneously (e.g. many users at once).
    - **No model coordination latency:** Each inference runs in isolation on one GPU, so there’s no cross-GPU communication needed during forward passes. This keeps per-request latency low (each request is just like running on a single GPU) and avoids networking overheads during inference.
    - **Simplicity:** It’s straightforward to implement – essentially running multiple model instances – and is compatible with existing inference servers. Scaling to more GPUs or nodes is as simple as adding more replica instances.
- **Trade-offs:**
    - **Memory duplication:** Every GPU must hold a complete copy of the model weights. This is wasteful for very large models and can become impossible if the model is larger than a single GPU’s memory (e.g. a 175B model cannot be served on one 24GB GPU by data parallelism alone) . It also means using N GPUs requires N times the memory for weights.
    - **No single-query speedup:** Data parallelism does **not** reduce the latency for a single input. One query can only run on one replica, so if you need to accelerate *one* long or complex inference (rather than many in parallel), data parallelism doesn’t help. It’s geared toward throughput over latency.
    - **Diminishing returns:** If the workload (number of concurrent requests) is low, many model replicas will sit idle – scaling out beyond the demand just wastes resources. And in multi-node scenarios, distributing requests adds slight overhead (e.g. load balancing, network transfer of input/output), though typically minor compared to the inference compute.
- **Scalability:** Data parallelism scales *horizontally* across GPUs and even across multiple machines. In theory, if you have 100 identical GPUs, you can serve ~100x more requests per second. In practice, near-linear scaling is achievable as long as the external factors (like request dispatch and network I/O) are not bottlenecks. Unlike model-sharding techniques, data parallel replicas don’t need high-speed interconnects between GPUs because they run independently – this makes it suitable for scaling across nodes with just standard networking. However, each replica still needs to fit the model in memory, so extremely large models might require model parallelism instead of or in addition to data parallelism.
- **Use cases:** Data parallelism shines for **multi-user inference serving** and high-throughput APIs. For example, a deployment of a 13B model that easily fits on a GPU can be replicated 8 times to handle 8 concurrent requests with minimal latency impact, which is common in production systems . It’s also used in batch inference: a large batch can be split so each GPU processes a chunk of the batch in parallel, then results are combined. However, data parallelism fails when the model is too big for one GPU or when one needs to speed up a **single** inference beyond the capability of one device – in those cases, model or pipeline parallelism is needed.

### **Interview Questions – Data Parallelism**

1. **Throughput vs. latency:** If you needed to increase the *throughput* of an LLM service (serving many queries per second), why is data parallelism a good choice? And conversely, why does it fail to improve the *latency* of a single query?
2. **Memory overhead:** In an inference deployment, what are the memory implications of using data parallelism with a 30B parameter model across 4 GPUs? How might this influence your decision to use data parallelism or not?
3. **Scaling across nodes:** When scaling data parallel inference to multiple machines, what network considerations arise (for input/output handling or model updates) even though the model replicas don’t directly communicate during inference?
4. **Combining parallelism:** If a model barely fits on one GPU, can you simply use data parallelism to leverage two GPUs for one request? Why not, and which parallelism technique would you consider in that scenario?
5. **Use-case judgment:** Imagine a scenario with sporadic, heavy single-user queries (long, expensive prompts) rather than many concurrent users. Would you invest in data parallel replicas or another approach to handle this load? Explain your reasoning.

## **Tensor (Model) Parallelism**

*Tensor parallelism (a type of model parallelism) splits each weight matrix or tensor across multiple devices. In this illustration, matrix B is split into two parts (green and blue) and multiplied with A in parallel; partial results are then combined (all-gather) to form the final output C  .*

**How it works:** Tensor parallelism (also called intra-layer model parallelism) partitions the computations *within* each model layer across multiple GPUs . Instead of replicating the whole model, different GPUs hold different slices of the model’s weight tensors. During inference, a single forward pass is **distributed**: each GPU computes its fragment of the layer and the partial results are then aggregated to produce the same output as the full layer. For example, if a transformer’s feed-forward layer has a large weight matrix, one can split that matrix into 2 or 4 chunks and place each chunk on a different GPU. Each GPU multiplies its chunk by the input simultaneously, and then the results are summed or concatenated to get the final output  . Similarly, for multi-head attention, different heads (or groups of heads) can be assigned to different GPUs to be computed in parallel . In essence, tensor parallelism “slices” the tensor operations along a dimension and uses an all-reduce or gather operation to combine outputs at the end of the layer. This allows a single huge model to be spread across multiple devices *within* each layer.

- **Benefits:**
    - **Enables larger models:** Tensor parallelism makes it possible to serve models that are too large for one GPU’s memory by dividing the model’s parameters. Each GPU only stores a fraction of the weights (e.g. 1/4 of each large matrix if using 4-way parallelism), reducing per-GPU memory usage proportionally . This was key to deploying models like GPT-3 on clusters of GPUs.
    - **Potential speedups:** By sharing the compute of a layer across multiple GPUs, you can shorten the wall-clock time for that layer’s forward pass. Each GPU handles a smaller matrix multiplication, which might complete faster, and the results are merged. In an ideal scenario with fast interconnect, splitting a compute-heavy layer across N GPUs could approach an N× speedup for that layer. This can reduce latency for a single large inference if the GPUs are efficiently utilized in parallel.
    - **Transparent model output:** The outputs are the same as if computed on a single device (just assembled from parts), so accuracy isn’t affected. It parallelizes the math without approximating it. From the outside, it’s still one model – just partitioned under the hood.
    - **Mix-and-match with other parallelism:** Tensor parallelism can be combined with data parallelism or pipeline parallelism. For example, you might shard the model across 2 GPUs (tensor parallel) and also have 2 replicas of that setup (data parallel) to serve more queries. It’s a flexible building block for scaling.
- **Trade-offs:**
    - **Communication overhead:** Splitting a layer means that at some point GPUs must exchange data (e.g. send their partial outputs to each other) to produce the final result . Typically this involves high-speed GPU-to-GPU communication (like an all-reduce or all-gather after every layer). This overhead can significantly hurt latency if the interconnect is slow. In fact, the network bandwidth between GPUs often becomes the bottleneck for tensor parallel scaling . Using NVLink or NVSwitch (on-node) or Infiniband (across nodes) is usually required to make it performant.
    - **Synchronization:** All parallel GPUs must wait for each other at the synchronization points each layer. The inference can only move at the speed of the slowest participant (e.g., if one GPU is momentarily slower or has more load, it delays the others). This means latency doesn’t always scale linearly – adding more GPUs might give diminishing returns if communication and synchronization costs grow.
    - **Memory overheads:** While each GPU stores only a slice of the big weight matrices, some parts of the model might still be replicated on all GPUs (for example, small layers like layernorm or embeddings might be kept full on each for simplicity, or you need memory for communication buffers). There’s also overhead in storing partial activations that must be exchanged. So memory per GPU is reduced, but not by exactly the factor of parallelism in all cases.
    - **Complex implementation:** Using tensor parallel inference often requires a custom runtime or libraries (Megatron-LM, FasterTransformer, etc.) that know how to split the computations and do all-reduce. It adds engineering complexity compared to running a model on one device. Load balancing the split (deciding how to partition tensors) can also be complex for different layer types.
- **Scalability:** Tensor parallelism is effective up to a point. Typically, we see good scaling for moderate numbers of GPUs (e.g. 2, 4, 8) especially with excellent interconnects. For instance, splitting an attention or MLP across 8 GPUs can work if they share NVSwitch, but beyond that, the communication overhead can dominate. Each additional GPU contributes less speedup if the network traffic (which grows with more shards) becomes the limiting factor . Also, some operations can’t be split arbitrarily – if a layer is very small, splitting it might not be worthwhile. In practice, large LLMs often use a combination of *tensor parallel groups* (e.g. groups of 8 GPUs) to shard the model, possibly combined with pipeline parallel groups for further scaling. The key scalability consideration is network bandwidth: a high-bandwidth, low-latency interconnect (such as NVLink or PCIe5 or better) is needed to make tensor parallel inference across GPUs efficient. Multi-node tensor parallelism is possible but requires extremely fast network (e.g. InfiniBand) and even then may incur significant latency per token.
- **Use cases:** Tensor parallelism is commonly used when a model just **cannot fit** on one GPU’s memory. For example, serving a 175B parameter GPT-3 model might involve splitting each layer across 8 GPUs (Megatron-LM style) so that each GPU holds ~22B parameters. It also finds use in accelerating inference for very large batches or very large models – e.g. big batch transformer scoring can be sped up by distributing the matmuls. Real-world deployments of ultra-large models (like those by OpenAI, Microsoft, etc.) rely on model parallelism under the hood. However, if the model *does* fit on one GPU, pure tensor parallelism is usually avoided for latency reasons – one GPU is often faster than two with communication. Thus, it shines primarily in the **“make it fit”** scenario (or the “use more GPUs because one would be too slow” scenario) and is often combined with pipeline parallelism (model sharding by layers) for models that are both deep and wide.

### **Interview Questions – Tensor/Model Parallelism**

1. **Networking bottleneck:** In an inference setup using tensor parallelism across 4 GPUs, what happens if the GPUs are connected only via a slow PCIe bus (and not NVLink)? How would this affect latency, and why is high-bandwidth GPU interconnect crucial in this context?
2. **Parallelizing attention vs. MLP:** Suppose you split the attention heads of a transformer across GPUs (each GPU handles some heads) and also split the feed-forward network matrix. Describe how the outputs are merged in each case. What communication (all-reduce or all-gather) is needed for attention and for the MLP, and what does that imply for scaling efficiency?
3. **Memory trade-off:** If you have a 60B parameter model and two 32GB GPUs, outline how you would use tensor parallelism to serve this model. What portions of the model might still be replicated on both GPUs, and how much memory could you save per GPU roughly? What additional memory overheads could appear when doing this?
4. **Combining with data parallel:** Imagine you have 8 GPUs and a model that *just* fits on 2 GPUs with tensor parallelism. You also need to handle 4 concurrent queries with low latency. How would you architect a solution with both tensor and data parallelism in this case? What would be the groups of GPUs and the communication pattern?
5. **Limits to scaling:** Why doesn’t splitting a model across 64 GPUs (with tensor parallelism alone) typically result in 64× faster inference for a single prompt? Discuss the factors that cause sublinear scaling and how an inference engineer might detect and alleviate those issues (e.g. by profiling communication vs compute).

## **Pipeline Parallelism**

*Pipeline parallelism (layer-wise model parallelism) splits the model’s layers into stages across GPUs. In this diagram, the neural network’s layers are divided into three stages on GPU0, GPU1, GPU2 respectively. Each stage processes and then passes activations to the next, like an assembly line .*

**How it works:** Pipeline parallelism partitions the model *vertically* by layers (or sets of layers) and assigns different layers to different GPUs . Each GPU is responsible for a consecutive chunk of the neural network’s layer stack. For example, in a model with 24 transformer layers and 4 GPUs, you might put layers 1-6 on GPU0, layers 7-12 on GPU1, 13-18 on GPU2, and 19-24 on GPU3. When an input comes in, it first passes through the layers on GPU0, then the intermediate result is sent to GPU1 to process the next set of layers, and so on in sequence until GPU3 produces the final output . In essence, the inference is like a relay: each device does its part and forwards the activations to the next. This allows a single forward pass to utilize multiple GPUs without each needing the full model. Moreover, when multiple inputs (or micro-batches) are processed, the pipeline can be kept busy – while GPU1 is working on the second batch’s layers, GPU0 can start on a third batch, etc., overlapping work like an assembly line. The key idea is to increase throughput by *simultaneously* processing different inputs at different stages of the network.

- **Benefits:**
    - **Memory distribution:** Like tensor parallelism, pipeline parallelism enables serving models larger than one GPU’s memory by distributing layers. Each GPU only needs to load the weights for the layers it owns. This is especially useful for very *deep* models. For instance, if one GPU can hold 10 layers worth of weights, an 30-layer model can be split across 3 GPUs in a pipeline.
    - **Increased throughput via concurrency:** Once the pipeline is filled with multiple inputs, all GPUs can work in parallel on different stages. This means you can achieve high device utilization and throughput – while one input is in later layers, a new input can start processing in the earlier layers. With enough micro-batching, pipeline parallelism approaches the throughput of data parallelism, but using a single model copy split across GPUs.
    - **Limited communication:** The communication in pipeline parallelism is only the passing of activations from one stage to the next. This is typically smaller in size than exchanging full model parameters. For example, sending an activation tensor (e.g. batch_size × hidden_dim) forward is often less data than synchronizing large weight gradients in training or combining large weight shards as in tensor parallel. Thus, pipeline parallelism’s networking cost can be moderate, mainly point-to-point between adjacent stages.
    - **Combining with tensor parallel:** In practice, pipeline parallelism is often combined with tensor parallelism to form 2D parallelism. For example, within each stage you might still shard the layer across 2 GPUs (tensor parallel) and you have, say, 4 such stages in sequence (pipeline). This hybrid can scale to very large numbers of GPUs for extreme model sizes while balancing memory and compute load.
- **Trade-offs:**
    - **Sequential latency:** An inference still has to traverse all pipeline stages in order. For a single input, the end-to-end latency is the sum of all stage computations plus the inter-stage communication overhead. Unlike tensor parallel (which splits one layer’s work concurrently), pipeline parallelism doesn’t reduce the *critical path* latency – in fact, it can **increase** it if there’s communication delay between stages . So a single inference may be slightly slower than if it ran on one GPU (due to network hops between stages).
    - **Pipeline “bubble”:** To get throughput benefits, you need multiple inputs in flight to fill the pipeline. The first input incurs a **bubble** – GPU2 and GPU3 are idle until the input reaches them. Similarly, after the last input, the earlier GPUs go idle while later stages finish. For short bursts of requests or small batch sizes, these bubbles (idle times) reduce efficiency. If you cannot keep the pipeline full, you won’t get good utilization.
    - **Load balancing:** All stages should ideally take roughly equal time, otherwise the slowest stage becomes a bottleneck that holds up others. But different layers have different compute characteristics. For example, in transformers the later layers might have the same dimension sizes as earlier ones (so similar compute per token). If the model is evenly divisible, great – if not, you may need to distribute layers unevenly or even duplicate some compute to balance. An imbalance means some GPUs sit idle waiting for a straggling stage.
    - **Complexity and failure modes:** Implementing pipeline parallel inference requires coordination – you need to manage asynchronous transfers of data between stages, possibly use technologies like CUDA streams or NCCL for communication, and handle batch splits. Also, if one stage fails or is overloaded, it stalls the whole pipeline. There’s added complexity in coding and debugging compared to single-device inference.
- **Scalability:** Pipeline parallelism can scale to a large number of GPUs by increasing the number of stages (you can even assign one layer per GPU if you want, up to the number of layers). It’s commonly used across nodes (e.g., a model split across 2 or 4 servers). The scalability is mainly limited by the depth of the model and the overhead of stage-to-stage communication. Each stage only directly communicates with its neighbor, so bandwidth requirements are typically between one pair at a time, which is more manageable than all-to-all communication. With efficient scheduling (such as interleaving micro-batches), pipeline parallelism achieves near-linear throughput scaling with number of stages, as long as there are enough concurrent inputs to keep all stages busy. However, the *latency* for a single input will always include all stages serially. Recent optimizations like 1F1B (one forward, one backward micro-batch schedule in training) are not needed for inference (no backward pass), but similar ideas of overlapping stage processing can maximize device utilization. In summary, it scales well for throughput on large models, but if you had, say, 16 pipeline stages and only one query, it would be very slow – so it’s best when batch or request concurrency is high.
- **Use cases:** Pipeline parallelism is a go-to for **very deep models or sequence of model segments**. For example, if serving a gigantic transformer with 100+ layers, one might allocate 10 layers per GPU across 10 GPUs in a pipeline. It’s used in deployment of models like PaLM or GPT variants where model size mandates multi-GPU splitting. It shines in **batch inference** scenarios where you can batch many inputs and achieve high throughput (all GPUs busy). In interactive systems with low batch sizes, pipeline parallelism is mainly used just to fit the model in memory, accepting some latency hit. A real-world scenario: A long-context LLM with 64 layers might be split over 4 GPUs; if you can batch 16 requests, each GPU is busy on a different request’s portion, giving high throughput. If you only have 1 request at a time, that request still has to pass GPU1 → GPU2 → … → GPU4 sequentially, incurring extra hops – that’s the latency trade-off . Pipeline parallelism can also fail or become inefficient if one stage becomes a bottleneck (e.g., if one partition of layers is especially slow or memory-bound) or if the pipeline frequently stalls due to lack of input workload.

### **Interview Questions – Pipeline Parallelism**

1. **Latency vs throughput:** Explain why pipeline parallelism tends to **increase** single-query latency but can **increase** overall throughput. How does the concept of pipeline fill/bubble illustrate this trade-off?
2. **Stage partitioning:** Suppose you have a 12-layer transformer and 3 GPUs. Give an example of how you would assign layers to each GPU. What factors would you consider to ensure each GPU’s workload is balanced, and what could happen if one stage takes twice as long as the others?
3. **Pipeline in deployment:** In an inference server, you receive a burst of 8 requests and then nothing for a while. How would pipeline parallelism handle this compared to data parallelism? Discuss what happens during the burst (pipeline fill) and after (pipeline flush) in terms of efficiency.
4. **Inter-stage communication:** What kind of data is sent between pipeline stages during LLM inference? How does the size of this data compare to, say, the model size or the size of activations in tensor parallelism? (Consider, for example, sending hidden states of shape [batch, seq_len, hidden_dim] to the next stage.)
5. **Hybrid parallel strategies:** In a deployment of a very large model, you decide to use 8 GPUs with a combination of pipeline and tensor parallelism. How might you organize these 8 GPUs (e.g., 4 pipeline stages with 2-way tensor parallel in each)? What benefits does this hybrid approach offer over pure pipeline or pure tensor parallel alone in terms of scalability and resource usage?

## **Expert Parallelism (Mixture-of-Experts)**

**How it works:** Expert parallelism refers to distributing parts of a **Mixture-of-Experts (MoE)** model across multiple GPUs . An MoE model contains multiple “expert” sub-networks (for example, many feed-forward networks specialized on different token patterns) and a gating mechanism that activates only a few experts per input. During inference, the router (gating network) examines each input token or sequence and decides which expert(s) should handle it. Only those expert networks are executed for that input, and their outputs are combined to produce the final result. Different experts reside on different GPUs – effectively, each GPU might host a subset of the experts. When a token needs a particular expert, the token’s data is sent to the GPU owning that expert, processed there, and the result is sent back or onward  . Because each token only uses a small fraction of all experts, most of the model’s parameters are “idle” for that token, and the computation is sparse. For instance, imagine a model with 16 experts where each token is handled by 2 experts: if you have 4 GPUs, each could hold 4 experts. For a given token, the system activates the chosen 2 experts (say one on GPU2 and one on GPU3); those GPUs compute their part in parallel and return their outputs which are combined (the combination might happen on the original GPU or a central place). Expert parallelism is essentially model parallelism at the level of entire sub-networks: different GPUs hold different chunks of the model (the experts), and an inference dynamically uses only a subset of those chunks.

- **Benefits:**
    - **Massive model capacity:** MoE allows extremely large models (hundreds of billions of parameters) by spreading many experts across many devices, without requiring that every inference activates all those parameters. You can increase the number of experts (and hence parameters) almost linearly with number of GPUs, gaining model capacity and potential quality improvements, while each individual inference only pays for a few experts’ worth of compute . This means you get the quality of a huge model at a fraction of the inference cost per request (if the gating is effective).
    - **Per-token efficiency:** Since only a subset of experts are used for a given input, the computation and memory access is **sparse**. Each GPU works on the tokens that need its experts and is idle for tokens that don’t – which can be efficient if workload is balanced. Inactive experts don’t consume compute for that token . Compared to a dense model of equal size, an MoE can be faster because not all weights are processed for each token.
    - **Scalability:** Need a bigger model? Add more experts (and GPUs to host them). Expert parallelism scales out model size almost arbitrarily – many recent MoE research models scale to dozens or hundreds of experts. It’s easier to reach trillion-parameter scales by adding experts than by making layers or hidden sizes huge. This is attractive for deployment because you can trade-off between model quality and inference cost by adjusting how many experts are used per token.
    - **Better hardware utilization than pure model slicing:** Unlike tensor parallelism which forces every GPU to work on every token (incurring synchronization), expert parallelism lets different GPUs truly work on different tokens (when routing differs). This can lead to better overall hardware utilization when serving batches of varied inputs – GPUs can operate more independently on the tokens assigned to their experts, with less fine-grained communication. In ideal cases, it’s more like a scatter-gather pattern than lockstep synchronization, which can reduce communication overhead .
- **Trade-offs:**
    - **Complex routing & communication:** The gating and routing of tokens to experts introduces communication overhead and complexity. In inference, you might have to perform an all-to-all exchange where each GPU sends the token data that chose its experts to the appropriate devices, then gather results back. This can be a heavy communication pattern, especially if many experts (GPUs) are involved per token . If the interconnect is not fast, the latency added by routing can be large.
    - **Load imbalance:** One expert might end up being popular for many inputs (depending on the data distribution and gating decisions) while others are rarely used. This can create hot spots where one GPU (with a popular expert) gets overloaded with work while others sit mostly idle. MoE inference performance is very sensitive to how evenly the token load can be balanced across experts. Systems often have to implement load-balancing strategies or capacity limits (each expert only handles X tokens, extra tokens go to a secondary choice etc.) which complicates the design.
    - **Memory overhead:** While each GPU doesn’t need the full model, an MoE often still has a *shared backbone* (like the transformer layers without the feed-forward experts, or embedding layers) that might be replicated on all GPUs. Also, all experts are loaded in memory across the cluster. For example, a 1T-parameter MoE with 64 experts might load ~15B params per GPU if evenly divided, which is manageable, but if gating only uses a few experts per token, you carry a lot of “inactive” weights in memory that are only occasionally used. Also the gating network itself is usually replicated and small.
    - **Consistency and debugging:** The indeterministic nature of routing (depending on input content) can make testing and debugging harder. It’s not as straightforward as a single sequence always taking the same code path – small input changes might activate different experts. For deployment, you must ensure the system can handle any routing patterns (including worst-case where everyone hits the same expert). Caching of expert outputs or dynamic batching of tokens per expert might be needed to maximize efficiency, adding to system complexity.
- **Scalability:** Expert parallelism shines in scalability of model size. You can scale to *many* GPUs by adding more experts and distributing them. It has been demonstrated on clusters with thousands of experts spread over hundreds of GPUs. The scalability of throughput and latency, however, depends on the routing efficiency. In the best case (perfectly balanced, low-cost routing), you could achieve near-linear throughput scaling: more GPUs = more experts = more tokens processed in parallel (since different tokens use different experts). In practice, implementations like GShard or DeepSpeed MoE use techniques to reduce communication (e.g. each token only goes to top-2 experts, maybe on two GPUs) to make scaling manageable. The networking fabric is a big factor: high-bandwidth interconnect can allow frequent token exchanges. There is also a notion of **capacity factor** – how many tokens an expert can handle at once – which if set properly, can keep each GPU busy. If you increase batch size or number of concurrent tokens, you might need to proportionally increase experts to maintain low latency. Thus, scaling an MoE inference system involves scaling the number of experts (GPUs) and careful load balancing. It is worth noting that beyond a certain point, the overhead of coordinating hundreds of experts can limit latency gains – so often you might use expert parallelism to reach a model quality target with fewer flops, rather than to make inference super fast. Recent systems research (e.g. Tutel, FastMoE) provide optimized communication patterns for experts to improve scalability.
- **Use cases:** Expert parallelism is used in **extremely large language models** where a dense model would be too slow or too memory-heavy to serve. For instance, Google’s Switch Transformer and GLaM, or META’s LM on TorchMoE, use MoE to get big model performance with fewer FLOPs per token. In inference, MoEs are useful when different inputs vary a lot – the model can specialize experts for different topics, and only those experts run for relevant inputs. It shines when you need the capacity of a huge model but want to keep inference cost per token lower. A real deployment scenario might be a multilingual model that has different experts for different languages – when a Chinese query comes in, the Chinese-language experts activate (others don’t), and for an English query, a different set of experts activates. This can save time relative to running a giant dense model with all weights. However, MoE can *fail* or become less beneficial if the overhead negates the sparse computation advantage – e.g., if every token ends up needing many experts or if load balancing is poor. In the worst case, if an MoE’s gating isn’t effective, you could end up using most experts for most tokens, which turns it into a bloated dense model with extra communication. Therefore, MoEs are most advantageous in *mixed or broad* deployment contexts where inputs truly differ and can be routed to specialized parts of the network . Together AI and others exploring inference infra for MoE focus on ensuring that the routing latency and imbalance issues are mitigated so that this approach yields actual speedups for large-scale systems.

### **Interview Questions – Expert Parallelism (MoE)**

1. **Routing overhead:** In a distributed MoE model, each token may need to be sent to its selected experts on various GPUs. Describe the communication pattern you’d expect in an inference step for a batch of tokens (each token going to top-2 experts for example). What challenges does this pattern pose compared to the more predictable all-reduce in tensor parallelism?
2. **Load balancing:** Suppose in production you observe that one expert GPU is maxed out while others are underutilized (many inputs are routed to the same expert). What strategies could you employ to alleviate this bottleneck? (Consider solutions like gating adjustments, adding duplicate experts, or capacity throttling.)
3. **Comparing dense vs sparse:** If you have a dense 20B model and a sparse MoE model also totaling 20B parameters (say 4 experts of 5B each, with 2 experts used per token), compare the inference process. Which model might be faster per token and why? What conditions need to hold true for the MoE to actually be faster than the dense model?
4. **Failure modes:** During inference of an MoE model, what could go wrong if the gating network suddenly starts routing every token to *all* experts instead of one or two? How would that affect performance and memory, essentially turning the MoE into what kind of parallel workload?
5. **System design:** Imagine you are designing a serving system for a 64-expert MoE model on 16 GPUs. How would you map the experts to GPUs (e.g., 4 experts per GPU)? And how would you orchestrate the inference pipeline to minimize latency – for instance, would you gather all token representations to a central GPU for gating then scatter to experts, or perform gating on each GPU for its local tokens? Discuss the design decisions and their implications on performance.

## **Sequence Parallelism**

**How it works:** Sequence parallelism is a parallelization strategy that splits the *sequence length* dimension of the input (and corresponding intermediate activations) across multiple GPUs . This is particularly useful for transformer models when dealing with very long input sequences or when certain operations (like layer normalization or dropout) are not easily partitioned by other means. In practice, sequence parallelism might mean each GPU is responsible for a different chunk of the input tokens. For example, if you have a sequence of 10,000 tokens to process (extremely long context) and 4 GPUs, you could assign ~2,500 tokens to each GPU for certain parts of the computation. Unlike data parallelism, which would replicate the model for different data samples, sequence parallelism is working on the **same sequence** but dividing the work by positions within that sequence. Each GPU holds a full copy of the model (or at least the parts of the model that will be sequence-partitioned) but only processes its subset of tokens, and they exchange information as needed to produce the final result.

In transformers, one way this manifests is by partitioning parts of the forward pass that are element-wise or independent per token. For instance, layer normalization and dropout at each time step can be done independently on each token, so those can be executed in parallel for different token subsets on different GPUs . More ambitiously, some approaches also split the *attention* computation across sequence: e.g., GPU0 handles attention updates for tokens 1-2500 attending to all tokens, GPU1 handles tokens 2501-5000, etc., and they share key/value info. Essentially, sequence parallelism tries to distribute the memory and compute load of very long sequences by chunking the sequence into segments that can be processed concurrently, then stitched back together.

- **Benefits:**
    - **Handles long contexts:** Sequence parallelism shines when the sequence length (number of tokens) is huge – potentially in the thousands or even millions. It allows multiple GPUs to cooperatively handle a single long sequence that would otherwise not fit in memory or would be too slow on one GPU. By splitting the sequence, each GPU deals with a smaller effective sequence length, reducing memory usage for activations (which often scale with sequence length) .
    - **Memory savings on activations:** In a transformer, the memory required for activations (especially for attention key/value caches or intermediate states) grows with sequence length. By partitioning the sequence, each GPU only needs to store activations for its portion of the sequence, not the entire sequence. This can drastically cut memory per GPU, enabling, for example, a 16k token sequence to be handled by 4 GPUs each seeing 4k tokens. This is critical for long-context LLM inference where KV cache can consume tens of GB of memory for long inputs.
    - **Parallel speed for long ops:** Some operations that scale with sequence length (like certain matrix multiplies in attention with complexity O(n^2) in sequence length) can be divided among GPUs. By splitting the sequence, those operations can run in parallel, potentially giving a speed boost. For example, computing attention scores for 10k tokens attending to 10k tokens is heavy; splitting into two 5k-token chunks per GPU can roughly half the work each does (with some overhead for combining results). Recent research shows significantly lower latency for long sequences using sequence parallel methods – e.g. Snowflake’s *Ulysses* technique achieved about 3.4× lower latency on long context inference by distributing the sequence across GPUs .
    - **Complements model parallelism:** Sequence parallelism can be used alongside tensor or pipeline parallelism. In fact, some frameworks (Megatron, DeepSpeed) integrate sequence parallelism as an option when using tensor parallelism, to handle those layernorm/dropout parts more efficiently. It’s another dimension to exploit: you can shard *across tokens* in addition to sharding across layers or across weight parameters. This can further reduce memory bottlenecks and improve throughput when context is the limiting factor.
- **Trade-offs:**
    - **Complex communication:** Unlike data parallel (different sequences) or tensor parallel (combining partial sums), sequence parallelism often requires exchanging partial results that depend on the *other* sequence parts. For example, in self-attention each token attends to all others, so if tokens are split across GPUs, they must share their key/value representations or partial attention scores. This means heavy communication of activations between GPUs at certain stages. If not carefully optimized (with strategies like ring-based all-reduce or scatter-gather pattern), this can eat up the benefit of parallelizing the work.
    - **Added synchronization points:** All the GPUs working on different parts of the same sequence will need to sync up at points where the sequence segments interact (e.g., after computing attention for their tokens, they might need to exchange to get the complete result for each token). These synchronization barriers can reduce parallel efficiency, especially if sequence chunks are imbalanced or some require more compute (e.g., maybe one chunk has tokens that attend to many others heavily in a sparse attention scenario).
    - **Limited by sequence independence:** Some parts of the model are perfectly parallelizable by sequence (like layernorm on each token), but other parts inherently involve the full sequence (like softmax across attention scores or a final classification over the entire sequence). Sequence parallel methods have to find ways to partition those or otherwise still handle global operations. In some cases, you might still need to gather the full sequence on one GPU for a final step, which could become a bottleneck.
    - **Software complexity:** Implementing sequence parallelism can be quite complex. It’s less straightforward than splitting by layers or weights, because you’re effectively parallelizing the *data* within a single sequence in the middle of model computations. Ensuring correctness (that each token’s result is as if it saw the whole sequence) requires careful coordination. Tools like DeepSpeed-Ulysses introduce custom kernels and scheduling to manage this. It’s cutting-edge enough that not all inference frameworks support it out-of-the-box.
- **Scalability:** Sequence parallelism is specifically targeted at scaling with longer sequences rather than more data items. Its scalability is measured by how long of a sequence you can handle, or how much you can cut latency for a given long sequence by adding GPUs. If you have an extremely long sequence, you can split it over more GPUs – for instance, an input of 1 million tokens could be conceptually split over 16 or 32 GPUs to make each chunk manageable. The latency can decrease sub-linearly as you add GPUs because each GPU’s share of work goes down, though communication might start to dominate beyond a point. The *elastic* sequence parallelism idea (as in LoongServe) even suggests dynamically adjusting the number of GPUs based on sequence length – using more GPUs for longer inputs to keep latency in check, and fewer for shorter ones . In terms of across nodes: yes, you can do sequence parallel across machines too, but you’ll need an extremely fast network, since you’ll be sending lots of activation data around (potentially every token embedding). So, usually sequence parallel is done on GPUs with fast interconnect (within a server or across servers with InfiniBand). As with other parallelism forms, diminishing returns hit when the overhead of merging results outweighs the reduced compute per GPU. But recent advances (like ring-based attention algorithms) attempt to reduce communication so that sequence parallel can scale to more GPUs efficiently by only exchanging needed parts. Overall, it scales the feasible **context length** roughly linearly with the number of GPUs (each GPU adds more memory and compute for more tokens), and can reduce *latency* for a fixed long context, though not perfectly linearly due to overhead.
- **Use cases:** Sequence parallelism is most relevant for **long-context LLM inference**. If you have a model that can handle, say, 32k or 100k tokens context (for long documents, multi-document QA, etc.), a single GPU might be too slow or run out of memory storing the attention keys/values for that many tokens. Splitting the sequence across 4 or 8 GPUs can make such long context use cases feasible in reasonable time. Real-world use: A company wants to deploy a summarization model for long legal documents (tens of thousands of words) – they could use sequence parallelism to split the document across multiple GPUs and process it in parallel, enabling a result in, say, 5 seconds instead of 30. Sequence parallelism also appears in training long-context models, but for inference, systems like **Ulysses** by Snowflake specifically target serving 256k-token contexts by dividing work among GPUs. It shines when *memory* for sequence is the limiting factor – e.g., serving many long conversations or code files. If sequences are short (a few hundred tokens), sequence parallelism isn’t needed and would only add unnecessary overhead. It also may not help if the model’s bottleneck isn’t the sequence length (for instance, if compute per token is the issue rather than number of tokens). But as we push toward models with very long contexts, sequence parallelism or similar “context splitting” strategies are becoming key enabling techniques for inference on those models .

### **Interview Questions – Sequence Parallelism**

1. **Long context challenge:** Suppose you want to serve an LLM with a 64k token context window on GPUs that normally can only handle 8k tokens before running out of memory. How could sequence parallelism be applied here? Describe what each GPU would do and how they would collaborate to handle an input nearing 64k tokens.
2. **Attention computation:** In sequence parallelism for self-attention, each GPU might handle a subset of query tokens. How do these GPUs get the information about the key/value for tokens they *didn’t* compute? Describe a mechanism to exchange or replicate data so that each query token still attends to the whole sequence.
3. **When not to use:** If an LLM usually processes inputs of only 256 tokens, would sequence parallelism provide any benefit? Explain why splitting such a short sequence across multiple GPUs might be counterproductive, touching on overhead vs. work.
4. **Combining with other parallelism:** Imagine a scenario with both a very large model *and* very long inputs – e.g., a 50B parameter model with 32k context. You have 8 GPUs. How might you combine tensor/model parallelism and sequence parallelism to handle this? (Hint: you could shard the model weights and also shard the sequence length).
5. **Performance debugging:** You tried sequence parallel inference on a 10k-token input with 2 GPUs and found it was *slower* than using 1 GPU. What might be the causes for this slowdown? How would you determine if the problem is communication overhead, poor load balance, or something else?

## **Operator Fusion / Kernel-Level Parallelism**

**How it works:** Operator fusion is a low-level optimization where multiple operations in the model’s computation graph are combined into a single GPU kernel launch. While not parallelism in the multi-GPU sense, it exploits the parallel computing resources *within* a GPU more efficiently. In large LM inference, there are many sequential tensor operations (matrix multiplications, additions, activations, layer norms, etc.). Normally, each of these operations would be executed as a separate GPU kernel call: the GPU loads data from memory, does the op, writes results, then moves to the next op. With operator fusion, we merge these steps so that one kernel performs several operations in one pass over the data, reducing the overhead of launching kernels and intermediate memory transactions .

For example, in a transformer block, instead of launching one kernel for matrix multiply, one for adding bias, one for activation (GELU), one for dropout, etc., a fused kernel could do “GEMM + bias + GELU” together on each element. This means the data for that layer is read from memory once, transformed through all those operations by the GPU’s threads, and written out once at the end. Another prominent example is **FlashAttention**, which fuses the attention score computation, softmax, and weighted sum operations into a single highly-optimized kernel to avoid excessive memory reads/writes and to utilize on-chip memory better .

In inference, operator fusion often involves using compiler optimizations (like TensorRT, XLA, or custom CUDA kernels) that detect sequences of ops that can be merged. It can also involve using specialized libraries (e.g., fused multi-head attention kernels). The end result is that the GPU executes a smaller number of heavier kernels that each do more work in parallel.

- **Benefits:**
    - **Lower kernel launch overhead:** Each kernel launch has some CPU overhead and latency. By fusing operations, you launch fewer kernels overall. This is especially important in inference where batch sizes might be small (even 1) and the model is large – the overhead of scheduling thousands of tiny operations can become a significant portion of latency. Fused kernels amortize this overhead by doing more per launch .
    - **Better memory locality:** Fusion reduces the number of times intermediate results are written to and read from GPU global memory. Instead, data can stay in registers or shared memory for the duration of the fused computation. This cuts down on memory bandwidth usage and avoids unnecessary data traffic, which is often a bottleneck. In a memory-bound scenario (common in LLM decode where we shuffle a lot of data), this can improve throughput.
    - **Improved parallel efficiency:** A single fused kernel can be optimized to use the GPU’s parallel threads and warps more effectively. For instance, combining elementwise ops means one thread can do an element’s series of ops sequentially without handing off to another thread or idling. It often increases occupancy and uses fewer global synchronizations. Overall, this means the GPU spends more time doing actual math and less time idling or switching context.
    - **Tailored optimizations:** Fused kernels can implement algorithmic optimizations that aren’t possible in a fragmented implementation. FlashAttention is a great example: by fusing, it can tile the attention computation to avoid storing the entire big attention matrix, drastically reducing memory usage and speeding up computation. This kind of optimization yields big gains (FlashAttention can 2-4× speed up the attention step and enable longer sequences) which benefit inference latency and throughput.
- **Trade-offs:**
    - **Complex development:** Writing and maintaining fused kernels (especially for complex sequences of operations) is hard. It often requires low-level CUDA or assembly programming and careful handling of different GPU architectures. Each time the model changes (say a new activation function), you might need new fusion code. Relying on compiler frameworks can abstract this, but compilers may not catch all fusion opportunities or might have bugs. So there’s an engineering cost and less flexibility.
    - **Less modularity:** Once ops are fused, it’s harder to debug or modify them individually. If a fused kernel is malfunctioning, it’s a monolithic block to troubleshoot. Also, you lose the ease of mixing and matching different operations (unless you generate a new fused kernel). This is usually fine for well-known architectures but can slow down iteration on new model designs.
    - **Hardware-specific tuning:** A fused kernel might be optimized for a specific GPU generation or require tuning parameters (block size, thread count) for efficiency. This can make it less portable. For example, a kernel fused and tuned on NVIDIA V100 might not be optimal on NVIDIA A100 unless re-tuned. Often, vendor libraries (NVIDIA’s FasterTransformer, etc.) provide these kernels but one must update them for new hardware.
    - **Diminishing returns:** Not every sequence of ops is worth fusing. Fusing two very large matrix multiplies, for instance, is not possible if there’s a data dependency between them (you need the result of first before second). Many critical ops (like the big GEMMs in transformers) are already quite optimized individually. The benefit of fusion is mostly on the “glue” operations (activations, elementwise adds, small matrix ops). Once those are fused, additional fusion won’t help much – so you eventually hit a point of no further speed-up. Also, if a kernel becomes too large or complex, it may use a lot of registers or not scale as well, potentially even *hurting* performance by reducing occupancy. So there’s a sweet spot in how much to fuse.
- **Scalability:** Operator fusion is about optimizing single-GPU performance. It doesn’t directly scale across GPUs, but it helps you better saturate each GPU’s compute and memory bandwidth. In multi-GPU inference, you’d typically apply the same fused kernels on each GPU (for whatever portion of the model it runs). The fused kernels ensure each device runs as efficiently as possible, which indirectly improves overall scalability because you need fewer GPUs to achieve a certain throughput. If each GPU is 30% faster due to fusion, you scale out 30% less to meet a throughput target. Additionally, by reducing memory bandwidth needs, fusion can make other parallelism schemes more effective (for example, if tensor parallel partitions have less data to exchange because some intermediate wasn’t materialized due to fusion, that’s a win). But mostly, scalability here means handling larger batch sizes or sequence lengths on a single GPU more efficiently. As batch size increases, kernel launch overhead amortizes naturally, so fusion is most impactful at smaller batch or streaming inference. Modern compilers can fuse many ops regardless of batch size, maintaining efficiency from batch=1 to batch=N. In summary, fused kernels keep the GPU pipelines busy and minimize idle gaps, ensuring you get closer to peak theoretical performance. This is crucial when trying to maximize throughput per GPU (e.g., to serve as many tokens/s from an expensive accelerator as possible).
- **Use cases:** Practically every high-performance inference engine uses operator fusion. For LLMs, when using frameworks like NVIDIA TensorRT, Hugging Face Accelerate, or ONNX Runtime with optimizations, a lot of fusion is done under the hood. For example, during the **decode phase** of autoregressive inference (generating one token at a time), the batch is often 1 and there are many small kernel launches – here fusion yields big latency improvements, making the model responsive. A concrete case: **vLLM** and **FasterTransformer** fuse the sampling and output projection steps, or fuse multiple small operations in the transformer block, to cut latency. FlashAttention (which is now widely used) is a case of kernel-level parallelism providing both speed and ability to handle longer sequences without blowing memory. Operator fusion especially shines in edge deployments or CPU inference too, but in GPUs it’s about squeezing out every bit of performance. One scenario where it might “fail” to help is if the model is already dominated by one huge operation (like a giant matmul) – you can’t fuse much into that, it’s already one kernel. But for Transformers, there are plenty of medium-sized ops to fuse. Essentially, whenever you see mention of “optimized kernels” or “TensorRT graph optimizations,” it’s operator fusion at work. It’s a behind-the-scenes technique, but inference engineers absolutely rely on it to hit latency and throughput targets .

### **Interview Questions – Operator Fusion**

1. **Latency at small batch:** In an LLM serving scenario with batch size 1, why can the overhead of kernel launches significantly affect latency? How does operator fusion mitigate this, and what kind of latency improvement might you expect from fusing a chain of elementwise operations?
2. **Memory bandwidth vs compute:** If a transformer model’s inference is memory-bound (not all GPU cores are busy because it’s waiting on memory transfers), what role can kernel fusion play in improving performance? Give an example with specific operations (e.g., softmax and scaling in attention) that illustrates this.
3. **FlashAttention insight:** FlashAttention is a fused-kernel approach for the attention mechanism. Can you explain what it does differently compared to a naive implementation of attention in terms of kernel-level operations and memory access? Why is this especially relevant for long sequences?
4. **Trade-off scenario:** Fusing operations can sometimes make a kernel very large. What could go wrong if you fuse too many operations into one kernel? (Consider things like GPU register pressure, complexity of scheduling, and debugging.) How do engineers decide what to fuse and what to leave separate?
5. **Tooling:** As an inference engineer, would you prefer to write custom fused kernels by hand or rely on a compiler/graph optimizer to do it for you? Discuss the pros and cons of using automated tools like NVidia TensorRT or TVM to perform operator fusion versus writing custom CUDA kernels for your model. How might the answer change if you need to deploy on multiple hardware platforms (GPU, CPU, etc.)?

## **Asynchronous / Overlapping Prefetch Parallelism**

**How it works:** Asynchronous execution and overlapping prefetching are techniques to utilize computation and communication concurrently, effectively creating a pipeline between data transfer and compute. In LLM inference, this often comes into play with things like *prefetching the next data* (weights, activations, or key/value cache) while current computations are ongoing. Modern GPUs and frameworks allow us to use multiple streams or hardware engines so that, for example, while the GPU cores are busy computing a matrix multiply, the DMA engine can be fetching the next layer’s weights from CPU memory or the next batch of data from system RAM into GPU memory. By the time the compute finishes, the needed data is already “in place,” reducing waiting time.

A concrete example is **KV cache prefetching** during autoregressive generation. When generating the next token, the model needs to read the key/value tensors from all previous tokens (which might be in GPU memory or offloaded to CPU if memory is limited). If you wait until you need them and then load them, the GPU might sit idle waiting for data. Instead, with asynchronous prefetch, the system predicts which parts of the KV cache will be needed soon and issues a transfer to L2 cache or HBM memory ahead of time, *overlapping* that data movement with the current token’s computation . As a result, when the compute units are ready for that data, it’s already available in fast memory, avoiding a stall.

Another example: In pipeline parallelism across GPUs, one can overlap the communication of activations between stages with computation. As soon as GPU0 produces its output, it sends it to GPU1 *while* GPU1 is still finishing up its previous task, so that by the time GPU1 is free, the new data has partially or fully arrived (this often uses async send/recv operations).

From a systems perspective, this often involves techniques like **double buffering** (one buffer being used in computation while another is being filled with new data), using CUDA streams (one stream does compute, another handles memcopies, set to overlap), or specialized instructions (like CUDA’s cp.async to prefetch data into shared memory or cache) . The goal is to hide latency of memory I/O or inter-device communication behind useful computation, so that the GPU (or multi-GPU system) is never idle waiting on data.

- **Benefits:**
    - **Latency hiding:** The primary benefit is reduced effective latency. If, say, loading a layer’s weights from CPU to GPU takes 5 ms and computing on them takes 5 ms, doing them sequentially would be 10 ms per layer. But if you prefetch the next layer’s weights during the 5 ms of computing on the current layer, you can potentially get it down to just ~5 ms (plus perhaps some overhead) total, as the transfer and compute happen in parallel. This can drastically speed up inference when memory transfers (or network transfers) are on the critical path.
    - **Throughput improvement:** By keeping the hardware busy (either doing compute or transfer at all times), you maximize utilization. This is especially important if the workload is mixture of compute-heavy and bandwidth-heavy tasks. Overlap ensures neither the compute units nor the communication channels are left idle. As a result, the overall tokens-per-second or requests-per-second of the system increases. For instance, an asynchronous KV cache prefetch was shown to improve attention throughput by ~2× in some research because the memory wait was eliminated.
    - **Scalability across heterogenous memory:** Overlap techniques enable using resources like CPU memory or slower GPU memory without incurring as high a penalty. You can offload large buffers (like KV cache, or large embedding tables) to CPU or slower memory and use prefetch so that it appears “fast enough” when needed. This effectively increases the usable memory for inference (allowing longer contexts or larger models than pure HBM would allow) without linear latency cost. Systems like PagedAttention or other offloading strategies rely on overlapping GPU compute with CPU-GPU data transfer to work well.
    - **Smooth pipeline:** In multi-request or batched scenarios, asynchronous scheduling allows one part of the system to prepare data for the next request while the current request is executing. For example, while the GPU is busy generating a token for user A, the CPU can simultaneously preprocess the request for user B and even transfer the input to GPU memory. This way, as soon as the GPU is free, user B’s data is ready to go. This leads to better overall responsiveness and throughput.
- **Trade-offs:**
    - **Complex scheduling logic:** Implementing effective overlap requires careful scheduling of events. You must predict what data will be needed (prefetching the wrong data or too early/late reduces benefit or can waste bandwidth). If done incorrectly, you could fetch data that isn’t ultimately used (wasting memory bandwidth) or you might still end up waiting because prefetch was late. It introduces complexity in code (using multiple streams, callbacks, etc.) and potential bugs (e.g., race conditions if not synchronized properly).
    - **Memory overhead:** Double-buffering and prefetching mean you might hold extra buffers. For example, you might need two copies of a weight chunk in flight (one being used, one being loaded). This can increase memory footprint slightly. Also, if you prefetch a lot of data “just in case,” you might evict useful data from caches or use up GPU memory for things not yet needed.
    - **Hardware limitations:** Not all operations perfectly overlap. Sometimes the compute is also using the memory bandwidth heavily, so a simultaneous transfer might contend for resources (PCIe or memory controllers), leading to less than ideal overlap. On some systems, there may be only one copy engine, etc., which means you can overlap compute with copy, but if you try to do two copies at once and a compute, maybe a bottleneck appears. Basically, theoretical overlap might be limited by underlying hardware concurrency.
    - **Difficult to generalize:** The optimal prefetch distance (how far ahead to load something) or overlap strategy can be workload-specific. For instance, if generation is going token by token, you might always fetch the next token’s data one token ahead. But if the pattern is irregular, or if using beam search, etc., logic gets complicated. Tuning the overlap for different batch sizes or sequence lengths might require experimentation. There’s also a risk of making things *worse* if the overlap saturates some resource – e.g., launching too many asynchronous transfers could clog the bus while GPU is also trying to use it for other memory accesses.
- **Scalability:** Overlap techniques help maintain scalability as you move to systems with multiple resources. On a single GPU, overlapping compute and memory operations lets you approach the hardware’s full capacity (both compute and bandwidth). In multi-GPU, overlapping communication (like all-reduce or point-to-point transfers) with computation is crucial for scaling efficiency. For example, in distributed data parallel training, techniques like overlapping gradient all-reduce with backward computation (known as wait-forward-backward propagation) are used; similarly for inference, overlapping the transfer of pipeline stage outputs with computation of those stages improves scaling in pipeline parallel inference. If you add more GPUs or more nodes, the communication overhead usually grows – overlapping helps hide that, so scaling is more linear. The asynchronous prefetch of KV cache is especially useful if you offload KV to CPU for many concurrent sequences – it basically parallelizes GPU compute and CPU-GPU IO, meaning adding more CPU memory (for longer context) doesn’t linearly slow you as long as you can overlap. Generally, a well-designed inference engine will use multiple threads/streams to ensure data movement, compute on GPU, and even CPU-side tasks (like token post-processing or scheduling) are all happening in parallel as much as possible. This turns the inference process into a pipeline with stages (data prep, compute, communication, etc.) overlapped. Scalability then improves because you’re not extending the critical path with each new component – you’re running parts concurrently. The limitation is when there’s no independent work to overlap with (then you’re back to sequential waits). So as model architectures evolve, one tries to restructure tasks to create opportunities for overlap.
- **Use cases:** Asynchronous and overlapping techniques are ubiquitous in high-performance inference servers. For example:
    - **KV Cache Offloading:** Many long-context inference implementations offload the oldest KV cache to CPU and use asynchronous prefetch (using cudaMemcpyAsync or even GPU direct RDMA) to fetch needed segments in time for attention. This allows, say, a 128GB CPU memory to supplement 40GB of GPU memory without stalling every generation step .
    - **Streaming generation:** When streaming tokens to a client, you can overlap the preparation of network packets with the generation of the next token. The GPU computes token N+1 while the CPU thread sends token N out over the socket.
    - **Multi-request batching:** Systems like HuggingFace Text Generation Inference or NVIDIA Triton will gather incoming requests into batches. They often use one thread to batch inputs while the GPU is busy on the current batch, thereby overlapping batching logic with computation.
    - **Pipeline parallel inference:** Overlapping communication between pipeline stages (using CUDA streams that do sends/receives concurrently with compute) is critical to avoid pipeline bubbles. This means stage i is sending data to stage i+1 at the same time stage i+1 is still finishing its previous micro-batch – classic producer/consumer overlap.
    - Where it fails: If the workload is purely compute-bound and there is little data transfer, then there’s not much to overlap – the GPU is 100% busy with math, and memory is mostly co-utilized. In such cases, async prefetch doesn’t buy much. Another tricky case is if dependencies are tight: e.g., you can’t preload data because you don’t know which branch of a model you will take (like in a conditional execution graph, though in inference this is rare for transformers). But by and large, any time there is *wait time* for data, we try to overlap. It’s a key part of achieving high throughput in systems like DeepSpeed and TensorRT engines. In fact, NVIDIA’s newer GPUs and software explicitly support prefetch instructions (like cp.async to L2 or shared mem) to facilitate exactly this kind of optimization .

### **Interview Questions – Asynchronous Prefetch & Overlap**

1. **Latency hiding principle:** What does it mean to *overlap communication with computation* in the context of LLM inference? Can you give a specific example (e.g., overlapping data transfer of the next token’s data with the current token’s computation) and explain how it hides latency?
2. **KV cache example:** In a long-context generation, assume the GPU can only hold the cache for the last 1000 tokens, and older ones are on CPU. How would you design a prefetching strategy to ensure the GPU has the necessary key/value tensors when computing attention for token 1001? What factors would you consider (like how early to prefetch, how much to prefetch)?
3. **Double buffering:** Explain the concept of double buffering in GPU inference. For instance, how could double buffering be used when loading model weights from CPU to GPU in a partial offload scenario, or when sending inputs to a GPU, to keep the compute pipeline fed?
4. **Streams and engines:** Modern GPUs have separate copy engines and compute engines. How does this hardware feature enable overlap of data transfer and kernel execution? What must a programmer do in code to actually achieve this overlap (consider CUDA streams and asynchronous calls)?
5. **Diagnosing overlap effectiveness:** Suppose you implemented asynchronous prefetching of data between GPUs for pipeline inference, but you’re not seeing any speedup – the timeline still shows communication and computation happening sequentially. What could be some reasons for this (e.g., synchronization issues, resource contention, small message sizes)? How would you verify and fix the overlap so that they truly run in parallel?

6. # **10 Rigorous LLM System Design Interview Questions**

1. **End-to-End LLM Serving Pipeline Design:** Suppose you must design a production inference service for a 70B-parameter language model that handles hundreds of requests per second while keeping tail latency low (e.g. 95th percentile under 500ms for a moderate response length). How would you architect this system from the ground up? Consider the model deployment across hardware (GPU distribution), how you’d batch or queue requests, and what optimizations you would employ to achieve high throughput *and* low latency. *(Tests the candidate’s ability to holistically design a large-scale LLM inference system under strict latency and throughput constraints.)*
2. **Dynamic Batching Strategies:** LLM inference is iterative and can benefit greatly from batching multiple requests together. How would you implement a **dynamic batching** mechanism for incoming queries to maximize GPU utilization without introducing unacceptable latency for individual requests? Discuss how you’d handle varying input lengths and generation lengths (to avoid long prompts slowing down shorter ones), and compare approaches like fixed-size batches vs. continuous batching with fine-grained scheduling. *(Tests understanding of request batching policies and how to balance throughput vs. per-request latency, e.g. via smart scheduling and grouping of requests.)*
3. **Key-Value Cache Utilization:** In autoregressive generation, the model can cache the key/value pairs from prior tokens to avoid recomputing them on each step. How could you leverage **KV caching** to speed up inference in a multi-turn conversation or for repeated prompts across requests? Describe how you might implement a cache for previously computed states and discuss the memory vs. compute trade-offs involved. How would you decide when to reuse or discard cached states, and what are the challenges in managing cache consistency in a high-throughput setting? *(Tests knowledge of transformer KV caching mechanics and the ability to weigh memory overhead against compute savings in practice.)*
4. **Model Quantization and Precision Trade-offs:** If GPU memory and throughput are at a premium, one option is to compress the model. Explain how you would use **quantization** (e.g. 8-bit or 4-bit weights) to reduce the model’s memory footprint and possibly increase inference speed. What are the impacts of lower precision on model accuracy and on hardware performance (throughput/latency)? Additionally, discuss any implementation considerations—such as quantization-aware training vs. post-training quantization or runtime decomposition techniques—and how those might affect a production inference pipeline. *(Tests understanding of model compression techniques and their real-world effects on performance and accuracy in an inference setting.)*
5. **Parallelism and Model Sharding:** When a single GPU isn’t sufficient to host or compute the model, how would you split a large LLM across multiple GPUs or machines? Compare **tensor/model parallelism** (splitting individual layers across GPUs) with **pipeline parallelism** (dividing the stack of layers among GPUs in sequence) for inference. How does each approach affect latency and throughput? Discuss the challenges you’d need to address (such as synchronizing between devices, communication overhead, and load balancing) to make multi-GPU inference efficient and reliable. *(Tests knowledge of distributing a model over multiple devices and the trade-offs between different parallelization strategies in terms of performance and complexity.)*
6. **Speculative Decoding for Faster Generation:** Describe what **speculative decoding** is and how it can be used to accelerate LLM inference. In what scenario would you employ a speculative decoding approach, and how does it leverage a smaller “draft” model alongside the large model to reduce end-to-end latency? Explain the potential speed-ups and also the complexities or downsides of this technique (for example, managing two models, ensuring consistency of the final output, or wasted computation when the speculation is incorrect). *(Tests understanding of an advanced inference optimization technique and the ability to reason about its benefits and implementation challenges.)*
7. **Memory Offloading and Management:** Imagine your model and its intermediate data (like activation maps or the KV cache) don’t all fit in GPU memory during inference, especially with long context inputs. How would you design an **offloading policy** to move parts of the model or data to CPU memory (or even NVMe storage) and bring them back when needed? Discuss what factors you’d consider in an offloading strategy – for instance, which layers or data to offload, how to overlap data transfer with computation to hide latency, and how PCIe or interconnect bandwidth constraints come into play. What are the performance trade-offs of offloading, and how can smart scheduling minimize the impact on latency? *(Tests the candidate’s grasp of memory–compute trade-offs and ability to manage limited GPU memory by trading off transfer overhead, as seen in large-model inference scenarios.)*
8. **Throughput vs. Latency Trade-offs:** In a high-volume LLM service, you often need to maximize total throughput (tokens/sec or queries/sec) while still meeting latency requirements for individual users. How would you balance this trade-off in practice? Consider ideas like using **adaptive batch sizes** (batching more aggressively during peak load vs. prioritizing low latency for realtime requests), deploying separate model replicas or service tiers for high-priority low-latency requests vs. lower-priority bulk requests, or any scheduling/allocation mechanism to ensure both objectives are met. Discuss how you would evaluate the latency–throughput sweet spot and adjust the system as load patterns change. *(Tests understanding of operational trade-offs in system design and the ability to devise strategies that cater to different service level objectives for throughput and latency.)*
9. **Fault Tolerance in Inference Pipelines:** Serving large models is not only about speed – it’s also about reliability. Suppose a generation request is part-way through when a GPU server fails or a network hiccup occurs. How could you design the system to be **fault-tolerant** in such cases? Discuss mechanisms like checkpointing or saving intermediate state so another node could resume if possible, retrying requests from scratch (and what that means for user experience), or running duplicate inference in parallel on redundant hardware to hedge against failures. What are the pros and cons (especially in cost and complexity) of these approaches in a production, high-throughput inference environment? *(Tests the candidate’s ability to incorporate reliability and failure-handling into system design, recognizing the challenges of long-running sequential processes like LLM inference.)*
10. **Cost-Efficiency and Scalability Considerations:** Large-scale LLM inference can be extremely expensive. What strategies would you use to **optimize cost** while maintaining acceptable performance? Discuss options such as using smaller or distilled models for certain tasks or routing simpler queries to cheaper models, leveraging spot instances or scale-to-zero for unused capacity, sharing GPUs across multiple models or clients (multi-tenancy) to increase utilization, and using techniques like batch processing or quantization to reduce resource usage. How would you ensure the system scales cost-effectively with demand, and what trade-offs might you have to accept to stay within budget? *(Tests the candidate’s ability to think beyond pure performance and design a solution that is economically sustainable, demonstrating awareness of real-world constraints like resource cost and utilization.)*






























# Practice 2

# 50 Essential PyTorch Coding Interview Questions (LLM Inference & Optimization)

## Easy Questions (Fundamentals & Basics)

1. **(Easy):** Use PyTorch's inference mode properly: put a model in evaluation mode and disable gradient calculations. Write a code snippet that wraps a forward pass in `model.eval()` and `torch.no_grad()` to ensure no gradients are tracked.
2. **(Easy):** Perform device management for inference: given a PyTorch model and input tensor, write code to move them to CUDA GPU for faster inference, then transfer the result back to CPU (e.g., using `model.to('cuda')` and `tensor.to('cuda')`, then `.cpu()` on the output).
3. **(Easy):** Implement the softmax function from scratch for a given logits tensor and use it to get a prediction. For example, compute probabilities with exponentiation and normalization (without using `torch.softmax`), then use `torch.argmax` to find the index of the highest probability.
4. **(Easy):** Define a simple neural network module in PyTorch and run a forward pass. For instance, implement an `nn.Module` with one `nn.Linear` layer followed by a ReLU activation. Show how to instantiate this model and feed a sample input through it.
5. **(Easy):** Use an embedding layer to map token IDs to vectors. For example, given a batch of token indices, create an `nn.Embedding` of appropriate size and show how to retrieve the embedding tensor for the batch (by calling the embedding layer on the input indices).
6. **(Easy):** Combine embedding vectors with positional encodings. Suppose you have a tensor of word embeddings and a tensor of positional encodings of the same shape; write code to add them together elementwise to form the final input for a transformer model.
7. **(Easy):** Pad sequences for batching: write a function that takes a list of sequences (lists of token IDs of varying lengths) and pads them with a PAD token (e.g., 0) to the same length. Also return an attention mask indicating which positions are real tokens (1) and which are padding (0).
8. **(Easy):** Implement a basic greedy decoding loop for text generation. Starting from an initial prompt (sequence of input IDs), iteratively feed it into the model to get next-token logits, pick the token with the highest probability (argmax), append it to the sequence, and repeat until an end-of-sequence token is produced.
9. **(Easy):** Calculate model size: write code to compute the total number of parameters in a given PyTorch model and estimate its memory footprint. (Hint: sum up `param.numel() * param.element_size()` for each parameter to get total bytes, and convert to MB or GB.)

## Intermediate Questions (Moderate Difficulty)

1. **(Medium):** Implement scaled dot-product attention. Given query, key, and value tensors (Q, K, V) of shape `(batch, seq_len, dim)`, compute the attention output = softmax$(QK^T / \sqrt{d}})$ · V. Include support for an attention mask (e.g., ignore certain positions by adding `inf` to logits before softmax for masked positions).
2. **(Medium):** Implement the Transformer's feed-forward network block. Given an input tensor of shape `(batch, seq_len, dim)`, pass it through a two-layer MLP: first `Linear(dim → hidden_dim)`, apply an activation (e.g., GELU), then `Linear(hidden_dim → dim)`. Show this in PyTorch code (you can assume some `hidden_dim` value).
3. **(Medium):** Implement top-k sampling for one step of language model decoding. Given a tensor of logits for the next token, filter it to the top k highest values (use `torch.topk`), then sample a token from those top-k probabilities (e.g., with `torch.multinomial`). The code should output an index for the sampled token.
4. **(Medium):** Implement nucleus (top-p) sampling for one decoding step. Given logits and a probability threshold p, sort the token probabilities, compute their cumulative sum, and select the smallest set of tokens whose cumulative probability ≥ p. Then sample the next token from that set. Provide code to perform this selection and sampling.
5. **(Medium):** Add caching to an autoregressive transformer decoding loop. Modify a naive generation function so that it passes a “past key-values” cache to the model. Show how you would store the K and V from each timestep (e.g., in lists or a preallocated tensor) and reuse them in subsequent model calls to avoid recomputing attention on previous tokens.
6. **(Medium):** Batch by sequence length for efficiency. Given a list of input sequences of different lengths, write code to sort them by length, batch those of similar lengths together, pad within each batch, and then run the model on each batch. (This minimizes padding and idle compute, improving throughput on variable-length inputs.)
7. **(Medium):** Implement micro-batching for inference. If a batch of N inputs is too large to process at once on the GPU, show how to split it into smaller sub-batches, run the model on each sub-batch sequentially (accumulating outputs), and then concatenate the results. Ensure the final outputs preserve the original input order.
8. **(Medium):** Optimize inference with TorchScript. Take a simple PyTorch model (or function) and demonstrate how to convert it to TorchScript using `torch.jit.trace` or `torch.jit.script`. Show the code for scripting/tracing the model and then using the compiled `scripted_model` to perform a forward pass.
9. **(Medium):** Use PyTorch 2.x compile (TorchDynamo + TorchInductor) to speed up inference. Write an example of wrapping a model with `torch.compile` and running inference on some input. (For instance: `optimized_model = torch.compile(model)` then use `optimized_model(x)` to execute the compiled graph for faster execution.)
10. **(Medium):** Profile a model's inference to find bottlenecks. Use `torch.profiler` (or `torch.autograd.profiler`) in a context manager to record the time taken by each operation during a forward pass. Provide code that runs the model under `torch.profiler.profile(...){ ... }` and then prints out a report of the most time-consuming ops or layers.
11. **(Medium):** Use mixed precision during inference. Show how to wrap model inference code in a `torch.cuda.amp.autocast()` context to run it in float16 where possible. For example, demonstrate loading a model and input, then calling `with torch.cuda.amp.autocast(): output = model(input_half)` to leverage tensor cores, and mention the speed/memory benefits of FP16.
12. **(Medium):** Apply dynamic quantization for faster CPU inference. Provide code to quantize a trained model (for example, using `torch.quantization.quantize_dynamic` on a model with linear layers or LSTMs) so that it uses int8/FP16 weights. Then show how to run inference with the quantized model and mention the potential speedup on CPU (with minimal accuracy drop).
13. **(Medium):** Prune insignificant weights in a model. For instance, take a fully connected layer's weight matrix and zero out all entries with magnitude below a threshold. Show code that identifies these small weights (e.g., `mask = (weight.abs() < threshold)`) and sets them to zero in-place. Comment on how this sparsity might affect model size or speed (noting that unstructured sparsity may need specialized kernels to see speedup).
14. **(Medium):** Use the HuggingFace Transformers library to run a model inference manually. For example, load a pretrained GPT-2 model and tokenizer, encode a prompt into input IDs, feed the input IDs to the model (e.g., `outputs = model(input_ids)`), get the logits, and then decode the model's output IDs back to text. (This tests using HF models without the high-level `.generate()` convenience function.)
15. **(Medium):** Use pinned memory to accelerate data transfer. Demonstrate creating an input tensor with `pin_memory=True` (or using a DataLoader with `pin_memory=True`), then transferring it to the GPU with `tensor.to('cuda', non_blocking=True)`. In code comments, explain how pinned (page-locked) host memory can improve throughput for CPU-to-GPU data copy operations by allowing DMA transfers.

## Advanced Questions (High Difficulty)

1. **(Hard):** Implement multi-head self-attention from scratch. Given an input tensor of shape `(batch, seq_len, model_dim)` and weight matrices for W_q, W_k, W_v (to project inputs to each head) and W_o (to project concatenated heads to output), write code to compute multi-head attention. Split the input into multiple heads, compute scaled dot-product attention for each head (without using `nn.MultiheadAttention`), then concatenate the head outputs and apply the output projection. (Ensure tensor shapes line up for the matrix multiplies.)
2. **(Hard):** Implement a full Transformer *decoder* block in PyTorch. The block should include self-attention followed by a feed-forward network, with residual connections and layer normalization around each sub-layer. Write the `forward` method assuming you have functions or modules for the attention and feed-forward parts. Show how you take the input `x`, compute `attn_out = SelfAttention(x)`, then `x = x + attn_out` followed by `LayerNorm`, then pass that through the feed-forward network, add the residual and normalize again.
3. **(Hard):** Implement beam search decoding for a language model. Write a function that given a model and an input prompt, performs beam search with a specified beam width *B*. It should keep track of multiple hypotheses (sequences and their cumulative log-probabilities), expand each hypothesis with new tokens at each step (using the model's output probabilities), and prune down to the top B candidates. Continue until a stopping condition (e.g., all beams have produced an end-of-sequence token or a max length is reached), then return the highest scoring completed sequence. Include code for managing the beam candidate lists at each step.
4. **(Hard):** Implement speculative decoding to accelerate generation. Suppose you have a large language model and a smaller "draft" model. Write a procedure where the draft model predicts the next *k* tokens in one go, and the larger model then verifies these tokens sequentially. If the large model's output matches the draft for a token, you accept it and move on to verifying the next token; as soon as it diverges, you discard the draft's remaining suggestions and resume generation from that point with the large model (perhaps resampling a new draft continuation). Provide a code outline showing how you'd interleave calls to the draft model and the main model, managing two sets of tokens (the proposed tokens and the confirmed tokens).
5. **(Hard):** Outline a continuous batching strategy for an LLM inference server (similar to what **vLLM** does). Write pseudo-code for a loop that continuously collects incoming requests and groups them into batches on the fly. For example, maintain a queue of incoming requests; at each iteration, take as many as available (up to some max batch size) to form a batch and run the model. If new requests arrive while a batch is running, they wait and then get batched in the next iteration. Ensure your outline handles a timeout or maximum delay so that no request waits indefinitely. (This tests understanding of dynamic batching in a live setting.)
6. **(Hard):** Use CUDA streams to overlap computation and data transfer. Provide an example where you create at least two `torch.cuda.Stream` objects: one stream that preloads or preprocesses data on the GPU while another stream runs the model inference on already-loaded data. Show how to use `with torch.cuda.stream(stream): ...` to assign operations to a stream, and ensure that you launch GPU-to-GPU copies or CPU-to-GPU transfers with `non_blocking=True`. The goal is to overlap the data copy of batch *n+1* with the compute of batch *n*.
7. **(Hard):** Write a custom PyTorch autograd Function for a new operation. For example, implement a custom ReLU. Define a subclass of `torch.autograd.Function` with a `forward(ctx, input)` that returns `input.clamp(min=0)`, and a `backward(ctx, grad_output)` that returns `grad_output * (input > 0).float()`. Provide the code for this class and then show how to use it in a model (e.g., `CustomReLU.apply(tensor)`) to verify it computes the same result as the built-in ReLU. (This tests low-level autograd understanding.)
8. **(Hard):** Write a GPU kernel using **OpenAI Triton** to perform an elementwise operation (for instance, add 1 to each element of an input tensor), and show how to launch it from PyTorch. Include the Triton kernel definition with `@triton.jit`, using `tl.load` to read from memory and `tl.store` to write results. Also demonstrate how to configure the launch grid/block size (e.g., using a `grid` lambda or specifying `BLOCK_SIZE`) and then call the kernel like `kernel[grid](..., BLOCK_SIZE=...)` on a sample tensor.
9. **(Hard):** Convert a PyTorch model to a TensorRT engine for deployment. Outline the steps in code: for example, export the model to ONNX format using `torch.onnx.export`, then use NVIDIA’s TensorRT Python API (or Torch-TensorRT) to load that ONNX model and build an optimized TensorRT engine. Finally, show how you would run inference using that TensorRT engine (e.g., by binding input/output and executing the context). Pseudo-code for the TensorRT part is fine — focus on the sequence of steps and any important parameters (like enabling FP16 or setting max workspace size).
10. **(Hard):** Parallelize preprocessing using CPU threads or processes to feed the model faster. For instance, if tokenization on CPU is a bottleneck, demonstrate how you could use Python’s `concurrent.futures.ThreadPoolExecutor` (for I/O-bound tasks) or `multiprocessing.Pool` (for CPU-bound tasks) to tokenize multiple inputs in parallel before batching them. Show a code snippet that takes a list of text strings, splits the work across threads or processes to produce token ID tensors for each, and then stacks them into a batch for the model.
11. **(Hard):** Integrate LoRA (Low-Rank Adaptation) into a model’s layer. Take a pre-trained weight matrix $W$ (for example, the weight of a transformer’s dense layer) and incorporate LoRA matrices into it. Show how you’d create two small trainable matrices $A$ and $B$ (with shapes like `[out_dim, r]` and `[r, in_dim]` for some small rank $r$) and modify the layer’s forward pass to use $W + \alpha \cdot A B$ (with $W$ kept frozen and only $A, B$ learned). Provide code snippets for modifying the model’s `__init__` to add the new parameters and the `forward` to add the LoRA contribution to $W x$.
12. **(Hard):** Demonstrate a case where `torch.jit.trace` is insufficient and `torch.jit.script` is needed. For example, write a small PyTorch function that has an if/else branch or a loop that depends on the *input data* (not just on tensor sizes). Show that tracing this function with a sample input will capture only one branch (thus not generalizing to the other case). Then show how using `torch.jit.script` instead can handle the dynamic control flow. Provide the code for both the traced and scripted versions, highlighting why the traced one is incorrect.
13. **(Hard):** Optimize a generation loop by avoiding CPU synchronization. For example, explain that calling `.item()` on a CUDA tensor forces a GPU-to-CPU sync. Show code where instead of doing `next_token_id = logits.argmax().item()` each iteration (which brings the value to CPU), you keep the computation on the GPU by using tensor operations. For instance, you can obtain the index of the max logit as a 0-dim tensor and use it directly to index into embedding for the next step. By not calling `.item()` in the loop, you allow asynchronous GPU execution to proceed without stall, greatly improving throughput in autoregressive generation.
14. **(Hard):** Perform model-parallel inference across multiple GPUs. For a very large model that cannot fit into one GPU’s memory, illustrate how to split the model’s layers between two GPUs. Provide a code sketch: for example, move `model.encoder` to `cuda:0` and `model.decoder` to `cuda:1`. In the forward pass, send the input to the encoder on GPU0, get its output, then transfer that output tensor to GPU1 to feed into the decoder. Show how you would coordinate the device placements and .to() operations so that each part of the model runs on the intended device.
15. **(Hard):** Use HuggingFace Optimum to accelerate inference. For example, show how to convert a Transformers model to an ONNX runtime for faster CPU/GPU inference using Optimum. You might demonstrate code that uses `Optimum` to export a model to ONNX (`ORTModel.from_pretrained(...)`) or wrap a model with `BetterTransformer`. Include the steps to load the optimized model and perform a sample inference. (This tests familiarity with external optimization tools for PyTorch models.)
16. **(Hard):** Perform post-optimization validation of a model. After applying an optimization (quantization, compilation, etc.), you need to ensure the model’s outputs are still correct. Write code to run the same input through both the original and the optimized model and compare the outputs. For example, compute the mean absolute difference between the two outputs, or if it's an classification/generation model, check that they produce the same top-1 prediction or sequence. This verification code helps confirm that the optimization didn’t degrade the model’s accuracy beyond an acceptable range.
17. **(Hard):** Use HuggingFace Text Generation Inference (TGI) server for serving an LLM. Assume a model is already loaded on a TGI server endpoint — write a small Python client snippet that sends a generation request to the server. For example, use the `requests` library to POST a JSON payload with a prompt and decoding parameters to the TGI HTTP endpoint, then parse the JSON response to get the generated text. (This tests understanding of integrating with an inference server via its API.)
18. **(Hard):** Configure NVIDIA Triton Inference Server for dynamic batching. Describe (or provide) what you would put in the model’s `config.pbtxt` to enable this. For example, specify a `max_batch_size` that the model can handle (e.g., 8 or 16) and add a `dynamic_batching` block with parameters like `max_queue_delay_microseconds` (to set the max wait time for forming a batch). Provide a short snippet of a config showing these settings, and explain that this will allow Triton to automatically batch incoming requests up to the max batch size or timeout.
19. **(Hard):** Implement sinusoidal positional encoding as used in the original Transformer. Write a function that takes a sequence length *L* and model dimension *d_model* and returns a tensor of shape `(L, d_model)` where each position $i$ (0-indexed) has a sinusoidal encoding. Use the formula: for each dimension $j$:

$$

\text{PE}[i, 2j] = \sin \left( \frac{i}{10000^{\frac{2j}{d_{\text{model}}}}} \right)
, 

\text{PE}[i, 2j+1] = \cos \left( \frac{i}{10000^{\frac{2j}{d_{\text{model}}}}} \right)
$$

Implement this calculation in PyTorch (avoiding explicit Python loops if possible).

1. **(Hard):** Refactor a computational Python loop into a vectorized PyTorch operation for speed. For example, suppose you have code that iterates over each element in a tensor to apply a function (which is very slow in Python). Show an example of such a loop and then show how to replace it with equivalent PyTorch tensor operations (which leverage parallelism). Explain in comments how removing Python-loop overhead and using vectorized operations speeds up inference, especially for large tensor computations.
2. **(Hard):** Diagnose and fix a memory leak in an inference loop. For instance, consider a scenario where you append each model output tensor to a Python list for logging or further processing. Write code that simulates this (e.g., a loop doing `outputs.append(model(x))` each time) and explain why this causes GPU memory to balloon (hint: the computation graph is being retained for each output tensor). Then show how to fix it by disabling grad tracking (`with torch.no_grad():` around inference) or detaching/cloning tensors before storing them (so that no reference to the computation graph remains), and by deleting or reusing tensors appropriately.
3. **(Hard):** Work around a part of the model that can’t be compiled by TorchDynamo. If `torch.compile` is failing or falling back for a certain section of your model (for example, a part with unsupported operations or dynamic Python logic), demonstrate how you can use `torch._dynamo.disable` as a context manager around that code block to exclude it from compilation. Provide a code example where you wrap a problematic function or section in `with torch._dynamo.disable():` so that the rest of the model runs under TorchDynamo optimization, but that section will run normally (ensuring the whole model can execute without errors).
4. **(Hard):** Use multiple GPUs to increase throughput via data parallelism. Show how you could replicate a model on two GPUs and split a batch between them for inference. For example, you might use `torch.nn.DataParallel` to automatically split input across GPUs, or do it manually: send half of the input batch to a model on `cuda:0` and the other half to a clone of the model on `cuda:1`, then concatenate the outputs. Provide code that demonstrates this parallel inference across two GPUs and mention how it can almost halve the per-batch latency (ignoring some overhead).
5. **(Medium):** Implement random sampling for text generation with a temperature parameter. At each decoding step, apply a softmax to the model’s logits (you can divide the logits by a temperature τ > 0 to control randomness), then use `torch.multinomial` to sample one token from the resulting probability distribution. Write a code snippet for one step of this process, given `logits` for the current step, and show how changing the temperature affects the sampling outcome (more random vs. more greedy).
6. **(Medium):** Apply weight tying in a language model to reduce parameters. For example, if a model uses an `nn.Embedding` for input tokens and an `nn.Linear` as the output projection for predicting next-token logits, set the linear layer’s weight equal to the embedding matrix so they share weights. Show in code how you would do this (e.g., `model.output_layer.weight = model.input_embed.weight`) and explain that this way the input and output embeddings remain the same, saving memory and typically improving model consistency.
7. **(Medium):** Tune PyTorch’s thread settings for CPU inference. For a model running on CPU (multi-core), show how to configure the number of threads used by PyTorch. For example, use `torch.set_num_threads(n)` and `torch.set_num_interop_threads(m)` to limit or set the parallelism. Provide code setting these values (e.g., to 4 threads), and explain in comments when you might adjust these — such as to prevent thread oversubscription in a multi-model deployment or to optimize throughput on CPU-bound inference workloads.

*each question above is designed to be hands-on and coding-focused, reflecting scenarios a Machine Learning Engineer (Inference) might encounter when optimizing LLM inference in production. Difficulty is labeled from easy fundamentals to challenging performance engineering tasks. The questions emphasize practical reasoning, awareness of inference speed and memory bottlenecks, and the ability to write clean, optimized PyTorch code under real-world constraints.*






