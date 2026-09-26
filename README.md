# mini-vllm

**[English](#english) · [Türkçe](#türkçe)**

A from-scratch LLM inference serving engine in Rust, with KV-cache, continuous batching and paged memory, built for learning.

---

<a id="english"></a>

## 🇬🇧 English

### What is this?

**mini-vllm** is a small inference engine for [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), written from scratch in Rust. It is not meant to compete with [vLLM](https://github.com/vllm-project/vllm) or TensorRT-LLM. The goal is to solve the same production problems they solve, on a small scale and by hand:

- **KV-cache**: avoid recomputing attention over the whole prefix for every new token
- **Continuous batching**: serve many requests at once, adding and removing them mid-flight
- **Paged memory**: manage KV-cache memory in fixed-size blocks to reduce fragmentation (PagedAttention-style)
- **Throughput vs. latency**: measure the trade-offs under concurrent load

All model logic lives in Rust. Python is used only as an HTTP client and for benchmarking.

> 📚 This is also a learning project: it is written by hand, phase by phase, while learning Rust.

### Architecture

```
                ┌──────────────────────────────────────────────┐
  Python        │               Rust engine (single process)   │
  client  ────► │  server (axum)                               │
  (httpx)  HTTP │     │                                        │
          ◄──── │     ▼                                        │
         stream │  scheduler ──► model (forward pass) ──► sampling
                │  (continuous      │                          │
                │   batching)       ▼                          │
                │               kv-cache (naive → paged)       │
                │                   │                          │
                │               tensor-core (safetensors, ops) │
                └──────────────────────────────────────────────┘
```

| Crate | Responsibility |
|---|---|
| `tensor-core` | Load `config.json` and safetensors weights, basic tensor ops |
| `model` | Qwen2 forward pass: RoPE, GQA attention, SwiGLU MLP, sampling |
| `kv-cache` | Naive (growing) cache, later a paged block-based cache |
| `scheduler` | Request queue, static batching, then continuous batching |
| `server` | HTTP API with streaming token output (main binary) |

### How it works

#### 1. Generating text, one token at a time

An LLM does not write a whole answer at once. It predicts **one token** (a word or word piece) at a time. It appends that token to the input and runs again:

```
"The capital of France"            → model → " is"
"The capital of France is"         → model → " Paris"
"The capital of France is Paris"   → model → "."
```

Each run is a **forward pass** through 24 transformer layers. At the end, **sampling** picks the next token: *greedy* takes the most likely one, while *top-k* picks randomly among the k most likely.

#### 2. KV-cache: don't recompute the past

Inside every layer, attention computes a **Key (K)** and a **Value (V)** vector for every token. Without a cache, generating token 100 means recomputing K and V for the 99 tokens before it, even though they have not changed.

The **KV-cache** stores K and V from earlier steps. Each new step computes K/V only for the **single new token** and reuses the rest from the cache:

```
without cache:  step n processes n tokens      → total work grows ~n²
with cache:     step n processes 1 new token   → total work grows ~n
```

The price is memory. For this model the cache costs about **12 KB per token** in bf16 (24 layers × K+V × 2 KV heads × 64 dims × 2 bytes).

#### 3. GQA: a smaller cache for free

Qwen2.5 uses **Grouped-Query Attention**. It has 14 query heads but only **2 key/value heads**, so each K/V head is shared by 7 query heads. The KV-cache is therefore **7× smaller** than with standard multi-head attention, with little loss in quality. This is why `k_proj` and `v_proj` output 128 values (2 × 64) while `q_proj` outputs 896 (14 × 64).

#### 4. Continuous batching: don't wait for the slowest request

Running several requests together as a **batch** uses the hardware far better than running them one by one. With **static batching**, however, the whole batch waits for its longest request:

```
static:       req A ████████████████████   (long)
              req B █████░░░░░░░░░░░░░░░   (done early, slot sits idle)
              req C          waits for the whole batch to finish...

continuous:   req A ████████████████████
              req B █████
              req C      ███████████       (joins as soon as B leaves)
```

**Continuous batching** makes this decision at every step. Finished requests leave the batch right away, and waiting requests take their place. Throughput goes up and short requests are no longer stuck behind long ones.

#### 5. Paged memory: fight fragmentation

A naive KV-cache gives each request one contiguous buffer. We do not know in advance how long an answer will be, so we either reserve too much (wasted memory) or have to reallocate and copy. When many requests of different lengths start and finish, free memory breaks into small unusable gaps. This is called **fragmentation**.

**Paged memory** borrows the idea from operating-system virtual memory. The KV-cache is split into fixed-size **blocks** (for example 16 tokens each). A **page table** maps each request to its blocks, which don't need to be contiguous:

```
physical blocks:  [A0][B0][A1][C0][B1][A2][ free ][ free ]
page table:       A → 0, 2, 5     B → 1, 4     C → 3
```

At most one partly filled block per request is wasted. Freed blocks can be reused right away by any request, so more requests fit in the same memory. This is the core idea behind vLLM's **PagedAttention**.

### Roadmap

| Phase | Crate | Done when | Status |
|---|---|---|---|
| 0: Setup | all | Workspace, dependencies, model download | ✅ Done |
| 1: Model loading | `tensor-core` | Weights load into Rust with correct shapes | 🚧 In progress |
| 2: Forward pass | `model` | A single prompt produces coherent text | ⏳ Planned |
| 3: KV-cache | `kv-cache` (naive) | Cached vs. uncached speed compared | ⏳ Planned |
| 4: Continuous batching | `scheduler` | 2+ concurrent requests of different lengths handled correctly | ⏳ Planned |
| 5: Paged memory | `kv-cache` (paged) | Fixed-size pages, measurably less fragmentation | ⏳ Planned |
| 6: Server + benchmark | `server` + client | Throughput and p50/p95/p99 latency measured under load | ⏳ Planned |

### Tech stack

| Layer | Tools |
|---|---|
| Tensor / model engine | Rust, [`candle`](https://github.com/huggingface/candle) (core, nn, transformers) |
| Tokenizer | [`tokenizers`](https://github.com/huggingface/tokenizers) |
| Async server | `tokio` + `axum` *(Phase 6)* |
| Weights | `safetensors`, downloaded from Hugging Face Hub |
| Client / benchmark | Python 3.13+, `httpx`, managed with [`uv`](https://github.com/astral-sh/uv) |
| Tests | `cargo test` / `pytest` |

### Model

[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): small, dense, runs on CPU.

| Property | Value |
|---|---|
| Layers | 24 |
| Hidden size | 896 |
| Attention heads (Q / KV) | 14 / 2 (Grouped-Query Attention) |
| Head dim | 64 |
| MLP intermediate size | 4864 (SwiGLU) |
| Vocabulary | 151,936 |
| RoPE theta | 1,000,000 |
| Weights | ~988 MB, bf16, single `model.safetensors` |
| Embeddings | Tied (`lm_head` reuses `embed_tokens`) |

### Getting started

**Prerequisites**
- [Rust](https://rustup.rs/) (edition 2024, stable toolchain)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- ~1.5 GB free disk space

**1. Clone**
```bash
git clone https://github.com/fatihkadim/mini-vllm.git
cd mini-vllm
```

**2. Download the model** (to `models/qwen2.5-0.5b-instruct/`, ignored by git)
```bash
uv run scripts/download_model.py
```

**3. Build and test the engine**
```bash
cd engine
cargo build
cargo test
```

**4. Run the server** *(available after Phase 6)*
```bash
cargo run --release -p server
```

**5. Client / benchmark** *(available after Phase 6)*
```bash
cd client
uv sync
uv run mini-vllm-client
```

### Repository layout

```
mini-vllm/
├── engine/                 # Rust workspace
│   └── crates/
│       ├── tensor-core/    # Phase 1: weight loading, basic ops
│       ├── model/          # Phase 2: forward pass, sampling
│       ├── kv-cache/       # Phase 3 & 5: naive and paged cache
│       ├── scheduler/      # Phase 4: batching
│       └── server/         # Phase 6: HTTP API (binary)
├── client/                 # Python client and benchmark (uv project)
├── scripts/
│   └── download_model.py   # Fetch weights from Hugging Face Hub
└── models/                 # Downloaded weights (git-ignored)
```

### Scope

**In scope:** a single small dense decoder model · CPU first (GPU is a stretch goal) · greedy and top-k sampling · naive and paged KV-cache · continuous batching · single node, single process · minimal HTTP API with streaming.

**Out of scope:** training or fine-tuning · multi-GPU or distributed serving · full OpenAI API compatibility · MoE models · quantization · auth, rate limiting, multi-tenancy · autograd or backward pass.

---

<a id="türkçe"></a>

## 🇹🇷 Türkçe

### Bu nedir?

**mini-vllm**, [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) modeli için Rust'ta sıfırdan yazılan küçük bir inference sunucusudur. [vLLM](https://github.com/vllm-project/vllm) veya TensorRT-LLM ile yarışmak için yapılmıyor. Amaç, bu araçların çözdüğü gerçek production problemlerini küçük ölçekte ve elle çözmek:

- **KV-cache:** her yeni token için tüm geçmişin attention'ını baştan hesaplamamak
- **Continuous batching:** birçok isteği aynı anda işlemek, istekleri çalışma sırasında eklemek ve çıkarmak
- **Sayfalı bellek (paged memory):** KV-cache belleğini sabit boyutlu bloklarla yönetip fragmentasyonu azaltmak (PagedAttention tarzı)
- **Throughput ve latency:** eşzamanlı yük altında bu ikisi arasındaki dengeyi ölçmek

Model mantığının tamamı Rust'tadır. Python yalnızca HTTP client ve benchmark için kullanılır.

> 📚 Bu aynı zamanda bir öğrenme projesidir: Rust öğrenilirken faz faz, elle yazılmaktadır.

### Mimari

```
                ┌──────────────────────────────────────────────┐
  Python        │               Rust motoru (tek process)      │
  client  ────► │  server (axum)                               │
  (httpx)  HTTP │     │                                        │
          ◄──── │     ▼                                        │
      streaming │  scheduler ──► model (forward pass) ──► sampling
                │  (continuous      │                          │
                │   batching)       ▼                          │
                │               kv-cache (naif → sayfalı)      │
                │                   │                          │
                │               tensor-core (safetensors, ops) │
                └──────────────────────────────────────────────┘
```

| Crate | Sorumluluk |
|---|---|
| `tensor-core` | `config.json` ve safetensors ağırlıklarını yükleme, temel tensör işlemleri |
| `model` | Qwen2 forward pass: RoPE, GQA attention, SwiGLU MLP, sampling |
| `kv-cache` | Naif (büyüyen) cache, sonra blok tabanlı sayfalı cache |
| `scheduler` | İstek kuyruğu, önce static batching, sonra continuous batching |
| `server` | Streaming token çıktılı HTTP API (ana binary) |

### Nasıl çalışır?

#### 1. Metin üretimi: her seferinde bir token

Bir LLM cevabın tamamını tek seferde yazmaz. Her adımda **bir token** (kelime veya kelime parçası) tahmin eder, bu token'ı girdinin sonuna ekler ve tekrar çalışır:

```
"The capital of France"            → model → " is"
"The capital of France is"         → model → " Paris"
"The capital of France is Paris"   → model → "."
```

Her çalıştırma, 24 transformer katmanından geçen bir **forward pass**'tir. Sonunda **sampling** bir sonraki token'ı seçer: *greedy* en olası token'ı alır, *top-k* ise en olası k token arasından rastgele seçer.

#### 2. KV-cache: geçmişi yeniden hesaplama

Her katmanda attention, her token için bir **Key (K)** ve bir **Value (V)** vektörü hesaplar. Cache olmadan 100. token'ı üretmek için önceki 99 token'ın K ve V'si baştan hesaplanır, oysa bunlar hiç değişmemiştir.

**KV-cache**, önceki adımların K ve V'sini saklar. Her yeni adımda K/V yalnızca **yeni gelen tek token** için hesaplanır, geri kalanı cache'den okunur:

```
cache'siz:  n. adım n token işler         → toplam iş ~n² büyür
cache'li:   n. adım 1 yeni token işler    → toplam iş ~n büyür
```

Bunun bedeli bellektir. Bu modelde cache, bf16'da **token başına yaklaşık 12 KB** tutar (24 katman × K+V × 2 KV head × 64 boyut × 2 byte).

#### 3. GQA: bedavaya küçük cache

Qwen2.5, **Grouped-Query Attention** kullanır. 14 query head'i var ama yalnızca **2 key/value head'i** var, yani her K/V head'i 7 query head tarafından paylaşılır. Bu sayede KV-cache, standart multi-head attention'a göre **7 kat küçük** olur ve kalite neredeyse hiç düşmez. `k_proj` ve `v_proj`'un 128 (2 × 64), `q_proj`'un ise 896 (14 × 64) değer üretmesinin sebebi budur.

#### 4. Continuous batching: en yavaş isteği bekleme

Birden fazla isteği bir **batch** halinde birlikte işlemek, donanımı tek tek işlemekten çok daha verimli kullanır. Ancak **static batching**'de tüm batch, içindeki en uzun isteği bekler:

```
static:       istek A ████████████████████   (uzun)
              istek B █████░░░░░░░░░░░░░░░   (erken bitti, yeri boşta bekliyor)
              istek C          batch'in bitmesini bekliyor...

continuous:   istek A ████████████████████
              istek B █████
              istek C      ███████████       (B çıkar çıkmaz katılıyor)
```

**Continuous batching** bu kararı her adımda verir. Biten istekler batch'ten hemen çıkar, bekleyen istekler onların yerine girer. Throughput artar, kısa istekler de uzunların arkasında beklemez.

#### 5. Sayfalı bellek: fragmentasyonla mücadele

Naif bir KV-cache her isteğe tek parça, bitişik bir bellek alanı verir. Cevabın ne kadar uzun olacağını önceden bilemediğimiz için ya fazla yer ayırırız (bellek israfı) ya da belleği yeniden ayırıp kopyalamak zorunda kalırız. Farklı uzunlukta birçok istek başlayıp bittikçe boş bellek küçük, kullanılamayan parçalara bölünür. Buna **fragmentasyon** denir.

**Sayfalı bellek**, fikrini işletim sistemlerindeki sanal bellekten alır. KV-cache sabit boyutlu **bloklara** bölünür (örneğin her blok 16 token). Bir **page table**, her isteği kendi bloklarına eşler. Blokların bitişik olması gerekmez:

```
fiziksel bloklar:  [A0][B0][A1][C0][B1][A2][ boş ][ boş ]
page table:        A → 0, 2, 5     B → 1, 4     C → 3
```

İstek başına en fazla bir blok yarı dolu kalır. Serbest kalan bloklar hemen başka bir istek tarafından kullanılabilir, böylece aynı belleğe daha fazla istek sığar. vLLM'in **PagedAttention** yaklaşımının temel fikri budur.

### Yol haritası

| Faz | Crate | Bitti sayılır | Durum |
|---|---|---|---|
| 0: Kurulum | hepsi | Workspace, bağımlılıklar, model indirme | ✅ Tamam |
| 1: Model yükleme | `tensor-core` | Ağırlıklar Rust'a doğru shape'lerle yükleniyor | 🚧 Devam ediyor |
| 2: Forward pass | `model` | Tek bir prompt'tan anlamlı metin üretiliyor | ⏳ Planlandı |
| 3: KV-cache | `kv-cache` (naif) | Cache'li ve cache'siz hız karşılaştırıldı | ⏳ Planlandı |
| 4: Continuous batching | `scheduler` | Farklı uzunlukta 2+ eşzamanlı istek doğru işleniyor | ⏳ Planlandı |
| 5: Sayfalı bellek | `kv-cache` (sayfalı) | Sabit boyutlu sayfalar, ölçülebilir şekilde daha az fragmentasyon | ⏳ Planlandı |
| 6: Server + benchmark | `server` + client | Yük altında throughput ve p50/p95/p99 latency ölçüldü | ⏳ Planlandı |

### Teknoloji yığını

| Katman | Araçlar |
|---|---|
| Tensör / model motoru | Rust, [`candle`](https://github.com/huggingface/candle) (core, nn, transformers) |
| Tokenizer | [`tokenizers`](https://github.com/huggingface/tokenizers) |
| Async server | `tokio` + `axum` *(Faz 6)* |
| Ağırlıklar | `safetensors`, Hugging Face Hub'dan indirilir |
| Client / benchmark | Python 3.13+, `httpx`, [`uv`](https://github.com/astral-sh/uv) ile yönetilir |
| Testler | `cargo test` / `pytest` |

### Model

[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): küçük, dense, CPU'da çalışır.

| Özellik | Değer |
|---|---|
| Katman sayısı | 24 |
| Hidden size | 896 |
| Attention head (Q / KV) | 14 / 2 (Grouped-Query Attention) |
| Head boyutu | 64 |
| MLP ara boyutu | 4864 (SwiGLU) |
| Kelime dağarcığı | 151.936 |
| RoPE theta | 1.000.000 |
| Ağırlıklar | ~988 MB, bf16, tek `model.safetensors` |
| Embedding | Paylaşımlı (`lm_head`, `embed_tokens` ağırlığını kullanır) |

### Kurulum ve çalıştırma

**Gereksinimler**
- [Rust](https://rustup.rs/) (edition 2024, stable toolchain)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- ~1,5 GB boş disk alanı

**1. Repoyu klonla**
```bash
git clone https://github.com/fatihkadim/mini-vllm.git
cd mini-vllm
```

**2. Modeli indir** (`models/qwen2.5-0.5b-instruct/` altına iner, git'e eklenmez)
```bash
uv run scripts/download_model.py
```

**3. Motoru derle ve test et**
```bash
cd engine
cargo build
cargo test
```

**4. Server'ı çalıştır** *(Faz 6'dan sonra)*
```bash
cargo run --release -p server
```

**5. Client / benchmark** *(Faz 6'dan sonra)*
```bash
cd client
uv sync
uv run mini-vllm-client
```

### Dosya yapısı

```
mini-vllm/
├── engine/                 # Rust workspace
│   └── crates/
│       ├── tensor-core/    # Faz 1: ağırlık yükleme, temel işlemler
│       ├── model/          # Faz 2: forward pass, sampling
│       ├── kv-cache/       # Faz 3 ve 5: naif ve sayfalı cache
│       ├── scheduler/      # Faz 4: batching
│       └── server/         # Faz 6: HTTP API (binary)
├── client/                 # Python client ve benchmark (uv projesi)
├── scripts/
│   └── download_model.py   # Ağırlıkları Hugging Face Hub'dan indirir
└── models/                 # İndirilen ağırlıklar (git'e eklenmez)
```

### Kapsam

**Kapsam içinde:** tek, küçük, dense bir decoder model · önce CPU (GPU isteğe bağlı hedef) · greedy ve top-k sampling · naif ve sayfalı KV-cache · continuous batching · tek düğüm, tek process · streaming destekli minimal HTTP API.

**Kapsam dışında:** eğitim veya fine-tuning · çoklu GPU veya dağıtık serving · tam OpenAI API uyumluluğu · MoE modeller · quantization · kimlik doğrulama, rate limiting, multi-tenancy · autograd veya backward pass.
