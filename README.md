# 🧩 NYT Connections – Automatic Solver

This project reframes the *NYT Connections* puzzle as a **machine learning problem**, where the goal is to automatically identify the four semantically related words within each daily puzzle set.

---

## 🧠 What Is *Connections*?

*Connections* is a daily word-grouping puzzle published by *The New York Times*, where players must find four disjoint groups of four words, each linked by a hidden theme. The player has four attempts to solve the puzzle — if all guesses are used, the solution is revealed.

If you haven’t played before, you can try it here:  
👉 https://www.nytimes.com/games/connections

---

## 🎯 Why Automate *Connections*?

The puzzle’s objective — splitting 16 words into 4 semantically coherent groups — is flexible and open-ended, ranging from simple lexical similarity to abstract themes (e.g., “titles of Matt Damon movies”).  

This makes *Connections* an excellent testbed for exploring:
- Word representation strategies (semantic, phonographic, or knowledge-based),
- Multi-modal reasoning (e.g., cultural, phonetic, and contextual cues),
- Different machine-learning architectures,
- And ranking methods that balance semantic similarity and structural constraints.

Beyond that, it’s just a fun way to apply machine learning to a challenging, creative reasoning problem. I aim to continue to flesh out this project with different architectures/approaches over the years and develop a benchmark for comparison.

---

## ⚙️ Current Approach: Set Transformer + Listwise Learning-to-Rank (LTR)

### Model Architecture

Because *Connections* depends on contextual relationships between words that vary per puzzle, simple embedding-based scorers (like an MLP) struggle to generalize.  

To address this, I use a **Set Transformer** architecture — a transformer variant that is **order-invariant**, making it naturally suited for reasoning over unordered word sets. The self-attention mechanism helps learn the *connections* among items within each puzzle.

**Levels of Application:**
1. **Global (16-word set):** Learn contextualized embeddings for all items.  
2. **Group-level (4-word subsets):** Use attention to generate a single vector representation per candidate group, which is then scored.

---

### Training Pipeline

1. **Train Baseline:** Simple MLP scorer (static word embeddings).  
2. **Add Attention:** Global set attention + MLP scorer.  
3. **Full Model:** Global attention + Set Transformer encoder–decoder + MLP scorer.

This incremental design ensures each architectural enhancement provides measurable improvements.

**Note**: 1 and 2 are implemented so far, but with limited tuning.

---

### 🧩 Listwise Learning-to-Rank (LTR)

LTR learns to assign scores to items in a list such that their relative order reflects true relevance. Unlike pairwise losses, **listwise losses** directly optimize metrics like **NDCG**, making them ideal for ranking tasks such as recommendation systems and search re-ranking.

#### Application to *Connections*

Each puzzle contains 16 words, so all 4-word combinations (1820 total) are computationally manageable on modern GPUs.  
I frame the problem as **graded relevance**:

| Group Composition | Relevance (y) |
|--------------------|---------------|
| All 4 from same group | 4 |
| 3 of 4 from same group | 3 |
| 2 of 4 from same group | 2 |
| All different groups | 1 |

The model learns to rank these combinations such that true groups appear near the top.

---

### 💡 Why LTR?

While many approaches are possible (LLM fine-tuning, contrastive clustering, differentiable grouping, etc.), LTR offers a **balanced mix of efficiency, interpretability, and scalability**:

- **Dense supervision:** Each list encodes rich relative information.  
- **Natural multi-guess structure:** If one guess is wrong, subsequent ranks can be tried.  
- **Process of elimination:** Correct guesses reduce the search space (similar to how humans solve the puzzle).
- **Practicality:** Easier to implement and tune with limited data.

Additionally, I have prior research experience with LTR methods, making this a practical and extensible starting point.

---

## ⚠️ Current Limitations

This is an early-stage, ongoing project — both practical and conceptual limitations exist:

- **Limited Dataset (~800 puzzles):**
  - Slowly grows over time (1 new puzzle per day).
- **Featurization:**
  - Currently uses only static semantic embeddings (**This is likely the greatest limitation as of now**)
  - Future work will explore hybrid features combining:
    - Pretrained contextual embeddings (e.g., MiniLM, Instructor)
    - Structured knowledge graph features
    - Phonetic or morphological similarity cues
- **Nebulous Definition of “Connections”:**
  - The concept of a “connection” is inherently open-ended and multi-modal (e.g., semantic, cultural, phonetic, or referential)
  - A single model may struggle to capture the extreme diversity of connection types, particularly when trap groupings exist that are superficially similar but contextually incorrect
  - Addressing this will likely require multi-embedding fusion or ensemble reasoning approaches

---

## 🚧 Next Steps

- Integrate **attention-based** and **set-transformer** models
- Explore **knowledge-based** or **phonetic** signals for multi-modal reasoning
- Evaluate **precision@k**, **MRR**, and group-level accuracy metrics
- Tune model **hyperparameters** 
- Explore other **architectures** (e.g., GNN-based, reinforcement learning, set building)
- Build an interactive **demo** that shows model guesses on real puzzles

