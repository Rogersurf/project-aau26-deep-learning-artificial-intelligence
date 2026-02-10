# Portfolio Assignment 1 (M4)
## SGD Mechanics & Attention Contextualization

**Course:** Applied Deep Learning  
**Module:** M4  
**Deadline:** Monday, 10th Feb | 12:00  
**Instructor:** Hamid B. (hamidb@business.aau.dk)
**Student/Group:** <https://github.com/Rogersurf>

---

## 📌 Overview

This assignment is divided into two independent parts:

- **Part A – Manual Stochastic Gradient Descent (SGD)**  
  Manually traces the optimization process of a simple linear model using the first three samples of a dataset.

- **Part B – Attention Contextualization**  
  Demonstrates how self-attention mathematically contextualizes word meanings using static embeddings and a homonym example.

Both parts are implemented in Python and can be executed sequentially.

---

## 🔢 Part A: Manual SGD (First 3 Samples)

### Description
This part demystifies the training process by manually computing:

1. Forward pass  
2. Loss  
3. Gradient  
4. Weight update  

using a simple linear regression model:

\[
\hat{y} = x \cdot w
\]

### Dataset
- **Swedish Auto Insurance Dataset**
- Loaded directly from a public GitHub URL
- Features are scaled using Min-Max normalization

### Hyperparameters (Student-specific)
- Initial weight `w₀`
- Learning rate `L`

Each SGD step is computed **manually** and verified using code.  
A summary table is generated to show all intermediate values.

---

## 🧠 Part B: Attention Contextualization

### Description
Static word embeddings assign a single vector per token, which fails to differentiate meanings of homonyms.  
This part demonstrates how **self-attention** produces different contextual representations for the same word.

### Task
- Homonym used: **"bat"**
- Two sentences with different meanings:
  - *The bat flew at night* (animal)
  - *He swung the bat hard* (object)

### Method
- Initialize small **2D static embeddings**
- Compute self-attention using:
  
\[
A = \text{softmax}(QK^T), \quad \text{where } Q = K = V = E
\]

- Extract the contextualized embedding of `"bat"` from each sentence

### Validation
- **Cosine Similarity**
- **Euclidean Distance**

These metrics demonstrate that the same static embedding results in different contextual representations depending on surrounding words.

---

## ▶️ How to Run the Code

### 1️⃣ Clone the repository

git clone <https://github.com/Rogersurf/project-aau26-deep-learning-artificial-intelligence.git>
cd <project-aau26-deep-learning-artificial-intelligence/assignments>

### 2️⃣ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Run the script

#### If using the converted Python script:

python assignment01(M4).py


#### Or open and run the notebook:

jupyter notebook