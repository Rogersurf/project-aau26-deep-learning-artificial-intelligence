#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


# import necessary libraries
import numpy as np  # Matrix and vector computation package
import pandas as pd
import matplotlib.pyplot as plt  # Plotting library
from tqdm import tqdm_notebook


# In[2]:


# load dataset
data = pd.read_csv('https://raw.githubusercontent.com/aaubs/ds-master/main/data/Swedish_Auto_Insurance_dataset.csv')


# In[3]:


# Using sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data_s = pd.DataFrame(data_scaled, columns=data.columns)


# In[4]:


data_s.head()


# In[5]:


w0 = 8
L = 42


# In[6]:


data_s.iloc[0], data_s.iloc[1], data_s.iloc[2]


# # Sample 0

# In[7]:


# Forward Pass
# ŷ₁ = w₀ · X₁
y_hat0 = w0 * data_s.iloc[0]["X"]
y_hat0


# In[8]:


# Error
# wx - y
error0 = y_hat0 - data_s.iloc[0]["Y"]
error0


# In[9]:


# Gradient
# ∂L/∂w = 2 · x · (wx − y)
grad0 = 2 * data_s.iloc[0]["X"] * error0
grad0


# In[10]:


# Weight update
# w₁ = w₀ − α · gradient
w1 = w0 - L * grad0
w1


# 📌 Comment:
# The prediction is much larger than the target, producing a large positive gradient, so the weight is strongly decreased.

# # Sample 1

# In[11]:


# Forward Pass
# ŷ₁ = w₀ · X₁
y_hat1 = w1 * data_s.iloc[1]["X"]
y_hat1


# In[12]:


# Error
# wx - y
error1 = y_hat1 - data_s.iloc[1]["Y"]
error1


# In[13]:


# Gradient
# ∂L/∂w = 2 · x · (wx − y)
grad1 = 2 * data_s.iloc[1]["X"] * error1
grad1


# In[14]:


# Weight update
# w₁ = w₀ − α · gradient
w2 = w1 - L * grad1
w2


# 📌 Comment:
# After the first update, the weight becomes a large negative value. This causes the prediction to be extremely far from the target, resulting in a very large negative error and gradient. Consequently, the weight update overshoots in the opposite direction, leading to a huge positive weight. This illustrates how large learning rates and unscaled features can cause unstable and divergent updates in stochastic gradient descent.

# # Sample 2

# In[15]:


# Forward Pass
# ŷ₁ = w₀ · X₁
y_hat2 = w2 * data_s.iloc[2]["X"]
y_hat2


# In[16]:


# Error
# wx - y
error2 = y_hat2 - data_s.iloc[2]["Y"]
error2


# In[17]:


# Gradient
# ∂L/∂w = 2 · x · (wx − y)
grad2 = 2 * data_s.iloc[2]["X"] * error2
grad2


# In[18]:


# Weight update
# w₁ = w₀ − α · gradient
w3 = w2 - L * grad2
w3


# 📌 Comment:
# After the second update, the weight has grown to an extremely large positive value. This leads to an enormous prediction and error for the third sample, producing a massive gradient. As a result, the weight update overshoots again and flips to a very large negative value. This behavior clearly demonstrates gradient explosion caused by unnormalized input features combined with a high learning rate when using stochastic gradient descent.

# In[19]:


import pandas as pd

# Build the results table manually (SGD, sample by sample)
results = {
    "Sample": [0, 1, 2],
    "w_old": [w0, w1, w2],
    "x": [
        data_s.iloc[0]["X"],
        data_s.iloc[1]["X"],
        data_s.iloc[2]["X"]
    ],
    "y_hat": [y_hat0, y_hat1, y_hat2],
    "gradient (dL/dw)": [grad0, grad1, grad2],
    "w_new": [w1, w2, w3]
}

sgd_table = pd.DataFrame(results)

sgd_table


# # 🧠 Part B: Attention Contextualization

# In[21]:


# Sentence A (animal)

sentenceA = "the bat flew at night."


# In[24]:


wordsA = sentenceA.strip().lower().split()
wordsA


# In[26]:


# Sentence B (object)

sentenceB = "he swung the bat hard."


# In[27]:


wordsB = sentenceB.strip().lower().split()
wordsB


# In[38]:


import numpy as np
from numpy.linalg import norm

tokens = list(dict.fromkeys(wordsA + wordsB))  # remove duplicados, preserva ordem
tokens


# In[39]:


X = np.array([
    [0.1, 0.0],
    [0.4, 0.6],  # bat
    [0.0, 0.8],
    [0.1, 0.1],
    [0.0, 0.9],
    [0.2, 0.1],
    [0.8, 0.0],
    [0.7, 0.2],
])


# In[40]:


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def euclidean_distance(a, b):
    return norm(a - b)


# In[41]:


embeddings = {
    "the":   np.array([0.1, 0.0]),
    "bat":   np.array([0.5, 0.5]),  # MESMO vetor nas duas frases
    "flew":  np.array([0.0, 0.6]),
    "at":    np.array([0.1, 0.1]),
    "night": np.array([0.0, 0.8]),
    "he":    np.array([0.2, 0.1]),
    "swung": np.array([0.6, 0.0]),
    "hard":  np.array([0.7, 0.2]),
}


# In[42]:


sentence_A = ["the", "bat", "flew", "at", "night"]   # animal
sentence_B = ["he", "swung", "bat", "hard"]          # objeto


# In[43]:


W_Q = np.array([[1.0, 0.0],
                [0.0, 1.0]])

W_K = np.array([[0.8, 0.2],
                [0.2, 0.8]])

W_V = np.array([[1.2, 0.0],
                [0.0, 0.8]])


# In[44]:


def self_attention(sentence, embeddings):
    E = np.stack([embeddings[token] for token in sentence])

    Q = E @ W_Q
    K = E @ W_K
    V = E @ W_V

    d = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d)
    A = np.apply_along_axis(softmax, 1, scores)

    E_context = A @ V
    return E_context, A


# In[ ]:




