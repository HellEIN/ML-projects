**`For the interactive version you should download the HTML file or run ipynb file locally`**

---

# **K-Nearest Neighbors (KNN) Algorithm - Explained Simply with Mathematical Examples**

## **Introduction**  
K-Nearest Neighbors (KNN) is one of the simplest machine learning algorithms used for **classification** and **regression**. It makes predictions by looking at the **K closest data points** to a given input. Think of it like asking your closest friends for advice—majority opinion is likely the best guess!

---

## **How Does KNN Work?**

### **1. Choose the Number of Neighbors (K)**
- **K** is a number that tells us **how many** nearby data points we should consider when making a decision.
- If **K is too small (e.g., 1 or 3)** → The model becomes sensitive to noise and can misclassify points.
- If **K is too large (e.g., 20 or 50)** → The model might over-generalize and miss finer details.

### **2. Measure Distance Between Points**  
To find the "nearest" neighbors, we need to calculate the **distance** between data points. The most common method is **Euclidean Distance**, which calculates the straight-line distance between two points.

#### **Euclidean Distance Formula**

For two points in a 2D space:

$$
\hat{y} = \arg\max_c \sum_{i=1}^{K} \mathbb{1}(y_i = c)
$$


The Euclidean distance between these points is:

$$
\[
d(A, B) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\]
$$

#### **Example (2D)**  
Let’s say we have:
- Point A: \( (3, 4) \)
- Point B: \( (7, 1) \)

The distance between A and B is:
\[
d(A, B) = \sqrt{(7-3)^2 + (1-4)^2} = \sqrt{4^2 + (-3)^2} = \sqrt{16 + 9} = \sqrt{25} = 5
\]

If the dataset has **more than two features**, the formula extends to:
\[
d(A, B) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]
where \( n \) is the number of features.

#### **Example in 3D Space**
If:
- \( A(1, 2, 3) \)
- \( B(4, 6, 8) \)

Then:
\[
d(A, B) = \sqrt{(4-1)^2 + (6-2)^2 + (8-3)^2} = \sqrt{3^2 + 4^2 + 5^2} = \sqrt{9 + 16 + 25} = \sqrt{50} \approx 7.07
\]

---

### **3. Find the K Nearest Neighbors**
- After calculating the distance from the test point to **every** point in the dataset, we **sort the distances** and pick the **K closest points**.

### **4. Make a Prediction**

#### **For Classification (Majority Voting):**

Once we have the K nearest neighbors, the prediction is made based on the **majority class** among them. Mathematically, the predicted class is:
\[
\hat{y} = \arg\max_c \sum_{i=1}^{K} \mathbb{1}(y_i = c)
\]
Where:
- \( c \) represents a class label.
- \( \mathbb{1}(y_i = c) \) is **1** if the \( i^{th} \) neighbor belongs to class \( c \), otherwise **0**.
- We choose the class with the **most votes**.

#### **Example (Classification)**
Imagine we have **three classes: Apple 🍎, Banana 🍌, and Cherry 🍒**, and we set \( K = 5 \). The nearest neighbors are:

| Neighbor | Class     |
|----------|-----------|
| 1st      | 🍎 Apple  |
| 2nd      | 🍌 Banana |
| 3rd      | 🍎 Apple  |
| 4th      | 🍎 Apple  |
| 5th      | 🍌 Banana |

Count of each class:
- 🍎 Apple = **3**
- 🍌 Banana = **2**
- 🍒 Cherry = **0**

Since **Apple** appears the most (3 times), the model predicts **🍎 Apple** as the class.

---

#### **For Regression (Averaging Neighbor Values):**

For regression, instead of taking a majority vote, we calculate the **average** of the K nearest neighbors' values. The prediction is given by:
\[
\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_i
\]

#### **Example (Regression)**
Let’s say we want to predict the **price of a house** based on its features. We set \( K = 3 \), and the nearest houses have prices:

| Neighbor | Price ($) |
|----------|-----------|
| 1st      | 250,000   |
| 2nd      | 270,000   |
| 3rd      | 230,000   |

The predicted price is:
\[
\hat{y} = \frac{250,000 + 270,000 + 230,000}{3} = \frac{750,000}{3} = 250,000
\]

So, our model predicts the house will cost **$250,000**.

---

### **5. Choosing the Best K Value**
- **Small K (e.g., 1, 3, 5)** → The model reacts to noise and may misclassify points.
- **Large K (e.g., 50, 100)** → The model generalizes too much and may overlook finer details.
- A good strategy is to **try different K values** aSure! Here's the complete K-Nearest Neighbors (KNN) explanation with the LaTeX formulas formatted for GitHub `README.md`:

---

# **K-Nearest Neighbors (KNN) Algorithm - Explained Simply with Mathematical Examples**

## **Introduction**  
K-Nearest Neighbors (KNN) is one of the simplest machine learning algorithms used for **classification** and **regression**. It makes predictions by looking at the **K closest data points** to a given input. Think of it like asking your closest friends for advice—majority opinion is likely the best guess!

---

## **How Does KNN Work?**

### **1. Choose the Number of Neighbors (K)**
- **K** is a number that tells us **how many** nearby data points we should consider when making a decision.
- If **K is too small (e.g., 1 or 3)** → The model becomes sensitive to noise and can misclassify points.
- If **K is too large (e.g., 20 or 50)** → The model might over-generalize and miss finer details.

### **2. Measure Distance Between Points**  
To find the "nearest" neighbors, we need to calculate the **distance** between data points. The most common method is **Euclidean Distance**, which calculates the straight-line distance between two points.

#### **Euclidean Distance Formula**

For two points in a 2D space:
$$
A(x_1, y_1), \quad B(x_2, y_2)
$$
The Euclidean distance between these points is:
$$
d(A, B) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

#### **Example (2D)**  
Let’s say we have:
- Point A: \( (3, 4) \)
- Point B: \( (7, 1) \)

The distance between A and B is:
$$
d(A, B) = \sqrt{(7-3)^2 + (1-4)^2} = \sqrt{4^2 + (-3)^2} = \sqrt{16 + 9} = \sqrt{25} = 5
$$

If the dataset has **more than two features**, the formula extends to:
$$
d(A, B) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$
where \( n \) is the number of features.

#### **Example in 3D Space**
If:
- \( A(1, 2, 3) \)
- \( B(4, 6, 8) \)

Then:
$$
d(A, B) = \sqrt{(4-1)^2 + (6-2)^2 + (8-3)^2} = \sqrt{3^2 + 4^2 + 5^2} = \sqrt{9 + 16 + 25} = \sqrt{50} \approx 7.07
$$

---

### **3. Find the K Nearest Neighbors**
- After calculating the distance from the test point to **every** point in the dataset, we **sort the distances** and pick the **K closest points**.

### **4. Make a Prediction**

#### **For Classification (Majority Voting):**

Once we have the K nearest neighbors, the prediction is made based on the **majority class** among them. Mathematically, the predicted class is:
$$
\hat{y} = \arg\max_c \sum_{i=1}^{K} \mathbb{1}(y_i = c)
$$
Where:
- \( c \) represents a class label.
- \( \mathbb{1}(y_i = c) \) is **1** if the \( i^{th} \) neighbor belongs to class \( c \), otherwise **0**.
- We choose the class with the **most votes**.

#### **Example (Classification)**
Imagine we have **three classes: Apple 🍎, Banana 🍌, and Cherry 🍒**, and we set \( K = 5 \). The nearest neighbors are:

| Neighbor | Class     |
|----------|-----------|
| 1st      | 🍎 Apple  |
| 2nd      | 🍌 Banana |
| 3rd      | 🍎 Apple  |
| 4th      | 🍎 Apple  |
| 5th      | 🍌 Banana |

Count of each class:
- 🍎 Apple = **3**
- 🍌 Banana = **2**
- 🍒 Cherry = **0**

Since **Apple** appears the most (3 times), the model predicts **🍎 Apple** as the class.

---

#### **For Regression (Averaging Neighbor Values):**

For regression, instead of taking a majority vote, we calculate the **average** of the K nearest neighbors' values. The prediction is given by:
$$
\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_i
$$

#### **Example (Regression)**
Let’s say we want to predict the **price of a house** based on its features. We set \( K = 3 \), and the nearest houses have prices:

| Neighbor | Price ($) |
|----------|-----------|
| 1st      | 250,000   |
| 2nd      | 270,000   |
| 3rd      | 230,000   |

The predicted price is:
$$
\hat{y} = \frac{250,000 + 270,000 + 230,000}{3} = \frac{750,000}{3} = 250,000
$$

So, our model predicts the house will cost **$250,000**.

---

### **5. Choosing the Best K Value**
- **Small K (e.g., 1, 3, 5)** → The model reacts to noise and may misclassify points.
- **Large K (e.g., 50, 100)** → The model generalizes too much and may overlook finer details.
- A good strategy is to **try different K values** and use cross-validation to choose the best K.

---

### **6. Weighted KNN (Giving More Importance to Closer Points)**
Instead of treating all K neighbors equally, we can **give more weight to closer neighbors** using **inverse distance weighting**:

#### **Formula for Weighted KNN**
$$
\hat{y} = \frac{\sum_{i=1}^{K} w_i y_i}{\sum_{i=1}^{K} w_i}
$$
Where:
$$
w_i = \frac{1}{d_i}
$$
Here, \( w_i \) is the weight, and \( d_i \) is the distance between the test point and the \( i^{th} \) neighbor.

#### **Example (Weighted KNN)**
Suppose we set \( K = 3 \) and have the following house prices with their distances:

| Neighbor | Price ($) | Distance \( d_i \) | Weight (\( w_i = \frac{1}{d_i} \)) |
|----------|-----------|--------------------|-------------------------------------|
| 1st      | 250,000   | 0.5                | 2.0                                 |
| 2nd      | 270,000   | 1.0                | 1.0                                 |
| 3rd      | 230,000   | 2.0                | 0.5                                 |

Now, we calculate:
$$
\hat{y} = \frac{(2.0 \times 250,000) + (1.0 \times 270,000) + (0.5 \times 230,000)}{2.0 + 1.0 + 0.5}
$$
$$
= \frac{500,000 + 270,000 + 115,000}{3.5} = \frac{885,000}{3.5} \approx 252,857
$$

So, the predicted house price is **$252,857**.

---

## **Pros and Cons of KNN**

### ✅ **Advantages:**
- Simple and easy to understand.
- No model training (lazy learning).
- Works well for small datasets.

### ❌ **Disadvantages:**
- Slow for large datasets (because it checks all points).
- Affected by irrelevant features (so we must scale data).
- Struggles with imbalanced data (where one class is much larger than the other).

---

## **Final Thoughts**
- **KNN** is a **lazy learning** algorithm because it doesn’t create a model during the training phase; it stores the data and uses it when making predictions.
- **Feature scaling (normalization)** is important because KNN depends on distance calculations.
- **Weighted KNN** improves accuracy by giving more importance to closer neighbors.

---

This is the final version of your KNN explanation for `README.md`, and now the formulas should render correctly on GitHub! Let me know if you need any further help.
