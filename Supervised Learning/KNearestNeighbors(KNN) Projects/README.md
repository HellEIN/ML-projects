**`For the interactive version you should download the HTML file or run ipynb file locally`**

# **K-Nearest Neighbors (KNN) Algorithm - A Simple Yet Powerful Approach**  

## **Introduction**  
K-Nearest Neighbors (**KNN**) is a simple yet effective machine learning algorithm used for **classification** and **regression**. It makes predictions by **finding the K nearest data points** and using them to determine the output.  

Think of it like asking your closest friends for advice—if most of them agree on something, that’s probably the best answer!  

---

## **How KNN Works**  

### **1. Choose the Number of Neighbors (K)**  
- \( K \) is a number that tells us how many nearby data points we should consider.  
- **Small \( K \) (e.g., 1 or 3):** The model is sensitive to noise and may overfit.  
- **Large \( K \) (e.g., 20 or 50):** The model smooths predictions but may lose important details.  
- The best \( K \) is usually found using **cross-validation**.  

### **2. Measure Distance Between Points**  
To find the **closest** neighbors, we calculate the **distance** between data points. The most common method is **Euclidean Distance**, which measures the straight-line distance between two points.  

#### **Formula for Euclidean Distance**  
For two points \( A(x_1, y_1) \) and \( B(x_2, y_2) \), the distance is:  
\[
d(A, B) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\]  

##### **Example**  
If we have:  
- \( A(3, 4) \)  
- \( B(7, 1) \)  

The distance is:  
\[
d = \sqrt{(7-3)^2 + (1-4)^2} = \sqrt{16 + 9} = \sqrt{25} = 5
\]  

In a dataset with multiple features (higher dimensions), the formula extends to:  
\[
d(A, B) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]  
where \( n \) is the number of features.  

##### **Example in 3D Space**  
If  
- \( A(1, 2, 3) \)  
- \( B(4, 6, 8) \)  

Then,  
\[
d(A, B) = \sqrt{(4-1)^2 + (6-2)^2 + (8-3)^2}
\]  
\[
= \sqrt{9 + 16 + 25} = \sqrt{50} \approx 7.07
\]  

### **3. Find the K Nearest Neighbors**  
- Compute the distance between the test point and **all** points in the dataset.  
- **Sort the distances** in ascending order.  
- Select the **K closest** points.  

### **4. Make a Prediction**  

#### **KNN for Classification (Majority Voting)**  
- Count how many of the K neighbors belong to each class.  
- The **most frequent class** is the predicted class.  

Mathematically:  
\[
\hat{y} = \arg\max_c \sum_{i=1}^{K} \mathbb{1}(y_i = c)
\]  
where:  
- \( c \) represents a class label.  
- \( \mathbb{1}(y_i = c) \) is **1** if the neighbor belongs to class \( c \), otherwise **0**.  
- We pick the class with the **highest count**.  

##### **Example**  
Imagine we have **three classes: Apple 🍎, Banana 🍌, and Cherry 🍒**, and we set \( K = 5 \). The nearest neighbors are:  

| Neighbor | Class |
|----------|-------|
| 1st      | 🍎 Apple |
| 2nd      | 🍌 Banana |
| 3rd      | 🍎 Apple |
| 4th      | 🍎 Apple |
| 5th      | 🍌 Banana |

Count of each class:  
- 🍎 Apple = **3**  
- 🍌 Banana = **2**  
- 🍒 Cherry = **0**  

Since **Apple appears the most (3 times)**, the model predicts 🍎 **Apple** as the class.  

#### **KNN for Regression (Averaging Neighbor Values)**  
- Instead of choosing the most frequent class, we **average** the K neighbors' values.  

Mathematically:  
\[
\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_i
\]  

##### **Example**  
Suppose we want to predict the **price of a house** based on its features. We set \( K = 3 \), and the nearest houses have prices:  

| Neighbor | Price ($) |
|----------|---------|
| 1st      | 250,000 |
| 2nd      | 270,000 |
| 3rd      | 230,000 |

The predicted price is:  
\[
\hat{y} = \frac{250,000 + 270,000 + 230,000}{3}
\]  
\[
= \frac{750,000}{3} = 250,000
\]  

So, our model predicts the house will cost **$250,000**.  

---

## **Weighted KNN (Giving More Importance to Closer Points)**  

Instead of treating all K neighbors equally, we **give more weight to closer neighbors** using **inverse distance weighting**:  

### **Formula for Weighted KNN**  
\[
\hat{y} = \frac{\sum_{i=1}^{K} w_i y_i}{\sum_{i=1}^{K} w_i}
\]  
where:  
\[
w_i = \frac{1}{d_i}
\]  

##### **Example**  
Suppose we set \( K = 3 \) and have the following house prices with their distances:  

| Neighbor | Price ($) | Distance | Weight (\( w_i = \frac{1}{d_i} \)) |
|----------|----------|----------|----------------|
| 1st      | 250,000  | 0.5      | 2.0            |
| 2nd      | 270,000  | 1.0      | 1.0            |
| 3rd      | 230,000  | 2.0      | 0.5            |

Now, we calculate:  
\[
\hat{y} = \frac{(2.0 \times 250,000) + (1.0 \times 270,000) + (0.5 \times 230,000)}{2.0 + 1.0 + 0.5}
\]  
\[
= \frac{500,000 + 270,000 + 115,000}{3.5} = \frac{885,000}{3.5} \approx 252,857
\]  

So, the predicted house price is **$252,857**.  

---

## **Pros and Cons of KNN**  

✅ **Advantages:**  
✔️ Simple and easy to implement.  
✔️ No training phase (just store data and make predictions).  
✔️ Works well for small datasets.  

❌ **Disadvantages:**  
- Slow when dataset is large (computes distance for every point).  
- Sensitive to irrelevant features (feature scaling is needed).  
- Struggles with imbalanced data.  

---

## **Final Notes**  
- KNN is a **lazy learning** algorithm—it doesn't train a model but makes predictions on the fly.  
- **Feature scaling (like normalization)** is crucial because KNN relies on distance calculations.  
- **Weighted KNN** improves accuracy by giving more importance to closer neighbors.  

