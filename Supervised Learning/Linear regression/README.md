**`For the interactive version you should download the HTML file or run ipynb file locally`**

# **Linear Regression - A Simple Explanation with Math**

## **Introduction**  
**Linear Regression** is one of the most basic and widely used algorithms for **predicting continuous values** (like price, temperature, or salary) based on input features. It assumes a **linear relationship** between the input (independent) variables and the output (dependent) variable.

The goal is to fit a **straight line** (in 2D) or a **hyperplane** (in higher dimensions) that best represents the data, allowing us to predict the dependent variable from the independent variables.

---

## **How Does Linear Regression Work?**

### **1. The Line Equation**
Linear Regression fits a **line** to the data points. In simple linear regression, we use the equation of a line: 


$$\
y = \beta_0 + \beta_1 x + \epsilon
\$$

where:  
- $$\( y \)$$ is the predicted output (dependent variable).  
- $$\( \beta_0 \)$$ is the **intercept** of the line (the value of $$\( y \)$$ when $$\( x = 0 \)$$).  
- $$\( \beta_1 \)$$ is the **slope** of the line (how much $$\( y \)$$ changes for a unit change in $$\( x \)$$).  
- $$\( x \)$$ is the input feature (independent variable).  
- $$\( \epsilon \)$$ is the error term, representing the difference between the predicted and actual values.

In **multiple linear regression** (when we have more than one input feature), the equation becomes:

$$\
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
\$$

where $$\( x_1, x_2, \dots, x_n \)$$ are the input features, and $$\( \beta_1, \beta_2, \dots, \beta_n \)$$ are the coefficients for each feature.

### **2. Fitting the Model (Finding the Best Line)**  
To find the best-fitting line, we use the **Least Squares Method**, which minimizes the difference between the actual values and the predicted values (also called the **residuals**).  

The key idea is to find the line that minimizes the sum of the squared differences between the predicted values $$\( \hat{y} \)$$ and the actual values $$\( y \)$$. This is done by minimizing the **cost function** (also called **loss function**):

$$\
J(\beta_0, \beta_1) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
\$$

where:
- $$\( m \)$$ is the number of data points.
- $$\( y_i \)$$ is the actual value for the $$\( i \)$$-th data point.
- $$\( \hat{y}_i \)$$ is the predicted value for the $$\( i \)$$-th data point, which is calculated using the equation:  

  $$\
  \hat{y}_i = \beta_0 + \beta_1 x_i
  \$$

The objective is to find $$\( \beta_0 \)$$ and $$\( \beta_1 \)$$ (and for multiple regression, $$\( \beta_2, \dots, \beta_n \)$$) that **minimize** this cost function.

### **3. Finding the Parameters (Coefficients)**
To minimize the cost function, we use **Gradient Descent** or **Normal Equation**. Let’s go over both methods briefly:

#### **Gradient Descent**
Gradient Descent is an optimization algorithm used to find the minimum of the cost function. It works by updating the coefficients iteratively in the direction that reduces the cost function. The update rule is:

$$\
\beta_j := \beta_j - \alpha \cdot \frac{\partial J(\beta_0, \beta_1)}{\partial \beta_j}
\$$

where:  
- $$\( \beta_j \)$$ is the parameter (coefficient) to be updated.  
- $$\( \alpha \)$$ is the **learning rate**, which controls how big each update step should be.  
- $$\( \frac{\partial J}{\partial \beta_j} \)$$ is the **partial derivative** of the cost function with respect to $$\( \beta_j \)$$, representing the slope of the cost function at the current value of $$\( \beta_j \)$$.

#### **Normal Equation**
An alternative method to find the optimal coefficients is using the **Normal Equation**. For simple linear regression, the formula to find $$\( \beta_1 \)$$ and $$\( \beta_0 \)$$ is:

$$\
\beta_1 = \frac{m \sum_{i=1}^{m} x_i y_i - \sum_{i=1}^{m} x_i \sum_{i=1}^{m} y_i}{m \sum_{i=1}^{m} x_i^2 - (\sum_{i=1}^{m} x_i)^2}
\$$

$$\
\beta_0 = \frac{1}{m} \sum_{i=1}^{m} y_i - \beta_1 \cdot \frac{1}{m} \sum_{i=1}^{m} x_i
\$$

These formulas give the values of $$\( \beta_1 \)$$ and $$\( \beta_0 \)$$ that minimize the cost function.

### **4. Making Predictions**
Once we’ve found the best-fitting line (with the coefficients $$\( \beta_0, \beta_1, \dots \)$$), we can make predictions for new data points by simply plugging in the values of $$\( x_1, x_2, \dots, x_n \)$$ into the equation.

$$\
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
\$$

---

## **Example: Simple Linear Regression**
Let’s walk through a basic example of **simple linear regression** with one input feature.

### **Dataset**  
| Hours of Study ($$\( x \)$$) | Score ($$\( y \)$$) |
|-------------------------|----------------|
| 1                       | 50             |
| 2                       | 55             |
| 3                       | 60             |
| 4                       | 65             |
| 5                       | 70             |

Here, we want to predict the **score** based on the **hours of study**.

### **Step 1: Calculate the Slope $$\( \beta_1 \)$$ and Intercept $$\( \beta_0 \)$$**
Using the **Normal Equation**, we first calculate the necessary sums:

$$\
\sum x_i = 1 + 2 + 3 + 4 + 5 = 15, \quad \sum y_i = 50 + 55 + 60 + 65 + 70 = 300
\$$

$$\
\sum x_i^2 = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55, \quad \sum x_i y_i = (1 \times 50) + (2 \times 55) + (3 \times 60) + (4 \times 65) + (5 \times 70) = 650
\$$

Now we can calculate $$\( \beta_1 \)$$ and $$\( \beta_0 \)$$:

$$\
\beta_1 = \frac{5(650) - (15)(300)}{5(55) - (15)^2} = \frac{3250 - 4500}{275 - 225} = \frac{-1250}{50} = -25
\$$

$$\
\beta_0 = \frac{300}{5} - (-25)(\frac{15}{5}) = 60 + 75 = 135
\$$

So the **line equation** becomes:

\$$
y = 135 - 25x
\$$

### **Step 2: Make Predictions**  
Let’s predict the score for **6 hours of study**:

$$\
y = 135 - 25(6) = 135 - 150 = -15
\$$

This predicted value doesn’t make sense in this context (score can't be negative), but it shows the need for feature scaling or adjusting the model for realistic results.

---

## **Pros and Cons of Linear Regression**

✅ **Advantages:**
- Simple to understand and implement.  
- Works well for datasets where the relationship between the input and output is linear.  
- Fast to train and predict.

❌ **Disadvantages:**
- Assumes a linear relationship between inputs and outputs (which may not always be the case).  
- Sensitive to outliers.  
- Doesn’t perform well when the data is too complex or has non-linear relationships.

---

## **Final Thoughts**  
- **Linear Regression** is one of the easiest algorithms for regression tasks, and it serves as a foundation for understanding more complex models.  
- It is **sensitive to outliers**, so data cleaning and preprocessing are crucial.  
- **Multiple Linear Regression** is an extension that works with more than one input feature.  
