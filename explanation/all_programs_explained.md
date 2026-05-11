# ANN Lab Programs - Simple Explanations & Viva Questions

---

## Program 1: Plot Activation Functions
**File:** `1_plot_activationfunctions.py`

### What it does
This program draws graphs of 4 popular activation functions used in neural networks. Think of activation functions as the "decision makers" inside a neuron — they decide how strongly a neuron fires.

The 4 functions plotted are:
- **Sigmoid** — squashes any number into a value between 0 and 1 (like a probability)
- **Tanh** — similar to sigmoid but outputs between -1 and +1
- **ReLU (Rectified Linear Unit)** — outputs 0 for negative inputs, and the same number for positive inputs
- **Leaky ReLU** — like ReLU but allows a tiny negative output (0.01x) instead of 0

The program uses `numpy` to generate 100 values from -10 to 10 and `matplotlib` to draw a 2x2 grid of plots.

### Viva Questions
1. **What is an activation function?**
   It is a mathematical function applied to a neuron's output to introduce non-linearity, allowing the network to learn complex patterns.

2. **Why do we need activation functions?**
   Without them, a neural network would just be a linear function no matter how many layers it has, and it couldn't solve complex problems.

3. **What is the range of the sigmoid function?**
   (0, 1) — it never actually reaches 0 or 1.

4. **What is the vanishing gradient problem and which functions suffer from it?**
   When gradients become very small during backpropagation, learning slows down or stops. Sigmoid and Tanh suffer from this because their gradients approach 0 for large or small inputs.

5. **Why is ReLU preferred over Sigmoid in hidden layers?**
   ReLU does not saturate for positive values, so gradients remain large and learning is faster. It is also computationally simple.

6. **What is the dying ReLU problem?**
   If a neuron always gets negative input, ReLU always outputs 0 and the neuron stops learning. Leaky ReLU fixes this by allowing a small negative slope.

7. **What is the difference between ReLU and Leaky ReLU?**
   ReLU outputs 0 for all negative inputs. Leaky ReLU outputs a small value (0.01 * x) for negative inputs so neurons never fully "die."

8. **What does `np.linspace(-10, 10, 100)` do?**
   It generates 100 evenly spaced values between -10 and 10.

---

## Program 2: McCulloch-Pitts AND-NOT Function
**File:** `2_ANDNOT_function_McCulloch-Pittsneural net.py`

### What it does
This is the **oldest and simplest neural network model** (proposed in 1943). It simulates a single artificial neuron that computes the AND-NOT logic function: output is 1 only when A=1 and B=0.

**How it works:**
- Input A gets weight +1 (positive → supports firing)
- Input B gets weight -1 (negative → inhibits firing)
- Net input = (A × 1) + (B × -1)
- If net input >= threshold (1), output = 1, else output = 0

**Truth Table:**

| A | B | Output |
|---|---|--------|
| 0 | 0 | 0      |
| 0 | 1 | 0      |
| 1 | 0 | 1      |
| 1 | 1 | 0      |

### Viva Questions
1. **Who proposed the McCulloch-Pitts neuron model?**
   Warren McCulloch (neuroscientist) and Walter Pitts (logician) in 1943.

2. **What is the threshold in a McCulloch-Pitts neuron?**
   A fixed value; if the total weighted input meets or exceeds it, the neuron fires (outputs 1), otherwise 0.

3. **What are the limitations of the McCulloch-Pitts model?**
   - Weights and threshold are fixed (not learned)
   - Only handles binary inputs/outputs
   - Cannot implement all logic functions (e.g., XOR)
   - No learning capability

4. **What is the role of a negative weight in this model?**
   A negative weight acts as inhibition — it reduces the net input and can prevent the neuron from firing.

5. **What is the AND-NOT function?**
   It outputs 1 only when A is 1 AND B is 0. It is also called "A inhibited by B."

6. **Can McCulloch-Pitts model solve XOR? Why not?**
   No. XOR is not linearly separable — you cannot draw a single straight line to separate its outputs, and the M-P model can only represent linearly separable functions.

7. **What activation function does the M-P neuron use?**
   A step function (Heaviside function): outputs 1 if input >= threshold, else 0.

---

## Program 3: Perceptron Neural Network (Even/Odd Classifier)
**File:** `3_Perceptron_NeuralNetwork.py`

### What it does
This program trains a **single-layer perceptron** to classify digits 0–9 as even (0) or odd (1).

**How it works:**
1. Converts each digit's ASCII value into an 8-bit binary number (e.g., '0' → ASCII 48 → `00110000`)
2. Uses these 8 bits as input features
3. Trains a perceptron with the **Perceptron Learning Rule** for 20 epochs
4. At the end, tests whether it correctly predicts even/odd for each digit

**Perceptron update rule:**
- If prediction is wrong: `weight = weight + learning_rate × error × input`
- Error = actual output − predicted output

### Viva Questions
1. **What is a perceptron?**
   The simplest type of artificial neural network — a single neuron with adjustable weights that can classify linearly separable data.

2. **Who invented the perceptron?**
   Frank Rosenblatt in 1958.

3. **What is the perceptron learning rule?**
   If the output is correct, do nothing. If wrong, adjust weights: `w = w + lr * error * x`.

4. **What activation function is used here?**
   A step function: outputs 1 if net input >= 0, else 0.

5. **Why use ASCII binary representation as input?**
   It converts each digit into a fixed-length numerical vector (8 bits) that the perceptron can process.

6. **What does the learning rate control?**
   How much the weights change after each mistake. Too high = unstable learning, too low = very slow learning.

7. **What is an epoch?**
   One complete pass through the entire training dataset.

8. **What is the perceptron convergence theorem?**
   If the data is linearly separable, the perceptron learning algorithm is guaranteed to find a solution in a finite number of steps.

---

## Program 4: Perceptron Learning Law with Decision Boundary
**File:** `4_Perceptron_learing_law.py`

### What it does
This program demonstrates the **Perceptron Learning Law** visually. It trains a perceptron to separate two classes of 2D points and then **draws the decision boundary** on a graph.

- **Class 0 (Blue):** Points near (1,1), (2,1), (1.5,2) — bottom-left cluster
- **Class 1 (Red):** Points near (4,4), (5,3), (4.5,5) — top-right cluster

After training, it draws a straight line (decision boundary) that separates the two classes, showing how the perceptron has learned to distinguish them.

### Viva Questions
1. **What is a decision boundary?**
   The line (or surface in higher dimensions) that separates different classes. Points on one side belong to Class 0, points on the other to Class 1.

2. **How is the decision boundary calculated for plotting?**
   From the equation `w[0]*x + w[1]*y + b = 0`, we solve for y: `y = -(w[0]*x + b) / w[1]`.

3. **What does linearly separable mean?**
   Two classes are linearly separable if a single straight line can separate them perfectly.

4. **Why does the perceptron fail on non-linearly separable data?**
   It can only learn a linear boundary, so it can never correctly classify data that requires a curved or complex boundary.

5. **What is the role of the bias term?**
   It shifts the decision boundary away from the origin, giving the model more flexibility.

6. **What happens if the learning rate is too large?**
   The weights oscillate wildly and the model may never converge.

7. **What is `np.vstack`?**
   It stacks arrays vertically (row-wise), combining Class 0 and Class 1 arrays into one dataset.

---

## Program 5: Bidirectional Associative Memory (BAM)
**File:** `5_bidirectional__associative_memory_twopairs_vectors.py`

### What it does
**BAM** is a type of associative memory — it can recall a stored pair given either half of the pair. Think of it like a two-way lookup: if you remember A, it gives you B; and if you remember B, it gives you A.

**What's stored:**
- Pair 1: X1 = [1,0,1] ↔ Y1 = [0,1]
- Pair 2: X2 = [0,1,0] ↔ Y2 = [1,0]

**Steps:**
1. Convert binary patterns to **bipolar** (0 → -1, 1 → +1)
2. Build weight matrix using **outer product**: W = X1⊗Y1 + X2⊗Y2
3. To recall: pass X through W to get Y, pass Y through W^T to get back X — repeat until stable

### Viva Questions
1. **What is associative memory?**
   A type of memory that retrieves stored patterns based on partial or noisy input — unlike RAM which needs an exact address.

2. **Who proposed BAM?**
   Bart Kosko in 1988.

3. **What is the difference between BAM and Hopfield Network?**
   Hopfield is single-directional (one set of patterns), while BAM is bidirectional (pairs of patterns from two different sets).

4. **Why do we use bipolar representation instead of binary?**
   Bipolar (+1/-1) works better mathematically with the Hebbian learning rule and gives a stronger signal difference.

5. **What is the outer product and why is it used for training?**
   The outer product of two vectors creates a matrix representing their correlation. Summing these for all pairs encodes all associations into the weight matrix.

6. **What is the sign activation function?**
   Outputs +1 for non-negative input and -1 for negative input. Used to make binary decisions in bipolar space.

7. **What does "recall" mean in BAM?**
   Given one pattern, the network iteratively computes forward and backward passes until it settles on a stored pair.

---

## Program 6: Digit Recognition Using 5x3 Matrix
**File:** `6__recognize_5_3_matrix.py`

### What it does
This program recognizes hand-crafted digits (0, 1, 2, 3, 9) represented as **5×3 pixel grids** (like tiny dot-matrix displays).

Each digit is a 5-row × 3-column matrix of 1s and 0s. The program:
1. Flattens each 5×3 matrix into a 15-element vector
2. Trains a **multi-class perceptron** (one weight vector per class)
3. Predicts a digit by computing a score for each class and picking the highest

**Training rule (Perceptron for multi-class):**
- If predicted class is wrong: increase weights for correct class, decrease for predicted class

### Viva Questions
1. **What is feature extraction in this context?**
   Converting a 5×3 grid into a 15-element flat vector so it can be used as numeric input to the classifier.

2. **How does multi-class classification differ from binary classification?**
   Instead of one output (0 or 1), we have one score per class and pick the class with the highest score.

3. **What is `np.argmax`?**
   Returns the index of the highest value in an array — used here to pick the predicted class.

4. **What is the Perceptron multi-class update rule?**
   If wrong: increase weights of the correct class by `lr * x`, and decrease weights of the wrongly predicted class by `lr * x`.

5. **What does `np.random.randn` do compared to `np.random.rand`?**
   `randn` gives values from a normal distribution (mean 0, std 1), while `rand` gives uniform values between 0 and 1.

6. **Why are biases initialized to zeros here?**
   It is a common starting point that works well with the perceptron update rule.

7. **Can this model handle noisy or partially corrupted digit inputs?**
   Not well — a basic perceptron has no noise tolerance. A Hopfield or BAM network handles noise better.

---

## Program 7: Backpropagation Network for XOR (Script Style)
**File:** `7__backpropagation_network_xor.py`

### What it does
This program solves the **XOR problem** using a neural network trained with **backpropagation** — written as a plain script (no class).

**Network structure:**
- Input layer: 2 neurons
- Hidden layer: 2 neurons (sigmoid)
- Output layer: 1 neuron (sigmoid)

**Training loop for each epoch:**
1. Forward pass: compute hidden and output values
2. Calculate error: `error = y - output`
3. Backpropagate: compute how much each weight contributed to the error
4. Update weights using gradient descent

After 10,000 epochs, it prints the raw output values (not rounded), showing how close the network got to the expected outputs.

### Viva Questions
1. **Why can't a single-layer perceptron solve XOR?**
   XOR is not linearly separable — no single straight line can separate its outputs. A hidden layer is needed to create a non-linear decision boundary.

2. **What is backpropagation?**
   An algorithm that computes the gradient of the loss function with respect to each weight by applying the chain rule backwards from the output to the input.

3. **What is the chain rule and why is it important in backpropagation?**
   The chain rule from calculus lets us compute derivatives of composed functions. In backprop, it allows us to find how each weight affects the final error.

4. **What is gradient descent?**
   An optimization method that updates weights in the direction that reduces the error: `w = w + lr * gradient`.

5. **What is `np.random.seed(42)` doing?**
   Setting a fixed starting point for random number generation so results are reproducible every run.

6. **What does the sigmoid derivative formula `x*(1-x)` assume?**
   That `x` is already the sigmoid output (not the raw input), so `sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))`.

7. **What is the role of the learning rate in backpropagation?**
   It controls how big each weight update step is. Too large causes overshooting; too small makes learning very slow.

8. **What is epoch vs iteration?**
   An epoch is one full pass through all training data. An iteration is one weight update (could be per sample or per batch).

---

## Program 8: Backpropagation Feedforward Neural Network (Class Style)
**File:** `8__back_propagation_feed_forward_neural__network_.py`

### What it does
This is the **same XOR problem as Program 7** but written as a clean **object-oriented class** (`NeuralNetwork`). This is a better-structured version.

**Key methods:**
- `__init__`: initializes random weights and biases
- `forward(X)`: computes predictions (input → hidden → output)
- `backward(X, y, output)`: computes gradients and updates weights
- `train(X, y, epochs)`: loops through forward + backward
- `predict(X)`: runs forward and rounds output to 0 or 1

**Output:**
```
Input: [0 0] → Output: 0
Input: [0 1] → Output: 1
Input: [1 0] → Output: 1
Input: [1 1] → Output: 0
```
All 4 XOR predictions are correct.

### Viva Questions
1. **What is the difference between Program 7 and Program 8?**
   Both solve XOR with backpropagation, but Program 8 uses a class-based design making it reusable and modular.

2. **What does `np.dot(X, self.W1)` compute?**
   Matrix multiplication of input X with weight matrix W1, computing the weighted sum of inputs for all neurons in the hidden layer at once.

3. **Why do we use `keepdims=True` in `np.sum`?**
   To preserve the shape of the result (keeps it as a 2D array), which is needed for correct broadcasting when updating biases.

4. **What does `np.round(output)` do in the predict method?**
   Rounds values >= 0.5 to 1 and < 0.5 to 0, converting the continuous sigmoid output into a binary prediction.

5. **What is the vanishing gradient problem in this network?**
   As gradients are multiplied repeatedly through layers, they can become very small, causing early layers to learn very slowly.

6. **What would happen if we used linear activation instead of sigmoid?**
   The network would collapse to a linear function regardless of depth and could not solve XOR.

7. **What is the weight initialization strategy here and is it good?**
   `np.random.rand` initializes with uniform values in [0,1). It works here but better practice is to use small values centered around 0 (e.g., Xavier initialization).

---

## Program 9: Hopfield Network
**File:** `9__hopfield_network.py`

### What it does
A **Hopfield Network** is a type of recurrent neural network used as **associative memory** — it can recover stored patterns even from noisy or incomplete inputs.

**What's stored:** 4 bipolar patterns of length 4:
```
[1, -1,  1, -1]
[-1,  1, -1,  1]
[1,  1, -1, -1]
[-1, -1,  1,  1]
```

**Training (Hebbian Rule):**
- For each pattern p: add p × p^T to weight matrix W
- Set diagonal to 0 (no self-connections)

**Recall:**
- Given a noisy input, repeatedly apply: `x = sign(W × x)`
- The network settles into the closest stored pattern

**Demo:** Input `[1, -1, -1, -1]` (corrupted version of pattern 1) → recovers `[1, -1, 1, -1]`

### Viva Questions
1. **Who proposed the Hopfield Network?**
   John Hopfield in 1982.

2. **What is the Hebbian learning rule?**
   "Neurons that fire together, wire together." Weights are set based on the correlation of neuron activations: `W += p * p^T`.

3. **What are attractors in a Hopfield Network?**
   Stable states that the network converges to. The stored patterns are attractors.

4. **What is the capacity of a Hopfield Network?**
   It can reliably store approximately 0.138 × N patterns, where N is the number of neurons.

5. **Why are diagonal elements set to 0?**
   To prevent a neuron from exciting itself (no self-connections), which would create artificial stability.

6. **What is a spurious state?**
   An unintended stable state that is not one of the stored patterns — can occur when too many patterns are stored.

7. **What is the energy function of a Hopfield Network?**
   E = -0.5 × x^T × W × x. The network always moves toward lower energy states, which correspond to stored patterns.

8. **How does the Hopfield Network differ from BAM?**
   Hopfield stores and recalls patterns from a single set. BAM stores and recalls pairs of patterns between two different sets.

---

## Program 10: Object Detection using YOLOv3 (CNN)
**File:** `10/cnn2.py`

### What it does
This program uses **YOLOv3** (You Only Look Once), a real-time object detection deep learning model, to detect and label objects in an image.

**Steps:**
1. Load **COCO class names** (80 object categories like person, car, dog)
2. Load the **YOLOv3 model** (pre-trained weights + configuration)
3. Read an image and convert it to a **blob** (normalized, resized format for the network)
4. Run a **forward pass** through the network to get detections
5. Filter detections with **confidence > 0.5**
6. Apply **Non-Maximum Suppression (NMS)** to remove duplicate boxes
7. Draw bounding boxes and labels on the image and display it

**Required files:** `yolov3.cfg`, `yolov3.weights`, `coco.names`, `image.png`

### Viva Questions
1. **What is YOLO and what does it stand for?**
   You Only Look Once — a real-time object detection algorithm that processes the entire image in a single forward pass.

2. **What is a CNN (Convolutional Neural Network)?**
   A deep learning architecture that uses convolutional layers to automatically learn spatial features from images, making it ideal for image recognition and detection tasks.

3. **What is a blob in OpenCV DNN?**
   A preprocessed image in a specific format: normalized pixel values, resized to network input size (416×416 for YOLO), and rearranged into NCHW format.

4. **What is Non-Maximum Suppression (NMS)?**
   A technique to remove redundant overlapping bounding boxes by keeping only the one with the highest confidence when multiple boxes detect the same object.

5. **What is confidence score in YOLO?**
   A combination of objectness (probability that an object is present) and class probability. Boxes with score below the threshold (0.5) are discarded.

6. **What is the COCO dataset?**
   Common Objects in Context — a large-scale dataset with 80 object categories used to train YOLO and many other detection models.

7. **What are the output layers in YOLO?**
   The unconnected output layers at 3 different scales, allowing YOLO to detect objects of different sizes.

8. **What is the difference between object detection and image classification?**
   Classification tells you what is in the image. Detection tells you what is in the image AND where it is (bounding box coordinates).

9. **What does `cv2.dnn.blobFromImage` do?**
   Converts an image into a 4D blob (batch, channels, height, width) with optional scaling, resizing, and mean subtraction.

10. **Why does YOLO process the image only once (unlike sliding window approaches)?**
    YOLO divides the image into a grid and predicts boxes and classes for each grid cell simultaneously in one forward pass, making it much faster than sliding window methods.

---

## Quick Summary Table

| # | Program | Topic | Key Concept |
|---|---------|-------|-------------|
| 1 | Plot Activation Functions | Neural Network Basics | Sigmoid, Tanh, ReLU, Leaky ReLU |
| 2 | McCulloch-Pitts AND-NOT | Earliest Neuron Model | Weighted sum + threshold |
| 3 | Perceptron Even/Odd | Perceptron Classifier | Perceptron learning rule |
| 4 | Perceptron Decision Boundary | Visualization | Linear separability |
| 5 | BAM | Associative Memory | Bidirectional recall |
| 6 | 5x3 Digit Recognition | Multi-class Perception | Pattern matching |
| 7 | XOR Backpropagation (script) | Backpropagation | Chain rule, gradient descent |
| 8 | XOR Backpropagation (class) | Feedforward NN | OOP + backpropagation |
| 9 | Hopfield Network | Recurrent Memory | Hebbian learning, attractors |
| 10 | YOLOv3 Object Detection | CNN / Deep Learning | Real-time detection, NMS |
