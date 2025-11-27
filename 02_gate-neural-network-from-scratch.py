import numpy as np


AND_DATA = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

OR_DATA = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

XOR_DATA = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])






# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)





# Initialize weights randomly
def init_weights(seed=42):
    rng = np.random.default_rng(seed)
    
    # Hidden layer weights
    w11, w12 = rng.uniform(-0.5, 0.5, 2)
    w21, w22 = rng.uniform(-0.5, 0.5, 2)
    b1 = rng.uniform(-0.5, 0.5)
    b2 = rng.uniform(-0.5, 0.5)
    


    # Output layer weights
    v1, v2 = rng.uniform(-0.5, 0.5, 2)
    b3 = rng.uniform(-0.5, 0.5)
    



    return w11, w12, w21, w22, b1, b2, v1, v2, b3






# Forward propagation
def forward(x, w11, w12, w21, w22, b1, b2, v1, v2, b3):

    x1, x2 = x
    
    # Hidden layer: z = w*x + b
    z1 = w11 * x1 + w12 * x2 + b1
    a1 = sigmoid(z1)
    
    z2 = w21 * x1 + w22 * x2 + b2
    a2 = sigmoid(z2)
    
    # Output layer
    z3 = v1 * a1 + v2 * a2 + b3
    y_pred = sigmoid(z3)
    
    return a1, a2, y_pred, z1, z2, z3







# Backward propagation
def backward(x, y, a1, a2, y_pred, z1, z2, z3, v1, v2, lr):

    x1, x2 = x
    
    # Output layer gradient: dL/dz3 = y_pred - y
    dz3 = y_pred - y
    
    # Output layer weight gradients
    dv1 = dz3 * a1
    dv2 = dz3 * a2
    db3 = dz3
    
    da1 = dz3 * v1
    da2 = dz3 * v2
    

    dz1 = da1 * sigmoid_derivative(a1)
    dz2 = da2 * sigmoid_derivative(a2)
    

    dw11 = dz1 * x1
    dw12 = dz1 * x2
    db1 = dz1
    
    dw21 = dz2 * x1
    dw22 = dz2 * x2
    db2 = dz2
    

    return dw11, dw12, dw21, dw22, db1, db2, dv1, dv2, db3









def train(dataset, epochs=10000, lr=0.1, seed=42):

    # Initialize weights
    w11, w12, w21, w22, b1, b2, v1, v2, b3 = init_weights(seed)
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for row in dataset:
            x = row[:2]
            y = row[2]
            


            # Forward pass
            a1, a2, y_pred, z1, z2, z3 = forward(x, w11, w12, w21, w22, b1, b2, v1, v2, b3)
            


            # Backward pass - get gradients
            dw11, dw12, dw21, dw22, db1, db2, dv1, dv2, db3 = backward(x, y, a1, a2, y_pred, z1, z2, z3, v1, v2, lr)
            


            # Update weights: w = w - lr * gradient
            w11 -= lr * dw11
            w12 -= lr * dw12
            w21 -= lr * dw21
            w22 -= lr * dw22
            b1 -= lr * db1
            b2 -= lr * db2
            v1 -= lr * dv1
            v2 -= lr * dv2
            b3 -= lr * db3
            


            # Calculate loss
            epsilon = 1e-8
            loss = -(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
            epoch_loss += loss



        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss = {epoch_loss:.4f}")
    


    # Return trained weights
    return w11, w12, w21, w22, b1, b2, v1, v2, b3






# Prediction function
def predict(x, w11, w12, w21, w22, b1, b2, v1, v2, b3):

    _, _, y_pred, _, _, _ = forward(x, w11, w12, w21, w22, b1, b2, v1, v2, b3)

    return 1 if y_pred >= 0.5 else 0


# Test function
def test_model(dataset, name, w11, w12, w21, w22, b1, b2, v1, v2, b3):

    correct = 0
    total = len(dataset)
    
    print(f"\n{name} Results:")


    for row in dataset:
        x = row[:2]
        y_pred = predict(x, w11, w12, w21, w22, b1, b2, v1, v2, b3)
        expected = row[2]


        status = "GOOD pred" if y_pred == expected else "BAD pred"
        print(f"  Input {x} => Prediction: {y_pred}, Expected: {expected} {status}")

        if y_pred == expected:
            correct += 1
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {correct}/{total} = {accuracy}")
    return correct == total










# Main execution
EPOCHS = 10000
LEARNING_RATE = 0.1
SEED = 42



# Train AND
print("___ Training AND Gate ___")
w11, w12, w21, w22, b1, b2, v1, v2, b3 = train(AND_DATA, EPOCHS, LEARNING_RATE, SEED)
test_model(AND_DATA, "AND", w11, w12, w21, w22, b1, b2, v1, v2, b3)



# Train OR
print("\n___ Training OR Gate ___")
w11, w12, w21, w22, b1, b2, v1, v2, b3 = train(OR_DATA, EPOCHS, LEARNING_RATE, SEED)
test_model(OR_DATA, "OR", w11, w12, w21, w22, b1, b2, v1, v2, b3)




# Train XOR
print("\n___ Training XOR Gate ___")
w11, w12, w21, w22, b1, b2, v1, v2, b3 = train(XOR_DATA, EPOCHS, LEARNING_RATE, SEED)
test_model(XOR_DATA, "XOR", w11, w12, w21, w22, b1, b2, v1, v2, b3)
