import numpy as np
from tqdm import tqdm
from lib import *

class LinearModel(Model):
    """
    A simple linear model with weights (bias is implicit).

    Attributes:
        _w (np.ndarray): The weights of the model, initialized with a normal distribution.
        _grad (np.ndarray): Stores the gradient w.r.t. the weights after the forward pass.
    """

    def __init__(self, in_features: int):
        """
        Initializes the LinearModel with random weights.

        Args:
            in_features (int): The number of input features (dimension of the input vectors).
        """

        self._w = np.random.normal(0, 0.01, size=(in_features,))
        self._grad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the linear model.

        Args:
            x (np.ndarray): The input data vectors of shape (n_samples, n_features).

        Returns:
            np.ndarray: The scalar results of the forward pass

        Forbidden:
            numpy.gradient, numpy.diff
        """

        # TODO: Implement the forward pass 
        y = x @ self._w
        # TODO: Compute the gradient and store it in self._grad
        self._grad = x

        return y    
    def backward(self) -> np.ndarray:
        """
        Performs the backward pass to retrieve the gradient for the last input.
        
        Args:
            None

        Returns:
            np.ndarray: The gradients.

        Forbidden:
            numpy.gradient, numpy.diff
        """

       
        # TODO: Implement the backward pass that returns the gradient 
        return self._grad
            
class L2Loss(Loss):
    """
    Computes the L2Loss between the target values and the predictions.
    
    Attributes:
        _grad (np.ndarray): The gradient of the loss with respect to the inputs.
    """

    def forward(self, t: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Computes the L2 loss value and caches the gradient.

        Args:
            t (np.ndarray): Target values.
            y (np.ndarray): Predicted values.

        Returns:
            np.float64: The computed loss.

        Forbidden:
            numpy.gradient, numpy.diff
        """

        # TODO: Implement L2Loss forward
         # Anzahl der Samples
        n = t.shape[0]

        # Differenz speichern (praktisch für backward)
        diff = y - t

        # Mean Squared Error
        loss = np.sum(diff ** 2) / n

        # Gradient nach y
        self._grad = (2.0 / n) * diff

        return loss   
    def backward(self) -> np.ndarray:
        """
        Returns the gradient of the loss with respect to the last inputs.

        Returns:
            np.ndarray: The computed gradient.

        Forbidden:
            numpy.gradient, numpy.diff
        """

        # TODO: Implement L2Loss backward
        return self._grad
    
class PerceptronLoss(Loss):
    """
    Computes the Perceptron loss between the target values and the predictions.
    
    Attributes:
        _grad (np.ndarray): The gradient of the loss with respect to the inputs.
    """

    def forward(self, t: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Computes the Perceptron Loss value and caches the gradient.

        Args:
            t (np.ndarray): Target values.
            y (np.ndarray): Predicted values.

        Returns:
            np.float64: The computed loss.

        Forbidden:
            numpy.gradient, numpy.diff
        """

        # TODO: Implement PerceptronLoss forward
        n = t.shape[0]
        # Für jedes Sample wird max(0, -t_i * y_i) berechnet.
        loss = np.mean(np.maximum(0.0, -t * y))
        # Wenn t_i * y_i < 0 → dL/dy_i = -t_i / n
        # Sonst → 0
        # (t * y < 0) erzeugt ein Boolean-Array, das zu 0/1 konvertiert wird
        self._grad = (-t / n) * (t * y < 0)

        return loss  
    def backward(self) -> np.ndarray:
        """
        Returns the gradient of the loss with respect to the last input.

        Returns:
            np.ndarray: The computed gradient.

        Forbidden:
            numpy.gradient, numpy.diff
        """

        # TODO: Implement PerceptronLoss backward
        return self._grad
class HingeLoss(Loss):
    """
    Computes the Hinge loss between the target values and the predictions.
    
    Attributes:
        _grad (np.ndarray): The gradient of the loss with respect to the input
    """

    def forward(self, t: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Computes the Hinge loss value and caches the gradient.

        Args:
            t (np.ndarray): Target values.
            y (np.ndarray): Predicted values.

        Returns:
            np.float64: The computed loss.

        Forbidden:
            numpy.gradient, numpy.diff
        """

        # TODO: Implement HingeLoss forward
        # Anzahl der Samples im Batch
        n = t.shape[0]

        # Margin: 1 - t*y
        margins = 1.0 - t * y

        # Loss: Mittelwert der positiven Margins
        loss = np.mean(np.maximum(0.0, margins))

        # Gradient dL/dy:
        # Falls margin > 0 (also t*y < 1) -> d/dy (1 - t*y) = -t
        # Sonst 0. Wegen mean zusätzlich /n.
        self._grad = np.where(margins > 0.0, (-t) / n, 0.0)

        return loss    
    def backward(self) -> np.ndarray:
        """
        Returns the gradient of the loss with respect to the last input.

        Returns:
            np.ndarray: The computed gradient.

        Forbidden:
            numpy.gradient, numpy.diff
        """

        # TODO: Implement HingeLoss backward
        return self._grad
    
def training_loop(
        model: Model, 
        X_train: np.ndarray, 
        t_train: np.ndarray,
        loss_fn: Loss,
        num_epochs: int = 30, 
        batch_size: int = 100, 
        step_size: float = 0.01,
        progress_bar: tqdm = None,
    ) -> Model:
    """
    Performs gradient descent on the model.

    Args:
        model (Model): The model on which gradient descent should be performed.
        X_train (np.ndarray): The training features.
        t_train (np.ndarray): The training labels.
        loss_fn (Loss): The loss function for the loss calculation.
        num_epochs (int): Number of epochs for which the model is trained.
        batch_size (int): Size of the batches.
        step_size (float): The step size of the gradient descent step.
        progress_bar (tqdm): Used to show the current training progress with a progress bar.

    Returns:
        Model: The trained model.
    """

    # n: number of data points
    # d: dimensionality of the data points
    n, d = X_train.shape

    for e in range(num_epochs):
        p = np.random.permutation(n)
        X_p = X_train[p]
        t_p = t_train[p]

        # update Progress Bar
        if  progress_bar is not None:
             progress_bar.update(1) 

        for batch in range(int(np.ceil(n/batch_size))):
            batch_start = batch * batch_size
            batch_end = min((batch+1) * batch_size, n)
            X_b = X_p[batch_start:batch_end]
            t_b = t_p[batch_start:batch_end]
            # TODO: Calculate the gradient of the model and update its parameters with gradient descent
                        # Forward: Modell-Ausgaben für den Batch berechnen
            y_b = model.forward(X_b)

            # Loss berechnen speichert intern dL/dy für backward
            _ = loss_fn.forward(t_b, y_b)

            # Backward vom Loss: Gradient dL/dy
            dL_dy = loss_fn.backward()

            # Gradient nach den Gewichten: dL/dw = X^T * dL/dy
            grad_w = X_b.T @ dL_dy

            #w := w - step_size * grad_w
            model._w = model._w - step_size * grad_w

    return model


def prepare_labels(t: np.ndarray, target_digit: int) -> np.ndarray:
    """
    Prepares labels for a one-vs-all training setup.

    All entries in t that correspond to the target digit are encoded as 1,
    while all other entries are encoded as -1.

    Parameters
    ----------
    t : np.ndarray
        Array containing digit labels (0-9).
    target_digit : int
        The digit treated as the positive class.

    Returns
    -------
    np.ndarray
        Array containing labels in {-1, 1}.
    """
    assert target_digit in range(10)

    # Erzeuge neue Labels für One-vs-All:
    # target_digit -> 1
    # alle anderen -> -1
    labels = np.where(t == target_digit, 1.0, -1.0).astype(np.float64)

    return labels


def train_models(
        loss_fn: Loss,
        X_train: np.ndarray,
        t_train: np.ndarray,
        num_epochs: int,
        digits_count: int,
        batch_size: int,
        step_size: float,
        progress_bar: tqdm = None,
    ) -> list:
    """
    Trains one linear model per digit using a one-vs-all strategy.

    For each digit from 0 to 9, a separate LinearModel is created and trained
    to distinguish the current digit (label 1) from all other digits (label -1),
    using the given loss function.

    Parameters
    ----------
    loss_fn : Loss
        The loss function used for training all models.

    Returns
    -------
    list[LinearModel]
        A list of 10 trained linear models, ordered by digit
        (model 0 corresponds to digit 0, model 1 to digit 1, etc.).
    """

    models = []
    # TODO: Train one linear model per digit with given loss function. Use the given variables as parameters when calling the training_loop() function.
    # Prepare the labels on the t_train before caling the training_loop().
   



def predict(models: list, X_test: np.ndarray) -> np.ndarray: 
    """
    Performs class prediction for the given test data.

    For each linear model in the list `models`, the model outputs are computed
    for all test samples. 
    (one-vs-all classification).

    Parameters
    ----------
    models : list[LinearModel]
        List of trained linear models, one per class.
    X_test : np.ndarray
        Test data of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Array of length `n_samples` containing the predicted class labels
        (index of the model with the maximum output) for each test sample.
    """
    # TODO: Use the LinearModel's forward() method to compute scores for the test samples.
    # The score represents how confident the model is that the input image depicts the digit it was trained to recognize.
    # Return a NumPy array of predicted digits for each Image in X_test. 
        # Array für die vorhergesagten Ziffern (eine pro Testbild)
    predictions = np.zeros(X_test.shape[0], dtype=int)

    # Gehe jedes Testbild einzeln durch
    for j in range(X_test.shape[0]):
        # Für das aktuelle Bild berechnen wir die Scores aller Modelle
        # (one-vs-all: jedes Modell steht für eine Ziffer)
        scores = [model.forward(X_test[j:j+1])[0] for model in models]

        # Wähle die Ziffer mit dem höchsten Score
        predictions[j] = np.argmax(scores)

    return predictions


if __name__ == "__main__":    
    num_epochs, digits_count, batch_size, step_size = 30, 10, 50, 0.001

    # You can change the data set to the original MNIST (will install from internet)
    X_train, t_train, X_test, t_test = load_dataset('cg-digits')
    modelsList = []
    l2loss = L2Loss()
    ploss = PerceptronLoss()
    hloss = HingeLoss()

    # Train on L2Loss
    with tqdm(total=num_epochs*digits_count, desc=l2loss.__class__.__name__ + "-Training") as progress_bar:
        models = train_models(l2loss, X_train, t_train, num_epochs, digits_count, batch_size, step_size, progress_bar)
    modelsList.append((models, l2loss.__class__.__name__))
    p = predict(models, X_test)
    print(f"L2Loss Accuracy: {100 * (p == t_test).mean()}%")

    # Train on PerceptronLoss
    with tqdm(total=num_epochs*digits_count, desc=ploss.__class__.__name__ + "-Training") as progress_bar:
        models = train_models(ploss, X_train, t_train, num_epochs, digits_count, batch_size, step_size, progress_bar)
    modelsList.append((models, ploss.__class__.__name__))
    p = predict(models, X_test)
    print(f"Perceptron Loss Accuracy: {100 * (p == t_test).mean()}%")

    # Train on HingeLoss
    with tqdm(total=num_epochs * digits_count, desc=hloss.__class__.__name__ + "-Training") as progress_bar:
        models = train_models(hloss,X_train, t_train, num_epochs, digits_count, batch_size, step_size, progress_bar)
    modelsList.append((models, hloss.__class__.__name__))
    p = predict(models, X_test)
    print(f"Hinge Loss Accuracy: {100 * (p == t_test).mean()}%")


    # Starts the Gui for interactive digit prediction
    root = tk.Tk()
    gui = DrawGUI(root)
    def on_digit_was_drawn(arr):
        x = arr.reshape(1, -1)         # shape (1, 784)
        x = np.hstack([x, np.ones((1, 1))])  # add bias term → shape (1, 785)
        predictList = []
        for models, name in modelsList:
            pred = predict(models, x)
            predictList.append(pred)
            print("Predicted digit for "+name+": ", pred[0])
            pred_text = "\n".join([f"{name} predicted {pred[0]}" for (pred, (models, name)) in zip(predictList, modelsList)])
        gui.pred_label.config(text=pred_text)

    gui.callback = on_digit_was_drawn 
    root.mainloop()
