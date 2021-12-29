import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.get_weights(),x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        scalar = nn.as_scalar(self.run(x))
        return 1.0 if scalar >= 0 else -1.0


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            accuracy = 1
            for vector, truth in dataset.iterate_once(1):
                prediction = self.get_prediction(vector)
                if prediction != nn.as_scalar(truth):
                    self.get_weights().update(vector, nn.as_scalar(truth))
                    accuracy = 0
            if accuracy == 1:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.m0 = nn.Parameter(1,50)
        self.b0 = nn.Parameter(1,50)
        self.m1 = nn.Parameter(50,50)
        self.b1 = nn.Parameter(1,50)
        self.m2 = nn.Parameter(50,50)
        self.b2 = nn.Parameter(1,50)
        self.m3 = nn.Parameter(50,1)
        self.b3 = nn.Parameter(1,1)
        self.loss = nn.SquareLoss

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        xm = nn.Linear(x,self.m0)
        predicted_y = nn.AddBias(xm,self.b0)
        relU_output = nn.ReLU(predicted_y)

        xm1 = nn.Linear(relU_output, self.m1)
        predicted_y1 = nn.AddBias(xm1,self.b1)
        relU_output1 = nn.ReLU(predicted_y1)

        xm2 = nn.Linear(relU_output1, self.m2)
        predicted_y2 = nn.AddBias(xm2, self.b2)
        relU_output2 = nn.ReLU(predicted_y2)

        xm3 = nn.Linear(relU_output2, self.m3)
        return nn.AddBias(xm3, self.b3)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return self.loss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for vector, truth in dataset.iterate_once(1):
                loss = self.get_loss(vector, truth)
                parameters = [self.m0,self.b0,self.m1,self.b1,self.m2,self.b2,self.m3,self.b3]
                gradients = nn.gradients(loss,parameters)
                i=0
                for weights in parameters:
                    weights.update(gradients[i],-0.005)
                    i += 1
            new_loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x),nn.Constant(dataset.y)))
            if new_loss <= 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    """

    def __init__(self):
        # Initialize your model parameters here
        self.m0 = nn.Parameter(784,50)
        self.b0 = nn.Parameter(1,50)
        self.m1 = nn.Parameter(50,50)
        self.b1 = nn.Parameter(1,50)
        self.m2 = nn.Parameter(50,50)
        self.b2 = nn.Parameter(1,50)
        self.m3 = nn.Parameter(50,10)
        self.b3 = nn.Parameter(1,10)
        self.loss = nn.SoftmaxLoss

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        xm = nn.Linear(x,self.m0)
        predicted_y = nn.AddBias(xm,self.b0)
        relU_output = nn.ReLU(predicted_y)

        xm1 = nn.Linear(relU_output, self.m1)
        predicted_y1 = nn.AddBias(xm1,self.b1)
        relU_output1 = nn.ReLU(predicted_y1)

        xm2 = nn.Linear(relU_output1, self.m2)
        predicted_y2 = nn.AddBias(xm2, self.b2)
        relU_output2 = nn.ReLU(predicted_y2)

        xm3 = nn.Linear(relU_output2, self.m3)
        return nn.AddBias(xm3, self.b3)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return self.loss(self.run(x),y)


    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for vector, truth in dataset.iterate_once(10):
                loss = self.get_loss(vector, truth)
                parameters = [self.m0,self.b0,self.m1,self.b1,self.m2,self.b2,self.m3,self.b3]
                gradients = nn.gradients(loss,parameters)
                i=0
                for weights in parameters:
                    weights.update(gradients[i],-0.025)
                    i += 1
            validation_score = dataset.get_validation_accuracy()
            if validation_score >= .975:
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        self.m0 = nn.Parameter(self.num_chars,5)
        self.b0 = nn.Parameter(1,5)
        self.m1 = nn.Parameter(self.num_chars,5)
        self.b1 = nn.Parameter(1,5)
        self.m2 = nn.Parameter(5,5)
        self.b2 = nn.Parameter(1,5)
        self.m3 = nn.Parameter(5,5)
        self.b3 = nn.Parameter(1,len(self.languages))


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        f_initial = nn.Linear(xs[0],self.m0)
        for vector in xs:
            f_initial = nn.Add(nn.Linear(vector, self.m1), f_initial)
            x_bias = nn.AddBias(f_initial, self.b0)
            relU = nn.ReLU(x_bias)
            f_initial = nn.Add(nn.Linear(relU, self.m2), x_bias)
            x_bias = nn.AddBias(f_initial, self.b1)
            relU = nn.ReLU(x_bias)
        f_initial = nn.Add(nn.Linear(relU, self.m3), x_bias)
        return nn.AddBias(f_initial, self.b2)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return self.loss(self.run(xs),y)


    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for vector, truth in dataset.iterate_once(20):
                loss = self.get_loss(vector, truth)
                parameters = [self.m0,self.b0,self.m1,self.b1,self.m2,self.b2,self.m3,self.b3]
                gradients = nn.gradients(loss,parameters)
                i=0
                for weights in parameters:
                    weights.update(gradients[i],-0.015)
                    i += 1
            validation_score = dataset.get_validation_accuracy()
            if validation_score > .71:
                break
