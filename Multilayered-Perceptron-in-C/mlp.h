#include <stdio.h>
#include <stdlib.h>

// Define constants for network configuration
#define EPOCHS        1000     // Number of epochs for training
#define d             2        // Number of inputs (x, y)
#define p             4        // Number of outputs (4 categoreis = 4 outputs)
#define H1            30       // Number of neurons in the first layer
#define H2            50       // Number of neurons in the second layer
#define H3            50       // Number of neurons in the third layer
#define HL            3        // Number of hidden layers
#define H             4        // Total number of layers (including output layer)
#define n             0.0001     // Learning rate
#define B             40       // Batch size for gradient descent (1=Stochastic, 4000=Batch, 400=Mini-Batch)
#define N             4000     // Number of training samples per set
#define WEIGHTS_NUM   H1*(d+1) + H2*(H1+1) + H3*(H2+1) + p*(H3+1)  // Total number of weights in the network
#define RANDOM_DOUBLE(A, B) ((double)rand() / (double)(RAND_MAX)) * (B - A) + A  // Generate a random double in [A, B]

// Structure representing an input with two features and a category
typedef struct Input {
    double x1;            // First input feature
    double x2;            // Second input feature
    int category[4];      // One-hot encoding of category (1 out of p encoding)
} Input_t;

// Structure representing a neuron in the network
typedef struct Neuron_t {
    double *w;            // Weight array
    double *error_derivative; // Array of error derivatives for each weight
    double output;        // Output value of the neuron
    double error_signal;  // Error signal, used by output layer neurons
    double delta_i;       // Delta value for weight update
} Neuron_t;

// Structure representing a layer in the network
typedef struct layer {
    Neuron_t *neuron;     // Array of neurons in the layer
} Layer_t;

// Structure representing the entire network
typedef struct Network {
    Layer_t layers[H];    // Array of layers in the network
} Network_t;

// Enumeration of possible activation functions
typedef enum {
    LOGISTIC,    // Logistic (sigmoid) function
    TANH,        // Hyperbolic tangent function
    RELU         // Rectified Linear Unit (ReLU)
} ActivationFunction;

// Function declarations
void init();                       // Initialize the network
void encode_input(double x1, double x2, Input_t *input, int i, char *type); // Encode input features
void categorize(char *category, Input_t *input, int i, char *type); // Categorize input data
void reverse_pass();               // Perform the reverse pass of backpropagation
void train_using_gradient_descent();           // Perform gradient descent weight update
void backprop(Input_t x);          // Perform backpropagation algorithm
void reset_partial_derivatives();  // Reset partial derivatives to zero
