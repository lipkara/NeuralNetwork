#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mlp.h"

Network_t network;
Input_t train_set[N];
Input_t test_set[N];


int neuronsPerLayer[4] = {H1, H2, H3, p}; // Hidden + output

ActivationFunction activation_function = TANH;// Set this to LOGISTIC, TANH, or RELU as needed

double partial_derivatives[WEIGHTS_NUM];  // This array will be used to update the weights after each batch.

int train_c1, train_c2, train_c3, train_c4;
int test_c1, test_c2, test_c3, test_c4;

/*
 *This function organizes each input in a category
 */
void encode_input(double x1, double x2, Input_t *input, int i, char *type)
{
    if ((pow((x1 - 0.5), 2) + pow((x2 - 0.5), 2) < 0.2) && x1 > 0.5)
        categorize("C1", input, i, type);
    else if ((pow((x1 - 0.5), 2) + pow((x2 - 0.5), 2) < 0.2) && x1 < 0.5)
        categorize("C2", input, i, type);
    else if ((pow((x1 + 0.5), 2) + pow((x2 + 0.5), 2) < 0.2) && x1 > -0.5)
        categorize("C1", input, i, type);
    else if ((pow((x1 + 0.5), 2) + pow((x2 + 0.5), 2) < 0.2) && x1 < -0.5)
        categorize("C2", input, i, type);
    else if ((pow((x1 - 0.5), 2) + pow((x2 + 0.5), 2) < 0.2) && x1 > 0.5)
        categorize("C1", input, i, type);
    else if ((pow((x1 - 0.5), 2) + pow((x2 + 0.5), 2) < 0.2) && x1 < 0.5)
        categorize("C2", input, i, type);
    else if ((pow((x1 + 0.5), 2) + pow((x2 - 0.5), 2) < 0.2) && x1 > -0.5)
        categorize("C1", input, i, type);
    else if ((pow((x1 + 0.5), 2) + pow((x2 - 0.5), 2) < 0.2) && x1 < -0.5)
        categorize("C2", input, i, type);
    else if (x1 > 0)
        categorize("C3", input, i, type);
    else
        categorize("C4", input, i, type);
}

/*
 * This function encodes each input according to their respective category
 * using the 1-out of-p encoding method
 */
void categorize(char *category, Input_t *input, int i, char *type)
{
    // Reset categories for the current input
    for (int j = 0; j < 4; ++j) {
        input[i].category[j] = 0;
    }

    // Variable to hold the category index
    int categoryIndex = -1;

    if (strcmp(category, "C1") == 0) {
        categoryIndex = 0;
    } else if (strcmp(category, "C2") == 0) {
        categoryIndex = 1;
    } else if (strcmp(category, "C3") == 0) {
        categoryIndex = 2;
    } else {
        categoryIndex = 3;
    }

    // Set the appropriate category
    input[i].category[categoryIndex] = 1;

    // Increment the corresponding counter
    if (strcmp(type, "test") == 0) {
        switch (categoryIndex) {
            case 0: test_c1++; break;
            case 1: test_c2++; break;
            case 2: test_c3++; break;
            case 3: test_c4++; break;
        }
    } else {
        switch (categoryIndex) {
            case 0: train_c1++; break;
            case 1: train_c2++; break;
            case 2: train_c3++; break;
            case 3: train_c4++; break;
        }
    }
}

double calculate_activation_function(double sum)
{
    switch(activation_function)
    {
        case LOGISTIC:
            return 1 / (1 + exp(-sum));
        case TANH:
            return (exp(sum) - exp(-sum)) / (exp(sum) + exp(-sum));
        case RELU:
            return sum > 0.0 ? sum : 0.0;
        default:
            return 0;
    }
}

double calculate_activation_derivative(double y)
{
    switch(activation_function)
    {
        case LOGISTIC:
            return y * (1 - y);
        case TANH:
            return 1 - pow(y, 2);
        case RELU:
            return y <= 0 ? 0 : 1;
        default:
            return 0;
    }
}

void forward_pass(Input_t x)
{
    double u_i = 0.0;

    for (int layer_index = 0; layer_index < H; layer_index++) //for each layer
    {
        if (layer_index == 0) // first hidden layer has input from x
        {
            for (int neuron_index = 0; neuron_index < neuronsPerLayer[layer_index]; neuron_index++)
            {
                u_i += network.layers[layer_index].neuron[neuron_index].w[0]; // add bias
                u_i += network.layers[layer_index].neuron[neuron_index].w[1] * x.x1;
                u_i += network.layers[layer_index].neuron[neuron_index].w[2] * x.x2;

                network.layers[layer_index].neuron[neuron_index].output = calculate_activation_function(u_i);

                u_i = 0.0;
            }
        }
        else //Second layer until output layer
        {
            u_i = 0.0;
            for (int neuron_index = 0; neuron_index < neuronsPerLayer[layer_index]; neuron_index++)
            {
                u_i += network.layers[layer_index].neuron[neuron_index].w[0];

                for (int j = 0; j < neuronsPerLayer[layer_index-1]; j++)
                {
                    u_i += network.layers[layer_index].neuron[neuron_index].w[j+1] * network.layers[layer_index-1].neuron[j].output;
                }

                if (layer_index == HL)
                {
                    network.layers[layer_index].neuron[neuron_index].output = 1 / (double)(1 + exp(-u_i)); // Always use logistic for the output layer
                }
                else
                {
                    network.layers[layer_index].neuron[neuron_index].output = calculate_activation_function(u_i);
                }

                u_i = 0.0;
            }
        }
    }

    for (int i = 0; i < p; i++)
    {
        network.layers[HL].neuron[i].error_signal = network.layers[HL].neuron[i].output - x.category[i];
    }
}

void reverse_pass()
{
    // Calculate δi for every neuron in the network
    for (int layer_index = HL; layer_index >= 0; layer_index--)
    {
        if (layer_index == HL) //Output layer
        {
            for (int neuron_index = 0; neuron_index < p; neuron_index++)
            {
                double u_i = network.layers[layer_index].neuron[neuron_index].output;
                network.layers[layer_index].neuron[neuron_index].delta_i = network.layers[HL].neuron[neuron_index].error_signal * calculate_activation_derivative(u_i);
            }
        }
        else // Hidden layers
        {
            for (int neuron_index = 0; neuron_index < neuronsPerLayer[layer_index]; neuron_index++)
            {
                double weighted_sum = 0.0;

                // Calculate the sum of the weights and deltas to the next layer
                for (int next_layer_neuron_index = 0; next_layer_neuron_index < neuronsPerLayer[layer_index + 1]; next_layer_neuron_index++)
                {
                    weighted_sum += network.layers[layer_index + 1].neuron[next_layer_neuron_index].w[neuron_index + 1] * network.layers[layer_index + 1].neuron[next_layer_neuron_index].delta_i;
                }

                double u_i = network.layers[layer_index].neuron[neuron_index].output;
                network.layers[layer_index].neuron[neuron_index].delta_i = calculate_activation_derivative(u_i) * weighted_sum;
                weighted_sum = 0.0;
            }
        }
    }
}


/**
 * Calculate derivatives for the first hidden layer.
 *
 * @param layer_index The index of the current layer.
 * @param x The input features.
 */
void calculate_first_layer_derivatives(int layer_index, Input_t x)
{
    for (int neuron_index = 0; neuron_index < neuronsPerLayer[layer_index]; neuron_index++)
    {
        Neuron_t *neuron = &network.layers[layer_index].neuron[neuron_index];
        neuron->error_derivative[0] = neuron->delta_i;
        neuron->error_derivative[1] = neuron->delta_i * x.x1;
        neuron->error_derivative[2] = neuron->delta_i * x.x2;
    }
}

/**
 * Calculate derivatives for layers other than the first.
 *
 * @param layer_index The index of the current layer.
 */
void calculate_other_layers_derivatives(int layer_index)
{
    for (int neuron_index = 0; neuron_index < neuronsPerLayer[layer_index]; neuron_index++)
    {
        Neuron_t *neuron = &network.layers[layer_index].neuron[neuron_index];
        neuron->error_derivative[0] = neuron->delta_i;

        for (int neuron_index = 1; neuron_index <= neuronsPerLayer[layer_index - 1]; neuron_index++)
        {
            neuron->error_derivative[neuron_index] = neuron->delta_i * network.layers[layer_index - 1].neuron[neuron_index - 1].output;
        }
    }
}

/**
 * Calculate the error derivatives for each neuron in the network.
 *
 * @param x The input structure containing input features.
 */
void calculate_partial_derivatives(Input_t x)
{
    // Iterate over all layers from last to first
    for (int layer_index = HL; layer_index >= 0; layer_index--)
    {
        // Process the first hidden layer separately
        if (layer_index == 0)
        {
            calculate_first_layer_derivatives(layer_index, x);
        }
        else // Process the rest of the layers
        {
            calculate_other_layers_derivatives(layer_index);
        }
    }
}

void backprop(Input_t x)
{
    forward_pass(x);
    reverse_pass();
    calculate_partial_derivatives(x);
}

void reset_partial_derivatives()
{
    for (int i = 0; i < WEIGHTS_NUM; i++)
    {
        partial_derivatives[i] = 0.0;
    }
}

void train_using_gradient_descent()
{
    int epoch = 0;       // We use this to check whether certain epochs have passed
    int input_count = 0; // We use this counter to check whether an epoch has passed
    int p_d_counter = 0;
    double sum = 0.0;

    FILE *training_errors = fopen("train_errors.csv", "w+");
    fprintf(training_errors, "%s,%s\n", "Error", "Epoch");

    printf("   __ ___________             .__       .__                 __   \n");
    printf("  / / \\__    ___/___________  |__| ____ |__| ____    ____   \\ \\  \n");
    printf(" / /    |    |  \\_  __ \\__  \\ |  |/    \\|  |/    \\  / ___\\   \\ \\ \n");
    printf(" \\ \\    |    |   |  | \\// __ \\|  |   |  \\  |   |  \\/ /_/  >  / / \n");
    printf("  \\_\\   |____|   |__|  (____  /__|___|  /__|___|  /\\___  /  /_/  \n");
    printf("                            \\/        \\/        \\//_____/        \n");
    printf("\n");

    while(epoch < EPOCHS)
    {
        // Reset partial derivatives;
        for (int i = 0; i < WEIGHTS_NUM; i++)
        {
            partial_derivatives[i] = 0.0;
        }

        // Start the gradient descent
        for (int b = 0; b < B; b++)
        {
            backprop(train_set[input_count]); // perform forward and reverse pass and calculate partial derivatives;
            input_count++;

            for (int h = 0; h < H; h++)
            {
                for (int i = 0; i < neuronsPerLayer[h]; i++)
                {
                    if (h == 0) // first hidden layer
                    {
                        for (int j = 0; j < d + 1; j++)
                        {
                            partial_derivatives[p_d_counter] += network.layers[h].neuron[i].error_derivative[j];
                            p_d_counter++;
                        }
                    }
                    else // All other layers
                    {
                        for (int j = 0; j < neuronsPerLayer[h-1] + 1; j++)
                        {
                            partial_derivatives[p_d_counter] += network.layers[h].neuron[i].error_derivative[j];
                            p_d_counter++;
                        }
                    }
                }
            }
            p_d_counter = 0;
        }

        p_d_counter = 0;
        // Update the weights
        for (int h = 0; h < H; h++)
        {
            for (int i = 0; i < neuronsPerLayer[h]; i++)
            {
                if (h == 0) // First hidden layer
                {
                    for (int j = 0; j < d + 1; j++)
                    {
                        network.layers[h].neuron[i].w[j] -= n * partial_derivatives[p_d_counter];
                        p_d_counter++;
                    }
                }
                else
                {
                    for (int j = 0; j < neuronsPerLayer[h-1] + 1; j++)
                    {
                        network.layers[h].neuron[i].w[j] -= n * partial_derivatives[p_d_counter];
                        p_d_counter++;
                    }
                }
            }
        }
        p_d_counter = 0;

        if(input_count == N)  // If an epoch is done calculate global error
        {
            //Calculate error
            for (int i = 0; i < N; i++)
            {
                forward_pass(train_set[i]);

                for (int j = 0; j < p; j++)
                {
                    sum += pow(train_set[i].category[j] - network.layers[HL].neuron[j].output, 2);
                }
                sum /= 2.0;
            }

            fprintf(training_errors, "%lf,%d\n", sum, epoch + 1);

            printf("Epoch: %d/%d, Train error: %lf\n", epoch + 1, EPOCHS, sum);

            epoch++;
            input_count = 0;
            sum = 0.0;
            fflush(stdout);
        }
    }
    printf("\n\n");
    printf("   __ ___________.__       .__       .__               .___ __   \n");
    printf("  / / \\_   _____/|__| ____ |__| _____|  |__    ____  __| _/ \\ \\  \n");
    printf(" / /   |    __)  |  |/    \\|  |/  ___/  |  \\\\_/ __ \\/ __ |   \\ \\ \n");
    printf(" \\ \\   |    \\    |  |   |  \\  |\\___ \\|  Y  \\   ___// /_/ |   / / \n");
    printf("  \\_\\  \\___  /   |__|___|  /__/____  >___|  /\\___  >____ |  /_/  \n");
    printf("           \\/            \\/        \\/     \\/     \\/     \\/       \n");
}


void init() {
    // Load datasets
    FILE *train = fopen("train_dataset.csv", "r");
    FILE *test = fopen("test_dataset.csv", "r");

    if(train == NULL || test == NULL)
    {
        printf("File could not be opened");
        exit(1);
    }

    double x1, x2;
    char category[4]; // Corrected buffer size to accommodate category label and null terminator

    for (int i = 0; i < N; i++)
    {
        // Read x1, x2, and the category for each entry in the training set
        fscanf(train, "%lf,%lf,%2s", &x1, &x2, category);
        train_set[i].x1 = x1;
        train_set[i].x2 = x2;
        categorize(category, train_set, i, "train");

        // Read x1, x2, and the category for each entry in the testing set
        fscanf(test, "%lf,%lf,%2s", &x1, &x2, category);
        test_set[i].x1 = x1;
        test_set[i].x2 = x2;
        categorize(category, test_set, i, "test");
    }

    fclose(train);
    fclose(test);

    // initialize the network
    srand(time(NULL)); // Ensure different random weights each time

    // Allocate memory for all neurons and initialize weights
    for (int layer = 0; layer < H; layer++) {
        network.layers[layer].neuron = (Neuron_t *) malloc(sizeof(Neuron_t) * neuronsPerLayer[layer]);
        if (network.layers[layer].neuron == NULL) {
            fprintf(stderr, "Failed to allocate memory for neurons in layer %d\n", layer);
            exit(EXIT_FAILURE);
        }

        for (int neuron = 0; neuron < neuronsPerLayer[layer]; neuron++) {
            int weightCount = (layer == 0) ? d + 1 : neuronsPerLayer[layer - 1] + 1;
            network.layers[layer].neuron[neuron].w = malloc(sizeof(double) * weightCount);
            network.layers[layer].neuron[neuron].error_derivative = malloc(sizeof(double) * weightCount);

            if (!network.layers[layer].neuron[neuron].w || !network.layers[layer].neuron[neuron].error_derivative) {
                fprintf(stderr, "Failed to allocate memory for weights or derivatives in layer %d, neuron %d\n", layer, neuron);
                exit(EXIT_FAILURE);
            }

            for (int weight = 0; weight < weightCount; weight++) {
                network.layers[layer].neuron[neuron].w[weight] = RANDOM_DOUBLE(-1.0, 1.0);
            }
        }
    }
}

void calculate_generalization_capability()
{
    double max = -1;
    int winner = 0;
    int category_1 = 0;
    int category_2 = 0;
    int category_3 = 0;
    int category_4 = 0;

    double error = 0.0;

    FILE *correct_guesses = fopen("correct_guesses.csv", "w+");
    FILE *wrong_guesses = fopen("wrong_guesses.csv", "w+");

    fprintf(correct_guesses, "%s,%s\n", "x", "y");
    fprintf(wrong_guesses, "%s,%s\n", "x", "y");


    for (int i = 0; i < N; i++)
    {
        forward_pass(test_set[i]);
        for (int h = 0; h < p; h++)
        {
            if (network.layers[HL].neuron[h].output > max)
            {
                max = network.layers[HL].neuron[h].output;
                winner = h;
            }
        }

        if (winner == 0 && test_set[i].category[0] == 1)
        {
            fprintf(correct_guesses, "%.5lf,%.5lf\n",test_set[i].x1, test_set[i].x2);
            category_1++;
        }
        else if(winner == 1 && test_set[i].category[1] == 1)
        {
            fprintf(correct_guesses, "%.5lf,%.5lf\n",test_set[i].x1, test_set[i].x2);
            category_2++;
        }
        else if(winner == 2 && test_set[i].category[2] == 1)
        {
            fprintf(correct_guesses, "%.5lf,%.5lf\n",test_set[i].x1, test_set[i].x2);
            category_3++;
        }
        else if(winner == 3 && test_set[i].category[3] == 1)
        {
            fprintf(correct_guesses, "%.5lf,%.5lf\n",test_set[i].x1, test_set[i].x2);
            category_4++;
        }
        else
        {
            fprintf(wrong_guesses, "%.5lf,%.5lf\n",test_set[i].x1, test_set[i].x2);
        }

        max = -1;
    }

    error = (1.0 - (category_1 + category_2 + category_3 + category_4)/ (double ) N) * 100.0;

    printf("C1: %d guessed correctly\n", category_1);
    printf("C2: %d guessed correctly\n", category_2);
    printf("C3: %d guessed correctly\n", category_3);
    printf("C4: %d guessed correctly\n", category_4);
    printf("Total guessed correctly: %d\n", category_1 + category_2 + category_3 + category_4);
    printf("Total guessed wrong: %d\n", N - (category_1 + category_2 + category_3 + category_4));
    printf("Error percentage: %.2lf%c\n", error, '%');
    printf("Accuracy: %.2lf%c\n", 100 - error, '%');
}


int main() {
    init();
    train_using_gradient_descent(); // train the network using gradient descent
    calculate_generalization_capability(); // Use the testing set to check the generalization capabilities of the network
    exit(0);
}
