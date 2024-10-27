CC=gcc                     # Compiler to use
CFLAGS=-Wall -Wextra -O2   # Compilation flags: Wall for all warnings, Wextra for extra warnings, O2 for optimizations
TARGET=mlp                 # Name of the final executable

# Rule to create the final executable
$(TARGET): mlp.c
	$(CC) $(CFLAGS) mlp.c -o $(TARGET) -lm

# Rule for cleaning up
clean:
	rm -f $(TARGET)
