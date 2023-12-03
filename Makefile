# Makefile for CUDA Connected Components Finder

# Compiler settings
CC=nvcc
CFLAGS=-std=c++17 -arch=sm_80 -O2 -I./include

# Directories
SRCDIR=src
OBJDIR=obj

# Source files
SRC_CONNECTED_COMPONENTS=$(SRCDIR)/connected_components.cu
SRC_MAIN=$(SRCDIR)/main.cu

# Object files
OBJ_CONNECTED_COMPONENTS=$(OBJDIR)/connected_components.o
OBJ_MAIN=$(OBJDIR)/main.o

# Output executable
OUTPUT=cc.e

# Default target
all: $(OBJDIR) $(OUTPUT)

# Create the object file directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile connected_components.cu
$(OBJ_CONNECTED_COMPONENTS): $(SRC_CONNECTED_COMPONENTS)
	$(CC) $(CFLAGS) -c $(SRC_CONNECTED_COMPONENTS) -o $(OBJ_CONNECTED_COMPONENTS)

# Compile main.cu
$(OBJ_MAIN): $(SRC_MAIN)
	$(CC) $(CFLAGS) -c $(SRC_MAIN) -o $(OBJ_MAIN)

# Link object files and create the executable
$(OUTPUT): $(OBJ_CONNECTED_COMPONENTS) $(OBJ_MAIN)
	$(CC) $(CFLAGS) $(OBJ_CONNECTED_COMPONENTS) $(OBJ_MAIN) -o $(OUTPUT)

# Clean target to remove object files and the executable
clean:
	rm -rf $(OBJDIR) $(OUTPUT)

# Help target to display makefile usage
help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  all    - Compiles and links the project (default target)"
	@echo "  clean  - Removes object files and the executable"
	@echo "  help   - Displays this help message"

# Phony targets
.PHONY: all clean help
