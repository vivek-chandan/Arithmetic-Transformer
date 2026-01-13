# Arithmetic Transformer

A Transformer-based neural network that learns to perform arithmetic operations (addition and subtraction) on character sequences.

## Overview

This project implements an encoder-decoder Transformer model that can process arithmetic expressions like `"123+45"` and produce the correct result `"168"`. The model treats arithmetic as a sequence-to-sequence translation task, learning to compute results character by character.

## What Does This Project Do?

The Arithmetic Transformer:
- Takes arithmetic expressions as input (e.g., `"123+45"`, `"500-234"`)
- Processes them using a Transformer neural network
- Outputs the computed result (e.g., `"168"`, `"266"`)

This demonstrates how neural networks can learn mathematical reasoning through pattern recognition, rather than using traditional computational methods.

## Features

- **Flexible Operations**: Handles both addition and subtraction
- **Multi-operand Support**: Can process expressions with 2-3 operands (e.g., `"12+34-56"`)
- **Edge Case Handling**: Trained on challenging cases including:
  - Negative numbers
  - Zero operands and results
  - Leading zeros
  - Carry and borrow operations
- **Generalization Testing**: Evaluates model performance on:
  - Longer digit sequences than seen during training
  - Complex carry/borrow scenarios
  - Multi-step calculations

## Project Structure

```
.
├── iNLP_Arithematic_Transformer.ipynb  # Jupyter notebook with full implementation
├── inlp_arithematic_transformer.py     # Python script version
├── INLP_Assignment_5.pdf               # Assignment documentation
└── README.md                            # This file
```

*Note: The original files use "Arithematic" spelling - this is preserved in filenames.*

## Dataset

The project uses custom-generated datasets for training and evaluation:

### Training Data (`train.csv`)
- **30,000 samples** for model training
- Operands: 2-4 digits in length
- Mix of 2-operand (80%) and 3-operand (20%) expressions
- Balanced operations: 50% addition, 50% subtraction

### Validation Data (`val.csv`)
- **2,000 samples** for model tuning
- Used for early stopping and hyperparameter optimization
- Similar distribution to training data

### Test Sets
The model is evaluated on three specialized test sets:

1. **test1.csv** - Extrapolation Test
   - Tests generalization to longer operands (5-7 digits)
   - 2,000 samples

2. **test2.csv** - Carry/Borrow Stress Test
   - Focus on complex arithmetic with heavy carry/borrow operations
   - 2,000 samples

3. **test3.csv** - Multi-step Generalization
   - Tests 3-operand expressions
   - 2,000 samples

### Edge Case Files
Specialized datasets for diagnostic analysis:
- `carry_cases.csv` - Addition with multi-digit carries
- `borrow_cases.csv` - Subtraction requiring borrowing
- `negative_results.csv` - Expressions with negative outputs
- `leading_zeros.csv` - Numbers with leading zeros
- `long_operands.csv` - Very long numbers (6-7 digits)
- `three_operand.csv` - Three-number expressions

## Model Architecture

The Transformer model consists of:

### Encoder
- Processes the input arithmetic expression
- Multi-head self-attention layers
- Positional encoding to capture digit positions
- Feed-forward neural networks

### Decoder
- Generates the output result character by character
- Masked multi-head attention (prevents looking ahead)
- Cross-attention to encoder outputs
- Auto-regressive generation

### Key Components
- **Vocabulary**: Digits (0-9), operators (+, -), special tokens (padding, start, end)
- **Positional Encoding**: Helps model understand digit positions
- **Attention Mechanism**: Learns which parts of input to focus on
- **Feed-Forward Networks**: Process representations between attention layers

## Usage

### Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- CSV support (built-in)

### Running the Model

1. **Generate Datasets** (if needed):
```python
# The script includes dataset generation functions
generateTrainVal("train", 30000, (2,4), {2: 0.8, 3: 0.2})
generateTrainVal("val", 2000, (2,5), {2: 0.8, 3: 0.2})
generateTestSets()
```

2. **Train the Model**:
```python
# Load your data
train_loader = get_data_loader('path/to/train.csv', token_to_id, batch_size=1024)
val_loader = get_data_loader('path/to/val.csv', token_to_id, batch_size=1024)

# Train
run_training(model, train_loader, val_loader, loss_function, 
             optimizer, device, pad_id, num_epochs=50)
```

3. **Evaluate**:
```python
# Test on different datasets
evaluate_final_test_set(model, test_loader, loss_function, device, 
                       pad_id, model_path='best_model.pt')
```

## Evaluation Metrics

The model is evaluated using:
- **Exact Match Accuracy**: Percentage of completely correct results
- **Character-Level Accuracy**: Accuracy at individual digit level
- **Perplexity**: Measure of model confidence (optional)

## How It Works

1. **Input Processing**: Arithmetic expression is converted to tokens
2. **Encoding**: Encoder processes the full expression with attention
3. **Decoding**: Decoder generates result one character at a time
4. **Training**: Model learns from 30,000 examples with various patterns
5. **Inference**: Given a new expression, model predicts the result

## Learning Approach

The model doesn't use traditional arithmetic algorithms. Instead, it:
- Learns patterns from training examples
- Develops internal representations of numbers and operations
- Discovers carry/borrow mechanisms through examples
- Generalizes to unseen number combinations

## Limitations

- Performance depends on training data coverage
- May struggle with numbers significantly longer than training examples
- Not as efficient as traditional calculators for basic arithmetic
- Requires substantial training data to achieve high accuracy

## Academic Context

This project was created as part of **INLP Assignment 5**, exploring how Transformer architectures (originally designed for natural language processing) can be applied to mathematical reasoning tasks.

## Files Description

- **iNLP_Arithematic_Transformer.ipynb**: Interactive Jupyter notebook with complete implementation, visualizations, and experiments
- **inlp_arithematic_transformer.py**: Python script version auto-generated from the notebook
- **INLP_Assignment_5.pdf**: Detailed assignment documentation and requirements

*Note: File names use "Arithematic" spelling as per the original project naming.*

## Getting Started

For beginners:
1. Open `iNLP_Arithematic_Transformer.ipynb` in Jupyter or Google Colab
2. Run cells sequentially to see dataset generation, model training, and evaluation
3. Experiment with different hyperparameters or dataset configurations

For advanced users:
- Modify model architecture in the Transformer classes
- Experiment with different attention mechanisms
- Try extending to multiplication/division operations
- Analyze attention patterns to understand model behavior

## Contributing

This is an academic project, but suggestions and improvements are welcome!

## License

This project is part of an academic assignment. Please respect academic integrity guidelines if you're using this for educational purposes.

## Acknowledgments

- Assignment designed for Introduction to Natural Language Processing (INLP) course
- Transformer architecture based on "Attention Is All You Need" (Vaswani et al., 2017)
- Implementation optimized for Google Colab environment
