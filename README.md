# LSTM Gold Price Prediction from Scratch

A professional implementation of LSTM neural network from scratch using only NumPy to predict gold prices. This project fetches real-time gold price data from Pegadaian API and trains an LSTM model to forecast future prices.

## Features

- 🏗️ **LSTM Implementation from Scratch**: Pure NumPy implementation without TensorFlow/PyTorch
- 📊 **Real-time Data Fetching**: Automatic data retrieval from Pegadaian Gold API
- 🔄 **Professional OOP Design**: Clean, modular, and maintainable code structure
- 📈 **Advanced Preprocessing**: Feature engineering with technical indicators
- 🎯 **Multiple Prediction Modes**: Single-step and multi-step forecasting
- 📊 **Comprehensive Visualization**: Training progress, predictions, and technical analysis
- 🔍 **Performance Metrics**: Regression, trading, and time-series evaluation
- 💾 **Model Persistence**: Save and load trained models with full pipeline
- 📝 **Professional Logging**: Comprehensive logging and error handling
- ⚙️ **Configurable Pipeline**: YAML-based configuration system
- 🚀 **Easy-to-Use Scripts**: Demo, examples, and main execution scripts

## Project Structure

```txt
LSTM-Gold-from-Scratch/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_fetcher.py      # Gold price data fetching
│   │   └── preprocessor.py      # Data preprocessing and feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_cell.py         # LSTM cell implementation
│   │   ├── lstm_layer.py        # LSTM layer implementation
│   │   └── lstm_network.py      # Complete LSTM network
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training logic and checkpointing
│   │   └── optimizer.py         # Adam and SGD optimizers
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Comprehensive evaluation metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py           # Professional visualization tools
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging utilities
│       └── config.py            # YAML configuration management
├── models/                      # Saved models and checkpoints
├── logs/                       # Training and execution logs
├── data/                       # Data cache and raw files
├── visualizations/             # Generated plots and charts
├── main.py                     # Main execution script
├── demo.py                     # Quick demo script
├── examples.py                 # Usage examples and tutorials
├── config.yaml                 # Default configuration file
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd LSTM-Gold-from-Scratch
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create required directories:

```bash
mkdir -p models logs data visualizations
```

## Quick Start

### 1. Run the Demo (Recommended)

```bash
# Quick demo with default settings
python demo.py

# Ultra-quick test (30 days data, 3 epochs)
python demo.py quick
```

### 2. Full Pipeline Execution

```bash
# Run complete pipeline with default configuration
python main.py

# Run with custom configuration
python main.py --config custom_config.yaml

# Train only
python main.py --mode train

# Predict future prices (requires trained model)
python main.py --mode predict --days 30

# Evaluate existing model
python main.py --mode evaluate --model-path models/my_model.pkl
```

### 3. Interactive Examples

```bash
# Run usage examples with different configurations
python examples.py
```

## Usage Examples

### Basic Usage

```python
from main import GoldPredictionPipeline

# Initialize the pipeline
pipeline = GoldPredictionPipeline()

# Run complete pipeline
pipeline.run_full_pipeline()
```

### Custom Configuration

```python
from main import GoldPredictionPipeline

# Initialize with custom config
pipeline = GoldPredictionPipeline('custom_config.yaml')

# Or modify configuration programmatically
pipeline.config['training']['epochs'] = 50
pipeline.config['model']['hidden_sizes'] = [128, 64, 32]

# Setup and run
pipeline.setup_components()
pipeline.fetch_data()
pipeline.preprocess_data()
history = pipeline.train_model()
```

### Advanced Usage

```python
from src.data.data_fetcher import GoldDataFetcher
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_network import LSTMNetwork
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import ModelEvaluator
from src.visualization.plotter import Visualizer

# Manual pipeline construction
data_fetcher = GoldDataFetcher()
preprocessor = DataPreprocessor(sequence_length=60)
model = LSTMNetwork(input_size=10, hidden_sizes=[64, 32], output_size=1)
trainer = LSTMTrainer(model)
evaluator = ModelEvaluator()
visualizer = Visualizer()

# Custom training loop
raw_data = data_fetcher.fetch_gold_prices(days=1000)
processed_data = preprocessor.fit_transform(raw_data)
sequences, targets = preprocessor.create_sequences(
    processed_data['features'], processed_data['targets']
)

splits = preprocessor.train_test_split(sequences, targets)
history = trainer.train(
    splits['X_train'], splits['y_train'],
    splits['X_val'], splits['y_val'],
    epochs=50, batch_size=32
)

# Evaluation and visualization
predictions = model.predict(splits['X_test'])
metrics = evaluator.calculate_all_metrics(splits['y_test'], predictions)
visualizer.create_dashboard({
    'history': history,
    'metrics': metrics,
    'actual': splits['y_test'],
    'predicted': predictions
})
```

## Configuration

The system uses YAML-based configuration. Key parameters:

```yaml
# Model Configuration
model:
  sequence_length: 60              # Input sequence length
  hidden_sizes: [64, 32, 16]       # LSTM layer sizes
  dropout_rate: 0.2                # Dropout for regularization

# Training Configuration  
training:
  epochs: 100                      # Maximum training epochs
  batch_size: 32                   # Training batch size
  learning_rate: 0.001             # Initial learning rate
  early_stopping_patience: 10     # Early stopping patience

# Data Configuration
data:
  days_to_fetch: 3650              # Historical data days
  validation_split: 0.2            # Validation data fraction
  test_split: 0.1                  # Test data fraction
```

## API Data Source

This project uses the Pegadaian Gold Price API:

- **URL**: <https://sahabat.pegadaian.co.id/gold/prices/chart?interval=3650&isRequest=true>
- **Data**: Historical gold prices for the last 3650 days
- **Fields**: Selling price (hargaJual), buying price (hargaBeli), timestamp

## Model Architecture

The LSTM implementation includes:

### Core Components

- **LSTM Cells**: Complete implementation with forget, input, and output gates
- **LSTM Layers**: Multi-layer LSTM with proper state management
- **Dense Output**: Final prediction layer with configurable activation
- **Dropout**: Regularization to prevent overfitting

### Mathematical Implementation

- **Gates**: σ(Wf·[ht-1,xt] + bf) for forget gate, similar for input/output
- **Cell State**: Ct = ft * Ct-1 + it * C̃t
- **Hidden State**: ht = ot * tanh(Ct)
- **Gradients**: Full backpropagation through time (BPTT)

### Features

- **Adam Optimizer**: Adaptive learning rate with momentum
- **Gradient Clipping**: Prevents exploding gradients  
- **Early Stopping**: Prevents overfitting
- **Checkpointing**: Save best models during training

## Performance Metrics

### Regression Metrics

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

### Trading Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum portfolio decline
- **Win Rate**: Percentage of profitable predictions
- **Profit Factor**: Ratio of gross profit to gross loss

### Time Series Metrics

- **Directional Accuracy**: Trend prediction accuracy
- **Uptrend/Downtrend Accuracy**: Specific trend accuracies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the CC0 1.0 Universal - see the LICENSE file for details.

## Acknowledgments

- Pegadaian for providing the gold price API
- NumPy community for the excellent numerical computing library
Complete LSTM implementation from scratch for predicting Indonesian gold prices (IDR). Built entirely with NumPy to demonstrate deep learning fundamentals without frameworks.
