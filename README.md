# English to Gujarati Machine Translation

A Transformer-based neural machine translation system that translates English text to Gujarati using PyTorch and Streamlit.

## Features

- **Transformer Architecture**: Built from scratch using PyTorch with multi-head attention mechanisms
- **Interactive Web Interface**: User-friendly Streamlit app for real-time translation
- **Translation History**: Keep track of your previous translations
- **Sample Sentences**: Quick testing with pre-defined examples

## Architecture

The model implements the classic Transformer architecture with:
- **Encoder-Decoder Structure**: 4 layers each with 8 attention heads
- **Multi-Head Attention**: Self-attention and cross-attention mechanisms
- **Positional Encoding**: Sinusoidal position embeddings
- **Feed-Forward Networks**: 2048-dimensional hidden layers
- **Layer Normalization**: Applied after each sub-layer
- **Dropout Regularization**: 0.1 dropout rate for training stability

### Model Specifications
- **Model Dimension (d_model)**: 512
- **Number of Heads**: 8
- **Number of Layers**: 4 (encoder) + 4 (decoder)
- **Feed-Forward Dimension**: 2048
- **Maximum Sequence Length**: 128 tokens
- **Vocabulary**: Built from training data with minimum frequency threshold

## Dataset

The model is trained on the English-Gujarati parallel corpus available on Kaggle: [English to Gujarati Machine Translation Dataset](https://www.kaggle.com/datasets/parvmodi/english-to-gujarati-machine-translation-dataset)

The dataset contains paired sentences in both languages for supervised learning.

**Important Note**: Due to computational limitations, this model was trained on a subset (150,000 sentences) of the complete dataset. The translation accuracy is therefore limited and could be significantly improved with:
- Training on the full dataset
- Longer training duration
- Larger model architecture

## Quick Start

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install streamlit
pip install numpy
pip install tqdm
pip install sacrebleu  # for evaluation metrics
```

### Running the Streamlit App

1. **Ensure model file exists**: Make sure `transformer_en_gu.pth` is in the same directory

2. **Launch the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the interface**: Open your browser and navigate to `http://localhost:8501`


## Evaluation Metrics

The model is evaluated using standard machine translation metrics:
- **BLEU Score**: Measures n-gram overlap between prediction and reference
- **CHRF Score**: Character-level F-score
- **TER Score**: Translation Error Rate

## Performance Notes

### Current Limitations
- **Limited Training Data**: Only 150,000 sentence pairs used due to computational constraints
- **Basic Decoding**: Uses greedy decoding instead of beam search
- **Vocabulary Coverage**: Limited by minimum frequency threshold
- **Training Duration**: Shorter training may lead to suboptimal convergence


## License

This project is open source and available under the MIT License.
