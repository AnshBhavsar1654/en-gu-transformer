import streamlit as st
import torch
import torch.nn as nn
import math
import os
import requests
from collections import Counter

# Set page config
st.set_page_config(
    page_title="English to Gujarati Translator",
    page_icon="🌐",
    layout="centered"
)

# Device configuration
@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# Constants (same as training)
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
dropout_rate = 0.1

# Vocab class (copied from your training code)
class Vocab:
    def __init__(self, min_freq=3):
        self.token_to_idx = {'<pad>': PAD_IDX, '<unk>': UNK_IDX, '<sos>': SOS_IDX, '<eos>': EOS_IDX}
        self.idx_to_token = {PAD_IDX: '<pad>', UNK_IDX: '<unk>', SOS_IDX: '<sos>', EOS_IDX: '<eos>'}
        self.idx = 4
        self.min_freq = min_freq
    
    def build_vocab(self, sentences):
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence.split())
        
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in self.token_to_idx:
                self.token_to_idx[word] = self.idx
                self.idx_to_token[self.idx] = word
                self.idx += 1
    
    def encode(self, sentence):
        return [self.token_to_idx.get(token, UNK_IDX) for token in sentence.split()]
    
    def decode(self, indices):
        return ' '.join([self.idx_to_token.get(idx, '<unk>') for idx in indices])
    
    def __len__(self):
        return len(self.token_to_idx)

# Model components (copied from your training code)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super().__init__()
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def generate_mask(self, src, tgt):
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(3)
        
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(device)
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src_embedded = self.dropout(self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_embedded = self.dropout(self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        return output

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive using file ID"""
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

# Function to download model if not exists
@st.cache_data
def download_model_if_needed():
    """Download the model from Google Drive if it doesn't exist locally"""
    model_path = 'transformer_en_gu.pth'
    
    if not os.path.exists(model_path):
        st.info("Model not found locally. Downloading from Google Drive...")
        
        # Extract file ID from Google Drive link
        # https://drive.google.com/file/d/1TGvSqDYXBEdw461agMNHSepDWNyY5nDZ/view?usp=sharing
        file_id = '1TGvSqDYXBEdw461agMNHSepDWNyY5nDZ'
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Downloading model file...")
            download_file_from_google_drive(file_id, model_path)
            
            progress_bar.progress(100)
            status_text.text("Model downloaded successfully!")
            
            # Verify file exists and has reasonable size
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
                st.success("✅ Model downloaded and verified!")
                return True
            else:
                st.error("❌ Downloaded model file appears to be corrupted.")
                return False
                
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            return False
    else:
        st.info("✅ Model found locally!")
        return True

# Translation function
def translate_sentence(model, sentence, src_vocab, tgt_vocab, max_len=50):
    model.eval()
    
    with torch.no_grad():
        # Encode input sentence
        tokens = src_vocab.encode(sentence)
        tokens = tokens[:max_len]
        src_tensor = torch.LongTensor([tokens + [PAD_IDX] * (max_len - len(tokens))]).to(device)

        # Prepare masks and encoder output
        src_mask = (src_tensor != PAD_IDX).unsqueeze(1).unsqueeze(2)
        src_emb = model.dropout(model.positional_encoding(model.src_embedding(src_tensor) * math.sqrt(model.d_model)))
        memory = src_emb
        for layer in model.encoder_layers:
            memory = layer(memory, src_mask)

        # Start with <sos>
        tgt_indices = [SOS_IDX]
        
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor([tgt_indices]).to(device)
            tgt_mask = (tgt_tensor != PAD_IDX).unsqueeze(1).unsqueeze(3)
            seq_len = tgt_tensor.size(1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool().to(device)
            tgt_mask = tgt_mask & nopeak_mask

            tgt_emb = model.dropout(model.positional_encoding(model.tgt_embedding(tgt_tensor) * math.sqrt(model.d_model)))
            output = tgt_emb
            for layer in model.decoder_layers:
                output = layer(output, memory, src_mask, tgt_mask)

            output = model.fc_out(output)
            next_token = output[0, -1].argmax().item()
            if next_token == EOS_IDX:
                break
            tgt_indices.append(next_token)
        
        return tgt_vocab.decode(tgt_indices[1:])

# Load model function
@st.cache_resource
def load_model():
    model_path = 'transformer_en_gu.pth'
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please check if the download was successful.")
        st.stop()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model parameters from your training code
        d_model = 512
        num_heads = 8
        num_layers = 4
        d_ff = 2048
        max_seq_len = 128
        
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
        
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, src_vocab, tgt_vocab
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("This might be due to:")
        st.error("1. Corrupted model file")
        st.error("2. Model architecture mismatch")
        st.error("3. PyTorch version compatibility issues")
        st.stop()

# Streamlit UI
def main():
    st.title("🌐 English to Gujarati Translator")
    st.markdown("---")
    
    # Download model if needed
    if not download_model_if_needed():
        st.stop()
    
    # Model info sidebar
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Device:** {device}")
        st.write("**Architecture:** Transformer")
        st.write("**Language Pair:** English → Gujarati")
        st.write("**Model Source:** Google Drive")
        
        if st.button("Clear History", type="secondary"):
            st.session_state.translation_history = []
            st.rerun()
    
    # Initialize session state for history
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model, src_vocab, tgt_vocab = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()
    
    # Input section
    st.subheader("Enter English Text")
    input_text = st.text_area(
        "Type your English sentence here:",
        height=100,
        placeholder="e.g., I am feeling very hungry"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        translate_button = st.button("Translate", type="primary")
    
    # Translation
    if translate_button and input_text.strip():
        with st.spinner("Translating..."):
            try:
                translation = translate_sentence(model, input_text.strip(), src_vocab, tgt_vocab)
                
                # Display result
                st.subheader("Translation Result")
                st.markdown(f"**English:** {input_text}")
                st.markdown(f"**Gujarati:** {translation}")
                
                # Add to history
                st.session_state.translation_history.append({
                    'english': input_text,
                    'gujarati': translation
                })
                
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
    
    elif translate_button and not input_text.strip():
        st.warning("Please enter some text to translate.")
    
    # Translation history
    if st.session_state.translation_history:
        st.markdown("---")
        st.subheader("Translation History")
        
        for i, translation in enumerate(reversed(st.session_state.translation_history[-5:]), 1):
            with st.expander(f"Translation {i}"):
                st.write(f"**English:** {translation['english']}")
                st.write(f"**Gujarati:** {translation['gujarati']}")
    
    # Sample sentences
    st.markdown("---")
    st.subheader("Try Sample Sentences")
    sample_sentences = [
        "I am feeling very hungry",
        "Can I get your contact number?",
        "What is your name?",
        "How are you today?",
        "Nice to meet you"
    ]
    
    cols = st.columns(len(sample_sentences))
    for i, sentence in enumerate(sample_sentences):
        with cols[i]:
            if st.button(f"'{sentence}'", key=f"sample_{i}"):
                st.session_state.sample_input = sentence
                st.rerun()
    
    # Handle sample input
    if hasattr(st.session_state, 'sample_input'):
        input_text = st.session_state.sample_input
        with st.spinner("Translating sample..."):
            try:
                translation = translate_sentence(model, input_text, src_vocab, tgt_vocab)
                st.success(f"**{input_text}** → **{translation}**")
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
        del st.session_state.sample_input

if __name__ == "__main__":
    main()
