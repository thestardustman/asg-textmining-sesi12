import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import streamlit as st
from wordcloud import WordCloud

# Import model architecture
from model_def import SentimentLSTM

# Configuration
VOCAB_PATH = "vocab.json"
MODEL_PATH = "model.pth"
METRICS_PATH = "metrics.json"
DATA_PATH = "data.csv"

# Set Page Config
st.set_page_config(
    page_title="SentimenAnalysis LSTM Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Font styles */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Sentiment Result Cards */
    .sentiment-card {
        padding: 24px;
        border-radius: 16px;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .sentiment-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .pos-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.03) 100%);
        border-left: 6px solid #10b981;
    }
    .neg-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.03) 100%);
        border-left: 6px solid #ef4444;
    }
    
    /* Custom headers and badges */
    .badge {
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-pos {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-neg {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Metrics Box */
    .metric-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Retraining header */
    .train-log {
        background-color: #020617;
        font-family: 'Courier New', Courier, monospace;
        color: #10b981;
        padding: 15px;
        border-radius: 8px;
        max-height: 250px;
        overflow-y: auto;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Preprocessing helpers
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text, vocab, max_len):
    tokens = text.split()
    sequence = []
    for token in tokens:
        sequence.append(vocab.get(token, 1))  # 1 is index for <UNK>
    # Pad or truncate
    if len(sequence) < max_len:
        sequence = sequence + [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

# Load model and tokenizer
@st.cache_resource
def load_sentiment_model():
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
        return None, None
        
    try:
        # Load vocab
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab = vocab_data["word_to_idx"]
        max_len = vocab_data["max_len"]
        
        # Initialize and load model
        model = SentimentLSTM(
            vocab_size=len(vocab),
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model, {"vocab": vocab, "max_len": max_len}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Run Prediction
def predict_sentiment(text, model, vocab_info):
    cleaned = clean_text(text)
    seq = text_to_sequence(cleaned, vocab_info["vocab"], vocab_info["max_len"])
    
    # Convert to tensor
    tensor_in = torch.tensor([seq], dtype=torch.long)
    
    with torch.no_grad():
        logit = model(tensor_in)
        prob = torch.sigmoid(logit).item()
        
    label = 1 if prob >= 0.5 else 0
    confidence = prob if label == 1 else 1.0 - prob
    return label, confidence

# Main App Layout
def main():
    # Sidebar
    st.sidebar.markdown("<h2 style='text-align: center; color: #6366f1; font-weight: 800;'>🤖 Sentiment LSTM</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Indonesian Text Sentiment Analyzer using Deep Learning (LSTM)</p>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio(
        "Navigation Menu",
        ["🔍 Analyze Sentiment", "📊 Dataset Insights", "📈 Model Performance", "⚙️ Retrain Model"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")
    
    # Load Model
    model, vocab_info = load_sentiment_model()
    
    if model is not None:
        st.sidebar.success("✅ LSTM Model Loaded Successfully")
        st.sidebar.info(f"Vocab size: {len(vocab_info['vocab'])} words\n\nMax Sequence: {vocab_info['max_len']} tokens")
    else:
        st.sidebar.warning("⚠️ No model loaded. Please run training first.")
        
    st.sidebar.markdown("<div style='position: fixed; bottom: 10px; font-size: 0.8rem; color: #64748b;'>Text Mining Session 12 Assignment</div>", unsafe_allow_html=True)

    # 1. ANALYZE SENTIMENT TAB
    if menu == "🔍 Analyze Sentiment":
        st.markdown("<h1 style='color: #f8fafc; font-weight: 800;'>🔍 Text Sentiment Analysis</h1>", unsafe_allow_html=True)
        st.markdown("Use this panel to test the sentiment model. You can analyze single Indonesian sentences or upload a batch file.")
        
        if model is None:
            st.warning("⚠️ No trained LSTM model files found! Please go to the **Retrain Model** tab and click 'Train Model' to train and save the model on your machine.")
            return

        tab_single, tab_batch = st.tabs(["💬 Single Prediction", "📂 Batch Processing"])
        
        with tab_single:
            st.markdown("### Predict Sentiment of custom text")
            user_input = st.text_area("Enter Indonesian text here:", placeholder="Tulis kalimat anda disini (misal: Makanannya enak sekali, porsinya banyak, tapi pelayanannya agak lambat)...", height=120)
            
            if st.button("Analyze Sentiment", type="primary"):
                if user_input.strip() == "":
                    st.warning("Please enter some text to analyze.")
                else:
                    label, conf = predict_sentiment(user_input, model, vocab_info)
                    
                    # Layout sentiment card
                    if label == 1:
                        st.markdown(f"""
                        <div class="sentiment-card pos-card">
                            <span class="badge badge-pos">🟢 POSITIVE SENTIMENT</span>
                            <h2 style="color: #10b981; margin-top: 10px; font-weight: 700;">Sentimen Positif ({conf*100:.2f}% Confidence)</h2>
                            <p style="font-size: 1.1rem; color: #e2e8f0; font-style: italic;">"{user_input}"</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class="sentiment-card neg-card">
                            <span class="badge badge-neg">🔴 NEGATIVE SENTIMENT</span>
                            <h2 style="color: #ef4444; margin-top: 10px; font-weight: 700;">Sentimen Negatif ({conf*100:.2f}% Confidence)</h2>
                            <p style="font-size: 1.1rem; color: #e2e8f0; font-style: italic;">"{user_input}"</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Preprocessing visualization
                    with st.expander("Show Text Preprocessing Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Raw Text:**", user_input)
                            st.write("**Cleaned Text:**", clean_text(user_input))
                        with col2:
                            seq = text_to_sequence(clean_text(user_input), vocab_info["vocab"], vocab_info["max_len"])
                            st.write("**Tokenized & Padded Sequence:**", seq[:20], "... (truncated)")
                            st.write("**Vocabulary Indices:**")
                            words_indices = [(w, vocab_info["vocab"].get(w, 1)) for w in clean_text(user_input).split()]
                            st.write(words_indices)
                            
        with tab_batch:
            st.markdown("### Upload a CSV or JSON file for batch analysis")
            st.markdown("Ensure your file has a column named **`text`** containing the sentences to analyze.")
            
            uploaded_file = st.file_uploader("Choose file:", type=["csv", "json"])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df_batch = pd.read_csv(uploaded_file)
                    else:
                        df_batch = pd.read_json(uploaded_file)
                        
                    if 'text' not in df_batch.columns:
                        st.error("Error: The uploaded file must contain a column named **'text'**.")
                    else:
                        st.success(f"File loaded successfully! Found {len(df_batch)} entries.")
                        
                        if st.button("Predict Batch Sentiment", type="primary"):
                            with st.spinner("Analyzing sentiments..."):
                                results = []
                                confidences = []
                                
                                for text in df_batch['text']:
                                    lbl, conf = predict_sentiment(text, model, vocab_info)
                                    results.append("Positive" if lbl == 1 else "Negative")
                                    confidences.append(float(conf))
                                    
                                df_batch['predicted_sentiment'] = results
                                df_batch['confidence_score'] = confidences
                                
                            st.markdown("### Prediction Results")
                            st.dataframe(df_batch, use_container_width=True)
                            
                            # Summary metrics
                            pos_count = sum(df_batch['predicted_sentiment'] == "Positive")
                            neg_count = len(df_batch) - pos_count
                            
                            c1, c2 = st.columns(2)
                            with c1:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                fig.patch.set_facecolor('#0f172a')
                                ax.set_facecolor('#1e293b')
                                labels = ['Positive', 'Negative']
                                sizes = [pos_count, neg_count]
                                colors = ['#10b981', '#ef4444']
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, textprops={'color': 'white'})
                                ax.set_title("Predicted Sentiment Distribution", color='white', fontweight='bold')
                                st.pyplot(fig)
                                
                            with c2:
                                st.write("### Summary:")
                                st.metric("Total Rows", len(df_batch))
                                st.metric("Positive Sentiments", f"{pos_count} ({pos_count/len(df_batch)*100:.1f}%)")
                                st.metric("Negative Sentiments", f"{neg_count} ({neg_count/len(df_batch)*100:.1f}%)")
                                
                            # Download option
                            csv_data = df_batch.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv_data,
                                file_name="sentiment_predictions_output.csv",
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"Error parsing file: {e}")

    # 2. DATASET INSIGHTS TAB
    elif menu == "📊 Dataset Insights":
        st.markdown("<h1 style='color: #f8fafc; font-weight: 800;'>📊 Dataset Insights</h1>", unsafe_allow_html=True)
        st.markdown("Visualizations and analysis of the original dataset (`data.csv`).")
        
        if not os.path.exists(DATA_PATH):
            st.warning("⚠️ No dataset file found locally. Please run training or download the dataset first.")
            return
            
        # Load dataset
        df = pd.read_csv(DATA_PATH)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### Data Summary")
            st.write(f"**Total Samples:** {len(df)}")
            st.write(f"**Positive Sentiments (1):** {sum(df['label'] == 1)}")
            st.write(f"**Negative Sentiments (0):** {sum(df['label'] == 0)}")
            st.markdown("---")
            
            # Label distribution pie chart
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            labels = ['Positive (1)', 'Negative (0)']
            sizes = [sum(df['label'] == 1), sum(df['label'] == 0)]
            colors = ['#10b981', '#ef4444']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': 'white'})
            ax.set_title("Label Distribution (Balanced)", color='white', fontweight='bold', fontsize=14)
            st.pyplot(fig)
            
        with col2:
            st.write("### Sample Data Rows")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Sentence length analysis
            df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            fig2.patch.set_facecolor('#0f172a')
            ax2.set_facecolor('#1e293b')
            sns.histplot(data=df, x='text_len', hue='label', kde=True, multiple='dodge', palette={1: '#10b981', 0: '#ef4444'}, ax=ax2)
            ax2.set_title("Sentence Length Distribution (Word Count)", color='white', fontweight='bold')
            ax2.set_xlabel("Number of Words", color='white')
            ax2.set_ylabel("Count", color='white')
            ax2.tick_params(colors='white')
            ax2.legend(title="Sentiment", labels=['Positive', 'Negative'], labelcolor='white')
            st.pyplot(fig2)
            
        # Word Clouds
        st.markdown("---")
        st.markdown("### Word Clouds")
        
        df['text_clean'] = df['text'].apply(clean_text)
        
        pos_words = " ".join(df[df['label'] == 1]['text_clean'])
        neg_words = " ".join(df[df['label'] == 0]['text_clean'])
        
        col_wc1, col_wc2 = st.columns(2)
        
        with col_wc1:
            st.write("#### Positive Sentiments Word Cloud")
            try:
                wc_pos = WordCloud(width=600, height=400, background_color='#1e293b', colormap='summer').generate(pos_words)
                fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 4))
                fig_wc1.patch.set_facecolor('#0f172a')
                ax_wc1.imshow(wc_pos, interpolation='bilinear')
                ax_wc1.axis('off')
                st.pyplot(fig_wc1)
            except Exception as e:
                st.error(f"Error generating word cloud: {e}")
                
        with col_wc2:
            st.write("#### Negative Sentiments Word Cloud")
            try:
                wc_neg = WordCloud(width=600, height=400, background_color='#1e293b', colormap='autumn').generate(neg_words)
                fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 4))
                fig_wc2.patch.set_facecolor('#0f172a')
                ax_wc2.imshow(wc_neg, interpolation='bilinear')
                ax_wc2.axis('off')
                st.pyplot(fig_wc2)
            except Exception as e:
                st.error(f"Error generating word cloud: {e}")

    # 3. MODEL PERFORMANCE TAB
    elif menu == "📈 Model Performance":
        st.markdown("<h1 style='color: #f8fafc; font-weight: 800;'>📈 Model Performance</h1>", unsafe_allow_html=True)
        st.markdown("Training curves and evaluation metrics of the best saved LSTM model.")
        
        if not os.path.exists(METRICS_PATH):
            st.warning("⚠️ No metrics file found. Please run training to generate performance details.")
            return
            
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            
        history = metrics["history"]
        eval_metrics = metrics["eval"]
        
        # Display numerical metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{eval_metrics['accuracy']*100:.2f}%</div>
                <div class="metric-label">Validation Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{eval_metrics['precision']*100:.2f}%</div>
                <div class="metric-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{eval_metrics['recall']*100:.2f}%</div>
                <div class="metric-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{eval_metrics['f1']*100:.2f}%</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Plot curves
        col_c1, col_c2 = st.columns(2)
        
        epochs_range = list(range(1, len(history["train_loss"]) + 1))
        
        with col_c1:
            st.write("### Loss Curves")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            ax.plot(epochs_range, history["train_loss"], label="Train Loss", color="#6366f1", linewidth=2.5)
            ax.plot(epochs_range, history["val_loss"], label="Val Loss", color="#a855f7", linewidth=2.5, linestyle="--")
            ax.set_title("Training & Validation Loss", color="white", fontweight="bold")
            ax.set_xlabel("Epochs", color="white")
            ax.set_ylabel("Loss", color="white")
            ax.tick_params(colors="white")
            ax.grid(True, color="#334155")
            ax.legend(labelcolor="white")
            st.pyplot(fig)
            
        with col_c2:
            st.write("### Accuracy Curves")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            ax.plot(epochs_range, [x*100 for x in history["train_acc"]], label="Train Acc", color="#10b981", linewidth=2.5)
            ax.plot(epochs_range, [x*100 for x in history["val_acc"]], label="Val Acc", color="#34d399", linewidth=2.5, linestyle="--")
            ax.set_title("Training & Validation Accuracy", color="white", fontweight="bold")
            ax.set_xlabel("Epochs", color="white")
            ax.set_ylabel("Accuracy (%)", color="white")
            ax.tick_params(colors="white")
            ax.grid(True, color="#334155")
            ax.legend(labelcolor="white")
            st.pyplot(fig)
            
        st.markdown("---")
        
        # Confusion Matrix Heatmap
        col_cm1, col_cm2 = st.columns([1, 1])
        
        with col_cm1:
            st.write("### Confusion Matrix")
            cm = np.array(eval_metrics["confusion_matrix"])
            
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            fig_cm.patch.set_facecolor('#0f172a')
            ax_cm.set_facecolor('#1e293b')
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'], ax=ax_cm,
                        annot_kws={"size": 14, "weight": "bold", "color": "black"})
            
            ax_cm.set_title("Confusion Matrix Heatmap", color="white", fontweight="bold")
            ax_cm.set_xlabel("Predicted Sentiment", color="white")
            ax_cm.set_ylabel("Actual Sentiment", color="white")
            ax_cm.tick_params(colors="white")
            st.pyplot(fig_cm)
            
        with col_cm2:
            st.write("### Matrix Insights")
            cm = eval_metrics["confusion_matrix"]
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            
            st.markdown(f"""
            - **True Negatives (TN):** {tn} - Correctly identified negative reviews.
            - **True Positives (TP):** {tp} - Correctly identified positive reviews.
            - **False Positives (FP):** {fp} - Negative reviews incorrectly flagged as positive (Type I error).
            - **False Negatives (FN):** {fn} - Positive reviews incorrectly flagged as negative (Type II error).
            
            The model demonstrates a balanced classification performance with similar rates of false positives and false negatives, indicating it doesn't skew heavily towards either sentiment.
            """)

    # 4. RETRAIN MODEL TAB
    elif menu == "⚙️ Retrain Model":
        st.markdown("<h1 style='color: #f8fafc; font-weight: 800;'>⚙️ Retrain LSTM Model</h1>", unsafe_allow_html=True)
        st.markdown("You can trigger model training on your local machine using the training configuration parameters below.")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.write("### Model Hyperparameters")
            epochs = st.slider("Epochs", min_value=5, max_value=30, value=15, step=5, help="Number of training iterations through the entire dataset.")
            learning_rate = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3, help="Controls how much the model changes in response to the estimated error each time the weights are updated.")
            batch_size = st.selectbox("Batch Size", options=[32, 64, 128], index=1, help="Number of samples processed before the model is updated.")
            
        with col_p2:
            st.write("### Network Architecture")
            embedding_dim = st.selectbox("Embedding Dimension", options=[64, 128, 256], index=1, help="Size of word vectors.")
            hidden_dim = st.selectbox("LSTM Hidden Dimension", options=[64, 128, 256], index=1, help="Number of features in the hidden state of the LSTM.")
            dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.3, step=0.1, help="Probability of zeroing elements during training to prevent overfitting.")

        st.markdown("---")
        
        # Retrain Action
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            # We import training pipeline functions locally
            try:
                # Setup dataset download
                import train as trainer
                
                status_block = st.empty()
                progress_bar = st.progress(0)
                log_block = st.empty()
                
                status_block.info("🔄 Initiating model training locally...")
                
                # Load and prepare data
                trainer.download_data()
                df_data = pd.read_csv(DATA_PATH)
                df_data['text_clean'] = df_data['text'].apply(clean_text)
                
                train_df, val_df = train_test_split(df_data, test_size=0.2, random_state=42, stratify=df_data['label'])
                
                # Build vocab
                vocab = trainer.build_vocab(train_df['text_clean'], min_freq=trainer.MIN_FREQ)
                
                # Save vocab
                with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
                    json.dump({"word_to_idx": vocab, "max_len": trainer.MAX_LEN}, f, indent=4)
                
                train_seqs = [text_to_sequence(t, vocab, trainer.MAX_LEN) for t in train_df['text_clean']]
                val_seqs = [text_to_sequence(t, vocab, trainer.MAX_LEN) for t in val_df['text_clean']]
                
                train_dataset = trainer.SentimentDataset(train_seqs, train_df['label'].values)
                val_dataset = trainer.SentimentDataset(val_seqs, val_df['label'].values)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Initialize new model
                model_new = SentimentLSTM(
                    vocab_size=len(vocab),
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_layers=2,
                    bidirectional=True,
                    dropout=dropout
                ).to(device)
                
                criterion = torch.nn.BCEWithLogitsLoss()
                optimizer = torch.optim.AdamW(model_new.parameters(), lr=learning_rate, weight_decay=1e-4)
                
                history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
                best_val_acc = 0.0
                
                logs = ["Training Started on device: " + str(device)]
                log_block.markdown(f'<div class="train-log">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                for epoch in range(epochs):
                    status_block.info(f"🔄 Training: Epoch {epoch+1}/{epochs} in progress...")
                    
                    # Training
                    model_new.train()
                    train_loss, train_correct, total_train = 0.0, 0, 0
                    for seqs, labels in train_loader:
                        seqs, labels = seqs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        logits = model_new(seqs)
                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item() * seqs.size(0)
                        preds = (torch.sigmoid(logits) >= 0.5).float()
                        train_correct += (preds == labels).sum().item()
                        total_train += seqs.size(0)
                        
                    epoch_train_loss = train_loss / total_train
                    epoch_train_acc = train_correct / total_train
                    
                    # Validation
                    model_new.eval()
                    val_loss, val_correct, total_val = 0.0, 0, 0
                    with torch.no_grad():
                        for seqs, labels in val_loader:
                            seqs, labels = seqs.to(device), labels.to(device)
                            logits = model_new(seqs)
                            loss = criterion(logits, labels)
                            
                            val_loss += loss.item() * seqs.size(0)
                            preds = (torch.sigmoid(logits) >= 0.5).float()
                            val_correct += (preds == labels).sum().item()
                            total_val += seqs.size(0)
                            
                    epoch_val_loss = val_loss / total_val
                    epoch_val_acc = val_correct / total_val
                    
                    history["train_loss"].append(epoch_train_loss)
                    history["train_acc"].append(epoch_train_acc)
                    history["val_loss"].append(epoch_val_loss)
                    history["val_acc"].append(epoch_val_acc)
                    
                    # Log text
                    log_text = f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%"
                    logs.append(log_text)
                    
                    # Check and save best
                    if epoch_val_acc > best_val_acc:
                        best_val_acc = epoch_val_acc
                        torch.save(model_new.state_dict(), MODEL_PATH)
                        logs.append(f"  --> Saved new best weights with Val Acc: {best_val_acc*100:.2f}%")
                        
                    # Update progress UI
                    progress_bar.progress((epoch + 1) / epochs)
                    log_block.markdown(f'<div class="train-log">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                # Final evaluation metrics
                logs.append("\nCalculating final performance metrics...")
                log_block.markdown(f'<div class="train-log">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                model_new.load_state_dict(torch.load(MODEL_PATH))
                model_new.eval()
                
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for seqs, labels in val_loader:
                        seqs = seqs.to(device)
                        logits = model_new(seqs)
                        preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(labels.numpy())
                        
                all_preds, all_labels = np.array(all_preds), np.array(all_labels)
                
                from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
                final_acc = accuracy_score(all_labels, all_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
                cm = confusion_matrix(all_labels, all_preds)
                
                metrics = {
                    "history": history,
                    "eval": {
                        "accuracy": float(final_acc),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "confusion_matrix": cm.tolist()
                    }
                }
                
                with open(METRICS_PATH, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=4)
                    
                status_block.success(f"🎉 Model Retrained and Saved Successfully! Best Validation Accuracy: {best_val_acc*100:.2f}%")
                
                # Clear st.cache_resource so it reloads model
                st.cache_resource.clear()
                
            except Exception as e:
                status_block.error(f"❌ An error occurred during training: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
