# Indonesian Text Sentiment Analysis with LSTM
This repository contains an Indonesian text sentiment analysis application built using PyTorch LSTM and Streamlit. The application features a web interface that loads a pre-trained bidirectional LSTM model to classify sentences into positive or negative sentiment.

This project was developed for my Text Mining class assignment.

## Model Performance
The model was trained locally on the Indonesian reviews dataset:
* **Architecture**: Bidirectional LSTM with word embedding and global average pooling layers.
* **Accuracy**: 99.88%
* **Precision**: 100.00%
* **Recall**: 99.76%
* **F1-Score**: 99.88%
* **Epochs**: 15 epochs

## Project Structure
* `app.py`: Streamlit web application.
* `model_def.py`: PyTorch LSTM model class definition.
* `train.py`: Preprocessing and model training script.
* `requirements.txt`: Python dependencies list.
* `vocab.json`: Vocabulary word-to-index mapping.
* `model.pth`: Pre-trained model weights.
* `metrics.json`: Training history and evaluation metrics.
* `data.csv`: Indonesian reviews dataset.
* `run_app.bat`: Launcher script to run the Streamlit app locally.

## Running Locally

Follow these steps to run the application on your local machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/asg-textmining-sesi12.git
   cd asg-textmining-sesi12
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py # Or just... Double-clik on the run_app.bat 
   ```
