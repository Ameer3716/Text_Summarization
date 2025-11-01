# 📝 T5 Text Summarization

AI-powered text summarization using fine-tuned T5 model on CNN/DailyMail dataset.

## 🚀 Features
- Summarize long articles into concise summaries
- Adjustable summary length and creativity
- Based on T5-base architecture
- ROUGE scores: R-1: 0.397, R-2: 0.183, R-L: 0.285

## 🎯 Model
Hosted on Hugging Face: [Ameer15/T5-Text-Summarization](https://huggingface.co/Ameer15/T5-Text-Summarization)

## 📊 Training Details
- Dataset: CNN/DailyMail
- Training Samples: 5,000 articles
- Validation Samples: 500
- Test Samples: 300
- Epochs: 1
- Base Model: t5-base

## 🌐 Live Demo
Try it on [Streamlit Cloud](https://share.streamlit.io)

## 💻 Local Usage
```bash
git clone https://github.com/Ameer3716/Text_Summarization.git
cd Text_Summarization
pip install -r requirements.txt
streamlit run app.py
```

## 📈 Performance
- ROUGE-1: 39.70%
- ROUGE-2: 18.25%
- ROUGE-L: 28.49%

## 🛠️ Tech Stack
- Transformers (Hugging Face)
- PyTorch
- Streamlit
- T5 Architecture

## 📝 Example
**Input:** "Ever noticed how plane seats appear to be getting smaller..."

**Output:** "U.S consumer advisory group says minimum space must be stipulated. Tests conducted by FAA use planes with more leg room than airlines offer."

## 👨‍💻 Author
**Ameer Sultan**
- GitHub: [@Ameer3716](https://github.com/Ameer3716)
- Hugging Face: [@Ameer15](https://huggingface.co/Ameer15)

## 📄 License
MIT License
