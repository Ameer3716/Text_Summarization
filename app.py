import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

st.set_page_config(page_title="T5 Text Summarizer", page_icon="📝", layout="wide")

@st.cache_resource
def load_model():
    model_name = "Ameer15/T5-Text-Summarization"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

st.title("📝 T5 Text Summarization")
st.markdown("**Summarize long articles into concise summaries using fine-tuned T5 model**")

try:
    tokenizer, model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Settings")
        max_length = st.slider("Summary Length", 30, 150, 64, help="Maximum length of generated summary")
        num_beams = st.slider("Beam Search", 2, 8, 4, help="Higher values = better quality but slower")
        temperature = st.slider("Creativity", 0.5, 2.0, 1.0, 0.1, help="Higher = more creative summaries")
    
    with col1:
        st.subheader("📄 Input Article")
        article = st.text_area(
            "Paste your article here:",
            height=300,
            placeholder="Enter a long article or news text to summarize...",
            help="Enter any long-form text content"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
        with col_btn1:
            summarize_btn = st.button("🚀 Summarize", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    if summarize_btn:
        if not article.strip():
            st.warning("⚠️ Please enter some text to summarize!")
        else:
            with st.spinner("🔄 Generating summary..."):
                input_text = f"summarize: {article}"
                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        length_penalty=2.0
                    )
                
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.subheader("✨ Generated Summary")
                st.success(summary)
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Original Words", len(article.split()))
                with col_stats2:
                    st.metric("Summary Words", len(summary.split()))
                with col_stats3:
                    compression = (1 - len(summary.split()) / len(article.split())) * 100
                    st.metric("Compression", f"{compression:.1f}%")
    
    with st.expander("ℹ️ About this Model"):
        st.markdown("""
        **T5 (Text-to-Text Transfer Transformer)** fine-tuned on CNN/DailyMail dataset.
        
        **Model Details:**
        - Base Model: `t5-base`
        - Training Samples: 5,000 articles
        - ROUGE-1: 0.397
        - ROUGE-2: 0.183
        - ROUGE-L: 0.285
        
        **Best For:**
        - News articles
        - Blog posts
        - Long-form content
        
        **Tips:**
        - Longer texts get better summaries
        - Adjust beam search for quality/speed tradeoff
        - Temperature affects creativity
        """)
    
    with st.expander("📊 Example Articles"):
        st.markdown("""
        **Example 1 - Technology:**
```
        Ever noticed how plane seats appear to be getting smaller and smaller? With increasing numbers of people taking to the skies, some experts are questioning if having such packed out planes is putting passengers at risk. They say that the shrinking space on aeroplanes is not only uncomfortable - it's putting our health and safety in danger. More than squabbles over the arm rest, shrinking space on planes putting our health and safety in danger? This week, a U.S consumer advisory group set up by the Department of Transportation said at a public hearing that while the government is happy to set standards for animals flying on planes, it doesn't stipulate a minimum amount of space for humans.
```
        
        **Example 2 - Sports:**
```
        Dougie Freedman is on the verge of agreeing a new two-year deal to remain at Nottingham Forest. Freedman has stabilised Forest since he replaced cult hero Stuart Pearce and the club's owners are pleased with the job he has done at the City Ground. Nottingham Forest manager Dougie Freedman is set to sign a new contract. Freedman replaced Stuart Pearce as manager in February. The 40-year-old Scot has led Forest to ninth in the Championship table.
```
        """)

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Make sure the model is uploaded to Hugging Face Hub!")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Model: [Ameer15/T5-Text-Summarization](https://huggingface.co/Ameer15/T5-Text-Summarization)")
