# # pip install streamlit transformers sentencepiece torch plotly
# # pip install streamlit-option-menu streamlit-lottie requests

# import streamlit as st
# import re
# import time
# import plotly.graph_objects as go
# import plotly.express as px
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# import torch
# from streamlit_option_menu import option_menu
# import pandas as pd
# from datetime import datetime
# import json

# # Page configuration
# st.set_page_config(
#     page_title="AI Text Summarizer",
#     page_icon="📝",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 1rem;
#         background: linear-gradient(90deg, #1f77b4, #17becf);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
    
#     .sub-header {
#         font-size: 1.2rem;
#         color: #666;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
    
#     .metric-card {
#         background: linear-gradient(45deg, #f0f8ff, #e6f3ff);
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 4px solid #1f77b4;
#         margin: 0.5rem 0;
#     }
    
#     .summary-box {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem;
#         border-radius: 15px;
#         color: white;
#         margin: 1rem 0;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
    
#     .info-box {
#         background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#         border: 1px solid #ff9a9e;
#     }
    
#     .stButton > button {
#         background: linear-gradient(45deg, #1f77b4, #17becf);
#         color: white;
#         border: none;
#         padding: 0.75rem 2rem;
#         border-radius: 25px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
#     }
    
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
#     }
    
#     .feature-card {
#         background: white;
#         padding: 1.5rem;
#         border-radius: 15px;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#         margin: 1rem 0;
#         border-top: 3px solid #1f77b4;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'summaries' not in st.session_state:
#     st.session_state.summaries = []
# if 'model_loaded' not in st.session_state:
#     st.session_state.model_loaded = False
# if 'model' not in st.session_state:
#     st.session_state.model = None
# if 'tokenizer' not in st.session_state:
#     st.session_state.tokenizer = None

# # Load model function with caching
# @st.cache_resource
# def load_model():
#     """Load the T5 model and tokenizer with caching"""
#     try:
#         with st.spinner("🔄 Loading AI Model... Please wait"):
#             model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
#             tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
            
#             # Set device
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             model = model.to(device)
            
#             return model, tokenizer, device
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None, None, None

# # Text cleaning function
# def clean_text(text: str) -> str:
#     """Clean and preprocess text"""
#     text = re.sub(r'\r\n', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'<.*?>', '', text)
#     text = text.strip().lower()
#     return text

# # Summarization function
# def summarize_dialogue(dialogue: str, model, tokenizer, device, max_length=150, num_beams=4):
#     """Generate summary from dialogue"""
#     dialogue = clean_text(dialogue)
    
#     inputs = tokenizer(
#         dialogue, 
#         return_tensors="pt", 
#         truncation=True, 
#         padding="max_length", 
#         max_length=512
#     )
#     inputs = {key: value.to(device) for key, value in inputs.items()}
    
#     # Generate summary
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs["input_ids"],
#             max_length=max_length,
#             num_beams=num_beams,
#             early_stopping=True,
#             do_sample=True,
#             temperature=0.7
#         )
    
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return summary

# # Text statistics function
# def get_text_stats(text):
#     """Calculate text statistics"""
#     words = len(text.split())
#     characters = len(text)
#     sentences = len(re.findall(r'[.!?]+', text))
#     return {
#         'words': words,
#         'characters': characters,
#         'sentences': sentences,
#         'avg_words_per_sentence': round(words/max(sentences, 1), 2)
#     }

# # Create visualization for text statistics
# def create_stats_chart(original_stats, summary_stats):
#     """Create comparison chart for text statistics"""
#     categories = ['Words', 'Characters', 'Sentences']
#     original_values = [original_stats['words'], original_stats['characters'], original_stats['sentences']]
#     summary_values = [summary_stats['words'], summary_stats['characters'], summary_stats['sentences']]
    
#     fig = go.Figure(data=[
#         go.Bar(name='Original', x=categories, y=original_values, marker_color='#1f77b4'),
#         go.Bar(name='Summary', x=categories, y=summary_values, marker_color='#17becf')
#     ])
    
#     fig.update_layout(
#         title="Text Statistics Comparison",
#         xaxis_title="Metrics",
#         yaxis_title="Count",
#         barmode='group',
#         template='plotly_white',
#         height=400
#     )
    
#     return fig

# # Sidebar navigation
# with st.sidebar:
#     st.markdown("## 🎯 Navigation")
#     selected = option_menu(
#         menu_title=None,
#         options=["Summarizer", "Analytics", "History", "About"],
#         icons=["file-text", "bar-chart", "clock-history", "info-circle"],
#         menu_icon="cast",
#         default_index=0,
#         styles={
#             "container": {"padding": "0!important", "background-color": "#fafafa"},
#             "icon": {"color": "#1f77b4", "font-size": "18px"},
#             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
#             "nav-link-selected": {"background-color": "#1f77b4"},
#         }
#     )
    
#     st.markdown("---")
#     st.markdown("### ⚙️ Settings")
    
#     # Model settings
#     max_length = st.slider("Summary Length", 50, 300, 150)
#     num_beams = st.slider("Quality (Beams)", 2, 8, 4)
    
#     st.markdown("---")
#     st.markdown("### 📊 Quick Stats")
#     if st.session_state.summaries:
#         total_summaries = len(st.session_state.summaries)
#         st.metric("Total Summaries", total_summaries)
        
#         avg_reduction = sum([s['reduction_ratio'] for s in st.session_state.summaries]) / total_summaries
#         st.metric("Avg Reduction", f"{avg_reduction:.1f}%")

# # Main content area
# if selected == "Summarizer":
#     # Header
#     st.markdown('<h1 class="main-header">🤖 AI Text Summarizer</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Transform long dialogues into concise, meaningful summaries using advanced AI</p>', unsafe_allow_html=True)
    
#     # Load model
#     if not st.session_state.model_loaded:
#         model, tokenizer, device = load_model()
#         if model is not None:
#             st.session_state.model = model
#             st.session_state.tokenizer = tokenizer
#             st.session_state.device = device
#             st.session_state.model_loaded = True
#             st.success("✅ Model loaded successfully!")
#         else:
#             st.error("❌ Failed to load model. Please check your model files.")
#             st.stop()
    
#     # Main interface
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown("### 📝 Input Text")
#         dialogue_input = st.text_area(
#             "Enter your dialogue or text to summarize:",
#             height=300,
#             placeholder="Paste your dialogue here...\n\nExample:\nPerson A: Hello, how are you today?\nPerson B: I'm doing great, thanks for asking. How about you?\nPerson A: I'm doing well too. I wanted to discuss our project timeline..."
#         )
        
#         col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
#         with col_btn1:
#             if st.button("🚀 Generate Summary", type="primary"):
#                 if dialogue_input.strip():
#                     with st.spinner("🔄 Generating summary..."):
#                         progress_bar = st.progress(0)
#                         for i in range(100):
#                             progress_bar.progress(i + 1)
#                             time.sleep(0.01)
                        
#                         summary = summarize_dialogue(
#                             dialogue_input, 
#                             st.session_state.model, 
#                             st.session_state.tokenizer, 
#                             st.session_state.device,
#                             max_length=max_length,
#                             num_beams=num_beams
#                         )
                        
#                         # Calculate statistics
#                         original_stats = get_text_stats(dialogue_input)
#                         summary_stats = get_text_stats(summary)
#                         reduction_ratio = ((original_stats['words'] - summary_stats['words']) / original_stats['words']) * 100
                        
#                         # Store in session state
#                         st.session_state.summaries.append({
#                             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                             'original_text': dialogue_input,
#                             'summary': summary,
#                             'original_stats': original_stats,
#                             'summary_stats': summary_stats,
#                             'reduction_ratio': reduction_ratio
#                         })
                        
#                         progress_bar.empty()
#                         st.success("✅ Summary generated successfully!")
#                 else:
#                     st.warning("⚠️ Please enter some text to summarize.")
        
#         with col_btn2:
#             if st.button("🗑️ Clear Text"):
#                 st.rerun()
        
#         with col_btn3:
#             if st.button("📋 Sample Text"):
#                 sample_text = """Person A: Good morning! I hope you're doing well. I wanted to discuss our quarterly sales report and the upcoming marketing campaign.

# Person B: Good morning! Yes, I'm doing great, thank you. I've been reviewing the numbers from last quarter, and I think we have some interesting insights to share.

# Person A: That's excellent! What are the key highlights you've found in the data?

# Person B: Well, our online sales increased by 35% compared to the previous quarter, which is fantastic. However, our in-store sales dropped by 15%, which we need to address.

# Person A: That's a significant shift towards digital. What do you think is driving this change?

# Person B: I believe it's a combination of factors - the improved user experience on our website, better mobile optimization, and of course, the ongoing preference for online shopping that started during the pandemic.

# Person A: Makes sense. How should we adjust our marketing strategy for the next quarter?

# Person B: I think we should allocate more budget to digital advertising and social media campaigns. We should also consider improving our in-store experience to attract more foot traffic."""
                
#                 st.session_state.sample_text = sample_text
#                 st.rerun()
    
#     with col2:
#         st.markdown("### 🎯 Features")
        
#         features = [
#             {"icon": "🧠", "title": "AI-Powered", "desc": "Advanced T5 transformer model"},
#             {"icon": "⚡", "title": "Fast Processing", "desc": "Quick and efficient summarization"},
#             {"icon": "📊", "title": "Detailed Analytics", "desc": "Comprehensive text statistics"},
#             {"icon": "🎨", "title": "Clean Interface", "desc": "User-friendly design"},
#             {"icon": "📱", "title": "Responsive", "desc": "Works on all devices"},
#             {"icon": "🔒", "title": "Privacy", "desc": "Your data stays local"}
#         ]
        
#         for feature in features:
#             st.markdown(f"""
#             <div class="feature-card">
#                 <h4>{feature['icon']} {feature['title']}</h4>
#                 <p>{feature['desc']}</p>
#             </div>
#             """, unsafe_allow_html=True)
    
#     # Display results
#     if st.session_state.summaries:
#         latest_summary = st.session_state.summaries[-1]
        
#         st.markdown("---")
#         st.markdown("### 📋 Summary Results")
        
#         # Summary display
#         st.markdown(f"""
#         <div class="summary-box">
#             <h3>🎯 Generated Summary</h3>
#             <p style="font-size: 1.1rem; line-height: 1.6;">{latest_summary['summary']}</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Statistics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric(
#                 "Original Words", 
#                 latest_summary['original_stats']['words'],
#                 delta=None
#             )
        
#         with col2:
#             st.metric(
#                 "Summary Words", 
#                 latest_summary['summary_stats']['words'],
#                 delta=f"-{latest_summary['original_stats']['words'] - latest_summary['summary_stats']['words']}"
#             )
        
#         with col3:
#             st.metric(
#                 "Reduction Ratio", 
#                 f"{latest_summary['reduction_ratio']:.1f}%",
#                 delta=None
#             )
        
#         with col4:
#             st.metric(
#                 "Sentences", 
#                 latest_summary['summary_stats']['sentences'],
#                 delta=f"{latest_summary['summary_stats']['sentences'] - latest_summary['original_stats']['sentences']}"
#             )

# elif selected == "Analytics":
#     st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
#     if st.session_state.summaries:
#         latest_summary = st.session_state.summaries[-1]
        
#         # Statistics comparison chart
#         st.markdown("### 📈 Text Statistics Comparison")
#         chart = create_stats_chart(latest_summary['original_stats'], latest_summary['summary_stats'])
#         st.plotly_chart(chart, use_container_width=True)
        
#         # Summary trends
#         if len(st.session_state.summaries) > 1:
#             st.markdown("### 📉 Summary Trends")
            
#             df = pd.DataFrame([
#                 {
#                     'Timestamp': s['timestamp'],
#                     'Original Words': s['original_stats']['words'],
#                     'Summary Words': s['summary_stats']['words'],
#                     'Reduction Ratio': s['reduction_ratio']
#                 } for s in st.session_state.summaries
#             ])
            
#             fig = px.line(df, x='Timestamp', y='Reduction Ratio', 
#                          title='Reduction Ratio Over Time',
#                          markers=True)
#             st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("📊 No summaries available yet. Create some summaries to see analytics!")

# elif selected == "History":
#     st.markdown('<h1 class="main-header">📚 Summary History</h1>', unsafe_allow_html=True)
    
#     if st.session_state.summaries:
#         for i, summary in enumerate(reversed(st.session_state.summaries)):
#             with st.expander(f"📄 Summary {len(st.session_state.summaries) - i} - {summary['timestamp']}"):
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("**Original Text:**")
#                     st.text_area("", value=summary['original_text'], height=150, disabled=True, key=f"orig_{i}")
                
#                 with col2:
#                     st.markdown("**Summary:**")
#                     st.text_area("", value=summary['summary'], height=150, disabled=True, key=f"sum_{i}")
                
#                 # Stats
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Original Words", summary['original_stats']['words'])
#                 with col2:
#                     st.metric("Summary Words", summary['summary_stats']['words'])
#                 with col3:
#                     st.metric("Reduction", f"{summary['reduction_ratio']:.1f}%")
        
#         # Clear history button
#         if st.button("🗑️ Clear History"):
#             st.session_state.summaries = []
#             st.success("History cleared!")
#             st.rerun()
#     else:
#         st.info("📝 No summaries in history yet. Start creating summaries to build your history!")

# elif selected == "About":
#     st.markdown('<h1 class="main-header">ℹ️ About This App</h1>', unsafe_allow_html=True)
    
#     st.markdown("""
#     ### 🚀 AI Text Summarizer
    
#     This application uses advanced AI technology to automatically summarize long dialogues and texts into concise, meaningful summaries.
    
#     ### 🔧 Technology Stack
#     - **Model**: T5 (Text-to-Text Transfer Transformer)
#     - **Framework**: Streamlit
#     - **Visualization**: Plotly
#     - **Backend**: PyTorch & Transformers
    
#     ### 🎯 Key Features
#     - **Intelligent Summarization**: Uses state-of-the-art T5 model
#     - **Real-time Processing**: Fast and efficient text processing
#     - **Interactive Analytics**: Comprehensive statistics and visualizations
#     - **History Management**: Keep track of all your summaries
#     - **Responsive Design**: Works perfectly on all devices
    
#     ### 📖 How to Use
#     1. Enter or paste your text in the input area
#     2. Adjust settings in the sidebar if needed
#     3. Click "Generate Summary" to process your text
#     4. View results with detailed analytics
#     5. Check your history anytime
    
#     ### 🛠️ Model Information
#     The T5 model is fine-tuned for dialogue summarization and provides high-quality, contextually relevant summaries.
#     """)
    
#     st.markdown("---")
#     st.markdown("### 💡 Tips for Better Summaries")
    
#     tips = [
#         "Use clear, well-structured dialogues for best results",
#         "Longer texts generally produce better summaries",
#         "Adjust the summary length based on your needs",
#         "Higher beam values produce more refined summaries",
#         "Review the analytics to understand text reduction patterns"
#     ]
    
#     for tip in tips:
#         st.markdown(f"• {tip}")

# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style="text-align: center; padding: 2rem; color: #666;">
#         <p>🤖 AI Text Summarizer | Built with Streamlit & T5 | © 2024</p>
#     </div>
#     """, 
#     unsafe_allow_html=True
# )






















# pip install streamlit tensorflow transformers sentencepiece plotly
# pip install streamlit-option-menu streamlit-lottie requests
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
import time
import json
from datetime import datetime
from streamlit_option_menu import option_menu
import requests
from typing import Dict, List
import numpy as np

# Configure page settings
st.set_page_config(
    page_title="AI Text Summarization Suite",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# AI Text Summarization Suite\n\nPowered by Advanced Deep Learning Models"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .summary-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    
    .processing-animation {
        text-align: center;
        padding: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summarization_history' not in st.session_state:
    st.session_state.summarization_history = []

if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {
        'total_summaries': 0,
        'total_words_processed': 0,
        'avg_compression_ratio': 0,
        'processing_times': []
    }

class TextSummarizationSystem:
    """
    Professional Text Summarization System using TensorFlow/Transformers
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the summarization model (commented for testing)"""
        try:
            # Uncomment these lines when you have the actual model
            # from transformers import T5ForConditionalGeneration, T5Tokenizer
            # self.model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
            # self.tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
            # 
            # # Ensure the model is on the correct device
            # device = "cuda" if tf.config.list_physical_devices('GPU') else "cpu"
            # self.model = self.model.to(device)
            # self.model_loaded = True
            
            # For testing purposes, simulate model loading
            time.sleep(2)  # Simulate loading time
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
        text = text.strip()  # Strip whitespace
        return text
    
    def get_text_statistics(self, text: str) -> Dict:
        """Get statistics about the input text"""
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'character_count': len(text),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1)
        }
    
    def summarize_text(self, text: str, max_length: int = 150) -> Dict:
        """
        Summarize the input text
        For testing, this returns a mock summary
        """
        start_time = time.time()
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Get text statistics
        stats = self.get_text_statistics(cleaned_text)
        
        # For testing purposes, create a mock summary
        # Uncomment the following lines when you have the actual model:
        
        # inputs = self.tokenizer(cleaned_text, return_tensors="pt", 
        #                        truncation=True, padding="max_length", max_length=512)
        # 
        # outputs = self.model.generate(
        #     inputs["input_ids"],
        #     max_length=max_length,
        #     num_beams=4,
        #     early_stopping=True
        # )
        # 
        # summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Mock summary for testing
        sentences = cleaned_text.split('.')[:3]  # Take first 3 sentences
        summary = '. '.join(sentences).strip() + '.'
        if len(summary) > max_length:
            summary = summary[:max_length] + '...'
        
        processing_time = time.time() - start_time
        
        # Calculate compression ratio
        compression_ratio = len(summary) / len(cleaned_text) if cleaned_text else 0
        
        return {
            'summary': summary,
            'original_stats': stats,
            'summary_stats': self.get_text_statistics(summary),
            'processing_time': processing_time,
            'compression_ratio': compression_ratio
        }

# Initialize the summarization system
@st.cache_resource
def load_summarization_system():
    """Load and cache the summarization system"""
    system = TextSummarizationSystem()
    return system

def create_metrics_dashboard():
    """Create a metrics dashboard"""
    stats = st.session_state.processing_stats
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats['total_summaries']}</h3>
            <p>Total Summaries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats['total_words_processed']:,}</h3>
            <p>Words Processed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_ratio = stats['avg_compression_ratio']
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_ratio:.1%}</h3>
            <p>Avg Compression</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_time = np.mean(stats['processing_times']) if stats['processing_times'] else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_time:.2f}s</h3>
            <p>Avg Processing Time</p>
        </div>
        """, unsafe_allow_html=True)

def create_performance_charts():
    """Create performance visualization charts"""
    if not st.session_state.processing_stats['processing_times']:
        st.info("No processing data available yet. Process some text to see performance metrics!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing time trend
        times = st.session_state.processing_stats['processing_times'][-10:]  # Last 10 operations
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=times,
            mode='lines+markers',
            name='Processing Time',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Processing Time Trend",
            xaxis_title="Recent Operations",
            yaxis_title="Time (seconds)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Compression ratio distribution
        history = st.session_state.summarization_history[-10:]
        if history:
            ratios = [item['compression_ratio'] for item in history]
            fig = px.histogram(
                x=ratios,
                nbins=10,
                title="Compression Ratio Distribution",
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(
                xaxis_title="Compression Ratio",
                yaxis_title="Frequency",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Text Summarization Suite</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform long texts into concise, meaningful summaries using advanced AI</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("robot.png", use_column_width=True)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Home", "📊 Analytics", "📋 History", "⚙️ Settings"],
            icons=["house", "bar-chart", "clock-history", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
    "container": {
        "padding": "0!important",
        "background-color": "#1e1e1e",  # Match your sidebar's dark background
        'color': '#ffffff'             # Light text on dark bg
    },
    "icon": {
        "color": "#C69749",            # Your theme gold
        "font-size": "20px"
    },
    "nav-link": {
        "font-size": "16px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#282A3A",    # Dark slate on hover
        'color': '#f0f0f0',            # Light text
        'font-weight': '500'
    },
    "nav-link-selected": {
        "background-color": "#735F32", # Your theme bronze
        'color': '#ffffff',            # White text for contrast
        'font-weight': '600'
    }
}

        )
    
    # Main content based on navigation
    if selected == "🏠 Home":
        show_home_page()
    elif selected == "📊 Analytics":
        show_analytics_page()
    elif selected == "📋 History":
        show_history_page()
    elif selected == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Main summarization interface"""
    
    # Load the summarization system
    system = load_summarization_system()
    
    # Model status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not system.model_loaded:
            if st.button("🚀 Load AI Model", type="primary"):
                with st.spinner("Loading advanced AI model..."):
                    success = system.load_model()
                    if success:
                        st.success("✅ Model loaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load model")
        else:
            st.success("✅ AI Model Ready")
    
    with col2:
        st.markdown("**Model Status:**")
        status = "🟢 Online" if system.model_loaded else "🔴 Offline"
        st.markdown(f"<p style='color: {'green' if system.model_loaded else 'red'};'>{status}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input section
    st.markdown("### 📝 Input Text")
    
    # Input options
    input_method = st.radio(
        "Choose input method:",
        ["✍️ Type/Paste Text", "📁 Upload File"],
        horizontal=True
    )
    
    text_input = ""
    
    if input_method == "✍️ Type/Paste Text":
        text_input = st.text_area(
            "Enter the text you want to summarize:",
            height=200,
            placeholder="Paste your dialogue, article, or any text here...",
            help="Enter any text you'd like to summarize. The AI will process it and generate a concise summary."
        )
    
    elif input_method == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt', 'md'],
            help="Upload a .txt or .md file to summarize"
        )
        
        if uploaded_file:
            text_input = str(uploaded_file.read(), "utf-8")
            st.text_area("File content preview:", text_input[:500] + "..." if len(text_input) > 500 else text_input, height=100)
    
    # Summarization options
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider(
            "Summary Length (words)",
            min_value=50,
            max_value=300,
            value=150,
            step=25,
            help="Adjust the maximum length of the summary"
        )
    
    with col2:
        summary_style = st.selectbox(
            "Summary Style",
            ["Balanced", "Detailed", "Concise", "Key Points"],
            help="Choose the style of summarization"
        )
    
    # Process button
    if st.button("🎯 Generate Summary", type="primary", disabled=not system.model_loaded):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to summarize!")
            return
        
        # Show processing animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🧠 AI is analyzing your text..."):
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("🔍 Analyzing text structure...")
                elif i < 60:
                    status_text.text("🧠 Extracting key information...")
                elif i < 90:
                    status_text.text("✨ Generating summary...")
                else:
                    status_text.text("🎉 Finalizing results...")
                time.sleep(0.01)
        
        # Get summary
        result = system.summarize_text(text_input, max_length)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.markdown("### 📋 Summary Results")
        
        # Summary box
        st.markdown(f"""
        <div class="summary-box">
            <h4>📝 Generated Summary</h4>
            <p>{result['summary']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Original Words",
                result['original_stats']['word_count'],
                help="Number of words in the original text"
            )
        
        with col2:
            st.metric(
                "Summary Words",
                result['summary_stats']['word_count'],
                help="Number of words in the summary"
            )
        
        with col3:
            st.metric(
                "Compression Ratio",
                f"{result['compression_ratio']:.1%}",
                help="Ratio of summary length to original length"
            )
        
        # Detailed statistics
        with st.expander("📊 Detailed Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Text Statistics:**")
                st.json(result['original_stats'])
            
            with col2:
                st.markdown("**Summary Statistics:**")
                st.json(result['summary_stats'])
        
        # Update session state
        st.session_state.summarization_history.append({
            'timestamp': datetime.now().isoformat(),
            'original_text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
            'summary': result['summary'],
            'compression_ratio': result['compression_ratio'],
            'processing_time': result['processing_time']
        })
        
        # Update stats
        stats = st.session_state.processing_stats
        stats['total_summaries'] += 1
        stats['total_words_processed'] += result['original_stats']['word_count']
        stats['processing_times'].append(result['processing_time'])
        
        # Update average compression ratio
        all_ratios = [item['compression_ratio'] for item in st.session_state.summarization_history]
        stats['avg_compression_ratio'] = sum(all_ratios) / len(all_ratios)
        
        # Download option
        st.download_button(
            label="💾 Download Summary",
            data=result['summary'],
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def show_analytics_page():
    """Analytics and performance dashboard"""
    st.markdown("### 📊 Analytics Dashboard")
    
    # Metrics overview
    create_metrics_dashboard()
    
    st.markdown("---")
    
    # Performance charts
    st.markdown("### 📈 Performance Metrics")
    create_performance_charts()
    
    # Processing efficiency
    if st.session_state.summarization_history:
        st.markdown("### ⚡ Processing Efficiency")
        
        df = pd.DataFrame(st.session_state.summarization_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Efficiency over time
        fig = px.line(
            df, 
            x='timestamp', 
            y='processing_time',
            title="Processing Time Over Time",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

def show_history_page():
    """Show summarization history"""
    st.markdown("### 📋 Summarization History")
    
    if not st.session_state.summarization_history:
        st.info("No summarization history yet. Create some summaries to see them here!")
        return
    
    # History controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search history:", placeholder="Search summaries...")
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.summarization_history = []
            st.success("History cleared!")
            st.rerun()
    
    # Display history
    history = st.session_state.summarization_history
    if search_term:
        history = [item for item in history if search_term.lower() in item['summary'].lower()]
    
    for i, item in enumerate(reversed(history)):
        with st.expander(f"Summary {len(history) - i}: {item['timestamp'][:19]}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Original Text:**")
                st.text(item['original_text'])
                
                st.markdown("**Summary:**")
                st.markdown(f"<div class='summary-box'>{item['summary']}</div>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Compression", f"{item['compression_ratio']:.1%}")
                st.metric("Processing Time", f"{item['processing_time']:.2f}s")
                
                st.download_button(
                    "💾 Download",
                    data=item['summary'],
                    file_name=f"summary_{i}.txt",
                    key=f"download_{i}"
                )

def show_settings_page():
    """Settings and configuration page"""
    st.markdown("### ⚙️ Settings & Configuration")
    
    # Model settings
    st.markdown("#### 🤖 Model Configuration")
    
    with st.expander("Model Settings"):
        st.selectbox(
            "Model Type",
            ["T5-Base", "T5-Small", "T5-Large", "BART", "Pegasus"],
            help="Choose the summarization model"
        )
        
        st.slider(
            "Max Input Length",
            min_value=256,
            max_value=2048,
            value=512,
            help="Maximum input sequence length"
        )
        
        st.slider(
            "Number of Beams",
            min_value=1,
            max_value=8,
            value=4,
            help="Number of beams for beam search"
        )
    
    # UI settings
    st.markdown("#### 🎨 Interface Settings")
    
    with st.expander("UI Preferences"):
        theme = st.selectbox(
            "Theme",
            ["Professional", "Dark", "Light", "Colorful"],
            help="Choose your preferred theme"
        )
        
        st.checkbox("Enable animations", value=True)
        st.checkbox("Show detailed metrics", value=True)
        st.checkbox("Auto-save summaries", value=False)
    
    # Export/Import settings
    st.markdown("#### 📤 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export History"):
            if st.session_state.summarization_history:
                df = pd.DataFrame(st.session_state.summarization_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    data=csv,
                    file_name="summarization_history.csv",
                    mime="text/csv"
                )
            else:
                st.info("No history to export")
    
    with col2:
        uploaded_history = st.file_uploader(
            "📤 Import History",
            type=['csv'],
            help="Upload a CSV file with summarization history"
        )
        
        if uploaded_history:
            try:
                df = pd.read_csv(uploaded_history)
                imported_history = df.to_dict('records')
                st.session_state.summarization_history.extend(imported_history)
                st.success(f"Imported {len(imported_history)} records!")
            except Exception as e:
                st.error(f"Error importing history: {str(e)}")

if __name__ == "__main__":
    main()