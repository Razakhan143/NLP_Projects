import streamlit as st
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from googletrans import LANGUAGES  # Keep for language codes
import pyperclip  # For copy to clipboard functionality
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Language Detector & Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling with subtle colors
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    .main-header h1 {
        color: #1e293b;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #64748b;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Language bar styling */
    .language-bar {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .language-bar-content {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .language-icon {
        font-size: 1.1rem;
    }
    
    .language-label {
        font-weight: 600;
        color: #374151;
        font-size: 0.95rem;
    }
    
    .auto-detect-badge {
        background: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-left: auto;
    }
    
    /* Translation output styling */
    .translation-output {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        min-height: 250px;
        padding: 1rem;
        margin-top: 0.5rem;
        position: relative;
        transition: border-color 0.3s ease;
    }
    
    .translation-output.has-content {
        border-color: #3b82f6;
        background: #f8fafc;
    }
    
    .output-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 220px;
        color: #94a3b8;
        text-align: center;
    }
    
    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .placeholder-text {
        font-size: 1rem;
        font-style: italic;
    }
    
    .translation-content {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #000000;
        padding: 0.5rem 0;
    }
    
    .detected-language-info {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .detected-language-info .language-icon {
        color: #0369a1;
    }
    
    .detected-language-info .language-text {
        color: #0f172a;
        font-size: 0.9rem;
    }
    
    .detected-language-info .language-name {
        font-weight: 600;
        color: #0369a1;
    }
    
    /* Success message styling */
    .success-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        color: #0f172a;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
        border: 1px solid #bae6fd;
    }
    
    .success-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
        color: #0369a1;
    }
    
    .success-card p {
        margin: 0;
        font-size: 1rem;
        color: #475569;
    }
    
    /* Translation result styling */
    .translation-result {
        background: linear-gradient(135deg, #fafafa 0%, #f4f4f5 100%);
        color: #1e293b;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #e4e4e7;
    }
    
    .translation-result h4 {
        margin: 0 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        color: #374151;
    }
    
    .translation-text {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 1.1rem;
        line-height: 1.6;
        border-left: 4px solid #3b82f6;
        color: #1f2937;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(71, 85, 105, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(71, 85, 105, 0.3);
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
        background-color: #ffffff;
        color: #000000;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
        background-color: #ffffff;
    }
    
    /* Warning and error styling */
    .stWarning {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
    }
    
    .stError {
        background-color: #fee2e2;
        border: 1px solid #ef4444;
        color: #dc2626;
    }
    
    .stSuccess {
        background-color: #d1fae5;
        border: 1px solid #10b981;
        color: #047857;
    }
    
    /* Metric styling */
    .stMetric {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .stMetric label {
        color: #64748b;
        font-size: 0.9rem;
    }
    
    .stMetric div[data-testid="metric-value"] {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .spinner {
        border: 4px solid #f1f5f9;
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom spinner for Streamlit */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    .footer-text p {
        margin: 0.5rem 0;
    }
    
    /* History item styling */
    .history-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .history-item:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .history-item-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #64748b;
    }
    
    .history-item-content {
        font-size: 0.95rem;
        color: #1e293b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .card {
            padding: 1.5rem;
        }
        
        .translation-result {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def detect_and_translate(text, target_lang):
    """Detect source language and translate text with improved error handling"""
    try:
        # Detect source language with retry logic
        max_retries = 2
        retry_count = 0
        detected_lang = None
        
        while retry_count < max_retries and detected_lang is None:
            try:
                detected_lang = detect(text)
            except LangDetectException:
                retry_count += 1
                if retry_count == max_retries:
                    detected_lang = "en"  # Fallback to English
                time.sleep(0.5)
        
        # Translate using deep_translator.GoogleTranslator
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_text = None
        retry_count = 0
        
        while retry_count < max_retries and translated_text is None:
            try:
                translated_text = translator.translate(text)
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
                time.sleep(1)
        
        return detected_lang, translated_text, None, None  # No pronunciation support in deep_translator
    except Exception as e:
        return None, None, None, str(e)

def save_to_history(source_text, translated_text, source_lang, target_lang):
    """Save translation to session history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_item = {
        'timestamp': timestamp,
        'source_text': source_text,
        'translated_text': translated_text,
        'source_lang': source_lang,
        'target_lang': target_lang
    }
    
    # Add to beginning of history (newest first)
    st.session_state.history.insert(0, history_item)
    
    # Limit history to 20 items
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]

def main():
    # Initialize session state
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""
    if 'detected_lang' not in st.session_state:
        st.session_state.detected_lang = ""
    if 'show_translation' not in st.session_state:
        st.session_state.show_translation = False
    if 'pronunciation' not in st.session_state:
        st.session_state.pronunciation = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Language Detector & Translator</h1>
        <p>Detect languages automatically and translate text with precision</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for side-by-side layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Dynamic language detection display
        if st.session_state.show_translation and st.session_state.detected_lang:
            detected_lang_name = LANGUAGES.get(st.session_state.detected_lang, 'Unknown').title()
            st.markdown(f"""
            <div class="language-bar">
                <div class="language-bar-content">
                    <span class="language-icon">🔍</span>
                    <span class="language-label">Source Language</span>
                    <span class="auto-detect-badge">{detected_lang_name}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="language-bar">
                <div class="language-bar-content">
                    <span class="language-icon">🔍</span>
                    <span class="language-label">Source Language</span>
                    <span class="auto-detect-badge">Auto-Detect</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        text = st.text_area(
            "Enter your text here:",
            height=250,
            placeholder="Type or paste your text here...",
            label_visibility="collapsed",
            key="input_text"
        )
        
        # Word count for input
        if text:
            input_word_count = len(text.split())
            input_char_count = len(text)
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Input Words", input_word_count)
            with col1b:
                st.metric("Input Characters", input_char_count)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Language dropdown using Googletrans LANGUAGES
        language_options = list(LANGUAGES.items())
        language_options.sort(key=lambda x: x[1])  # Sort by name
        
        lang_dict = {v.title(): k for k, v in LANGUAGES.items()}  # 'English': 'en'
        selected_lang_name = st.selectbox(
            "Select target language:",
            list(lang_dict.keys()),
            index=list(lang_dict.keys()).index('English'),
            label_visibility="collapsed",
            key="target_lang"
        )
        target_lang = lang_dict[selected_lang_name]
        
        # Translation output area
        if st.session_state.show_translation and st.session_state.translated_text:
            detected_lang_name = LANGUAGES.get(st.session_state.detected_lang, 'Unknown').title()
            st.markdown(f"""
            <div class="translation-output has-content">
                <div class="detected-language-info">
                    <span class="language-icon">🌍</span>
                    <span class="language-text">Detected Language: <span class="language-name">{detected_lang_name}</span></span>
                </div>
                <div class="translation-content">
                    {st.session_state.translated_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # No pronunciation support in deep_translator
            if st.session_state.pronunciation:
                st.markdown(f"""
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #64748b;">
                    <strong>Pronunciation:</strong> {st.session_state.pronunciation}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="translation-output">
                <div class="output-placeholder">
                    <div class="placeholder-icon">📝</div>
                    <div class="placeholder-text">Translation will appear here...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Translate button
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    if st.button("🚀 Translate Now"):
        if text.strip() == "":
            st.warning("⚠️ Please enter some text to translate.")
        else:
            # Show loading spinner
            with st.spinner("🔄 Detecting language and translating..."):
                detected_lang, translated_text, pronunciation, error = detect_and_translate(text, target_lang)
            
            if error:
                st.error(f"❌ An error occurred: {error}")
                st.session_state.show_translation = False
            else:
                # Update session state
                st.session_state.translated_text = translated_text
                st.session_state.detected_lang = detected_lang
                st.session_state.pronunciation = pronunciation
                st.session_state.show_translation = True
                
                # Save to history
                save_to_history(text, translated_text, detected_lang, target_lang)
                
                # Show success message
                st.success("✅ Translation completed successfully!")
                
                # Rerun to update the display
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional features (only show if translation exists)
    if st.session_state.show_translation and st.session_state.translated_text:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Copy Translation"):
                try:
                    pyperclip.copy(st.session_state.translated_text)
                    st.success("Translation copied to clipboard!")
                except Exception as e:
                    st.error("Could not copy to clipboard. Please try again.")
        
        with col2:
            if st.button("🔄 Swap Languages"):
                if st.session_state.detected_lang in lang_dict.values():
                    # Find the language name for the detected language
                    lang_name = [k for k, v in lang_dict.items() if v == st.session_state.detected_lang]
                    if lang_name:
                        st.session_state.target_lang = lang_name[0]
                        st.session_state.input_text = st.session_state.translated_text
                        st.session_state.show_translation = False
                        st.rerun()
        
        with col3:
            word_count = len(st.session_state.translated_text.split())
            st.metric("Word Count", word_count)
        
        with col4:
            char_count = len(st.session_state.translated_text)
            st.metric("Character Count", char_count)
    
    # Translation history
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📚 Translation History")
        
        for i, item in enumerate(st.session_state.history[:5]):  # Show only last 5 items
            with st.expander(f"{item['timestamp']} - {LANGUAGES.get(item['source_lang'], 'Unknown')} → {LANGUAGES.get(item['target_lang'], 'Unknown')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Text**")
                    st.text(item['source_text'][:200] + ("..." if len(item['source_text']) > 200 else ""))
                with col2:
                    st.markdown("**Translation**")
                    st.text(item['translated_text'][:200] + ("..." if len(item['translated_text']) > 200 else ""))
                
                if st.button(f"↩️ Restore #{i+1}", key=f"restore_{i}"):
                    st.session_state.input_text = item['source_text']
                    st.session_state.target_lang = list(lang_dict.keys())[list(lang_dict.values()).index(item['target_lang'])]
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
        <p>✨ Built by Raza Khan</p>
        <p>🌍 Supporting 100+ languages worldwide</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()