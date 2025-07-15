# 🌐 Language Detector & Translator

A Streamlit-based web application that automatically detects the language of input text and provides accurate translations to over 100 languages using the Google Translate API.

![App Screenshot](https://via.placeholder.com/800x500?text=Language+Detector+%26+Translator+Screenshot)

## ✨ Features

- **Automatic Language Detection**: Identifies the source language of any input text
- **Multi-language Translation**: Supports translation to 100+ languages
- **Pronunciation Guide**: Shows pronunciation for translated text when available
- **Translation History**: Keeps track of your recent translations
- **Swap Languages**: Quickly switch between source and target languages
- **Word & Character Count**: Provides statistics for both input and output text
- **Copy to Clipboard**: One-click copy of translated text
- **Responsive Design**: Works well on both desktop and mobile devices

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Razakhan143/language-translator.git
   cd language-translator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install them manually:
   ```bash
   pip install streamlit langdetect googletrans==4.0.0-rc1 pyperclip
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📋 Requirements

- Python 3.7+
- Streamlit
- langdetect
- googletrans (version 4.0.0-rc1)
- pyperclip

## 🚀 Usage

1. Enter your text in the input box
2. Select your target language from the dropdown
3. Click "Translate Now"
4. View your translation with pronunciation (if available)
5. Use additional features like:
   - Copy to clipboard
   - Swap languages
   - View translation history

## 🌍 Supported Languages

The application supports all languages available in the Google Translate API, including:

- English
- Spanish
- French
- German
- Chinese
- Japanese
- Russian
- Arabic
- And 100+ more...

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Translate API for language detection and translation
- Streamlit for the web application framework
- The open source community for various supporting libraries

