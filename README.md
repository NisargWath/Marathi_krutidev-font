# Voice to Krutidev Font Converter

## Overview
This is a Streamlit-based web application that converts spoken Hindi/Marathi audio into Krutidev text format. It allows users to either record their voice or upload an audio file, transcribe it into text, and convert it to Krutidev font.

## Features
- **Record Audio**: Record speech directly in the browser.
- **Upload Audio**: Supports `.wav`, `.mp3`, and `.m4a` formats.
- **Speech-to-Text**: Uses Google Speech Recognition for transcribing Hindi and Marathi.
- **Krutidev Conversion**: Converts transcribed Unicode text to Krutidev font.
- **Download Options**: Provides downloads in `.txt` and `.docx` formats.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/NisargWath/voice-to-krutidev.git
   cd voice-to-krutidev
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Open the provided URL in your browser.
3. Choose either to record audio or upload an audio file.
4. Select the language (Hindi/Marathi).
5. Click **Transcribe** to get the text.
6. Convert to Krutidev and download the output.

## Version
**Basic First Stage Version**

