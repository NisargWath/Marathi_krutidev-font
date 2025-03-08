import streamlit as st
import speech_recognition as sr
import tempfile
import os
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from docx import Document
import base64
import io
from audio_recorder_streamlit import audio_recorder
import time

# Function to convert Unicode text to Krutidev
def unicode_to_krutidev(unicode_text):
    # This is a simplified conversion - in a real application,
    # you would use a more comprehensive mapping for Krutidev font
    # Here we're using the sanscript library for Devanagari to convert to a close approximation
    # Note: For actual Krutidev encoding, you might need a specialized converter
    
    # First, ensure text is in Unicode Devanagari
    try:
        # For Krutidev-like representation, we can use ITRANS as an approximation
        # A complete solution would require a specific Krutidev converter
        return transliterate(unicode_text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception as e:
        return f"Conversion error: {str(e)}"

# Function to recognize speech from audio file
def recognize_speech(audio_file, language='mr-IN'):
    r = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        try:
            # Use Google's speech recognition service
            # For Hindi use 'hi-IN', for Marathi use 'mr-IN'
            text = r.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error with the speech recognition service: {e}"
        except Exception as e:
            return f"Error: {str(e)}"

# Function to create a download link for a Word document
def create_download_link_docx(text, filename):
    # Create a Word document
    doc = Document()
    doc.add_paragraph(text)
    
    # Save document to a BytesIO object
    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    
    # Create a download link
    b64 = base64.b64encode(doc_io.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{filename}">Download Word document</a>'

def main():
    st.title("Voice to Krutidev Font Converter")
    
    st.write("""
    ## Convert voice to Krutidev text
    Record your voice in Hindi or Marathi or upload an audio file, and this app will convert it to text and display it in Krutidev font.
    """)
    
    # Language selection
    language = st.radio(
        "Select Language:",
        ("Marathi", "Hindi")
    )
    
    language_code = "mr-IN" if language == "Marathi" else "hi-IN"
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
    
    with tab1:
        st.write("Record your voice directly in the browser:")
        
        # Using audio_recorder_streamlit component for recording
        audio_bytes = audio_recorder()
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            # Save the recorded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                recorded_file_path = tmp_file.name
            
            if st.button("Transcribe Recorded Audio"):
                with st.spinner("Processing your recording..."):
                    # Recognize speech
                    transcribed_text = recognize_speech(recorded_file_path, language_code)
                    
                    if not transcribed_text.startswith("Error") and not transcribed_text.startswith("Could not"):
                        # Show the transcribed text
                        st.subheader("Transcribed Text:")
                        st.write(transcribed_text)
                        
                        # Convert to Krutidev and display
                        krutidev_text = unicode_to_krutidev(transcribed_text)
                        st.subheader("Text in Krutidev Format:")
                        st.text(krutidev_text)
                        
                        # Provide download options
                        st.download_button(
                            label="Download as Text File",
                            data=krutidev_text,
                            file_name="krutidev_text.txt",
                            mime="text/plain"
                        )
                        
                        # Create Word document download link
                        st.markdown(
                            create_download_link_docx(krutidev_text, "krutidev_text.docx"),
                            unsafe_allow_html=True
                        )
                    else:
                        st.error(transcribed_text)
                
                # Clean up the temporary file
                os.unlink(recorded_file_path)
    
    with tab2:
        # File uploader for audio
        uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_filepath = tmp_file.name
            
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("Transcribe Uploaded Audio"):
                with st.spinner("Processing audio..."):
                    # Recognize speech
                    transcribed_text = recognize_speech(tmp_filepath, language_code)
                    
                    if not transcribed_text.startswith("Error") and not transcribed_text.startswith("Could not"):
                        # Show the transcribed text
                        st.subheader("Transcribed Text:")
                        st.write(transcribed_text)
                        
                        # Convert to Krutidev and display
                        krutidev_text = unicode_to_krutidev(transcribed_text)
                        st.subheader("Text in Krutidev Format:")
                        st.text(krutidev_text)
                        
                        # Provide download options
                        st.download_button(
                            label="Download as Text File",
                            data=krutidev_text,
                            file_name="krutidev_text.txt",
                            mime="text/plain"
                        )
                        
                        # Create Word document download link
                        st.markdown(
                            create_download_link_docx(krutidev_text, "krutidev_text.docx"),
                            unsafe_allow_html=True
                        )
                    else:
                        st.error(transcribed_text)
                
                # Clean up the temporary file
                os.unlink(tmp_filepath)

if __name__ == "__main__":
    main()