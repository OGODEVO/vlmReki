import streamlit as st
import cv2
import ollama
import time

# --- App Configuration ---
st.set_page_config(page_title="VLM Observer", layout="wide")
st.title("VLM Observer")

# --- VLM Configuration ---
# SET YOUR OLLAMA ENDPOINT AND MODEL HERE
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llava"
# -------------------------

# --- Helper Functions ---
def analyze_image(image_bytes, prompt):
    """Sends an image and a prompt to the Ollama model and returns the response."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        st.sidebar.error(f"Ollama Error: {e}")
        st.sidebar.warning(f"Ensure Ollama is running and the host ({OLLAMA_HOST}) is correct.", icon="🤖")
        return None

# --- Session State Initialization ---
if 'mode' not in st.session_state:
    st.session_state.mode = "idle"
if 'monitoring_target' not in st.session_state:
    st.session_state.monitoring_target = ""
if 'last_check_time' not in st.session_state:
    st.session_state.last_check_time = 0
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar UI ---
st.sidebar.header("Configuration")
MONITORING_INTERVAL = st.sidebar.slider("Monitoring Interval (sec)", 1, 10, 2)

st.sidebar.header("Controls")
# --- Mode: Idle ---
if st.session_state.mode == "idle":
    st.sidebar.info("The application is idle. Choose a mode below.")
    
    with st.sidebar.expander("👀 Monitoring Mode", expanded=True):
        monitor_input = st.text_input("What should I look for?", key="monitor_input_idle")
        if st.button("Start Monitoring"):
            if monitor_input:
                st.session_state.monitoring_target = monitor_input
                st.session_state.mode = "monitoring"
                st.rerun()
            else:
                st.warning("Please enter something to look for.")

    with st.sidebar.expander("💬 Chat Mode", expanded=True):
        if st.button("Start Chatting"):
            st.session_state.mode = "chat"
            st.session_state.chat_history = []
            st.rerun()

# --- Mode: Monitoring ---
elif st.session_state.mode == "monitoring":
    st.sidebar.info(f"Monitoring for: **{st.session_state.monitoring_target}**")
    if st.sidebar.button("Stop Monitoring"):
        st.session_state.mode = "idle"
        st.session_state.monitoring_target = ""
        st.rerun()

# --- Mode: Chat ---
elif st.session_state.mode == "chat":
    st.sidebar.info("You are in chat mode.")
    if st.sidebar.button("End Chat"):
        st.session_state.mode = "idle"
        st.rerun()
    
    st.header("Chat")
    chat_container = st.container(height=400)
    with chat_container:
        for author, message in st.session_state.chat_history:
            with st.chat_message(author):
                st.write(message)
    
    chat_input = st.chat_input("Ask a question...")
    if chat_input:
        st.session_state.chat_history.append(("user", chat_input))
        st.rerun()

# --- Main Video Feed ---
st.header("Live Camera Feed")
video_placeholder = st.empty()

# --- Camera and Core Logic Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not open camera. Please check permissions.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # --- Monitoring Logic ---
        if st.session_state.mode == "monitoring":
            current_time = time.time()
            if current_time - st.session_state.last_check_time > MONITORING_INTERVAL:
                st.session_state.last_check_time = current_time
                
                _, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()
                
                prompt = f"The user is looking for '{st.session_state.monitoring_target}'. Is this in the image? Answer with only the word 'yes' or 'no'."
                answer = analyze_image(image_bytes, prompt)

                if answer and 'yes' in answer.lower():
                    st.toast(f"I see '{st.session_state.monitoring_target}'!", icon="✅")
                    st.session_state.mode = "idle"
                    st.session_state.monitoring_target = ""
                    st.rerun()

        # --- Chat Logic ---
        elif st.session_state.mode == "chat":
            if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
                user_question = st.session_state.chat_history[-1][1]
                
                _, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()

                with st.spinner('Thinking...'):
                    model_response = analyze_image(image_bytes, user_question)
                
                if model_response:
                    st.session_state.chat_history.append(("assistant", model_response))
                    st.rerun()

        time.sleep(0.01)

cap.release()