import streamlit as st
import cv2
import ollama
import time
import threading
import collections
import contextlib

# --- App Configuration ---
st.set_page_config(page_title="VLM Observer", layout="wide")
st.title("VLM Observer")

# --- VLM Configuration ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llava"

# --- Thread-Safe Frame Buffer ---
FRAME_BUFFER_SIZE = 10  # Store the last 10 frames for potential motion analysis
frame_buffer = collections.deque(maxlen=FRAME_BUFFER_SIZE)
frame_lock = threading.Lock()
stop_event = threading.Event()

def camera_capture_thread():
    """Thread function to continuously capture frames from the camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera.")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                frame_buffer.append(frame)
        else:
            # Brief pause if we fail to read a frame
            time.sleep(0.1)
    cap.release()

@contextlib.contextmanager
def camera_manager():
    """A context manager to handle the camera thread lifecycle."""
    cam_thread = threading.Thread(target=camera_capture_thread)
    cam_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        cam_thread.join()

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
if st.session_state.mode == "idle":
    st.sidebar.info("The application is idle. Choose a mode below.")
    with st.sidebar.expander("👀 Monitoring Mode", expanded=True):
        monitor_input = st.text_input("What should I look for?", key="monitor_input_idle")
        if st.button("Start Monitoring"):
            if monitor_input:
                st.session_state.monitoring_target = monitor_input
                st.session_state.mode = "monitoring"
                st.rerun()

    with st.sidebar.expander("💬 Chat Mode", expanded=True):
        if st.button("Start Chatting"):
            st.session_state.mode = "chat"
            st.session_state.chat_history = []
            st.rerun()

elif st.session_state.mode == "monitoring":
    st.sidebar.info(f"Monitoring for: **{st.session_state.monitoring_target}**")
    if st.sidebar.button("Stop Monitoring"):
        st.session_state.mode = "idle"
        st.rerun()

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

# --- Main App Logic ---
st.header("Live Camera Feed")
video_placeholder = st.empty()

with camera_manager():
    while True:
        with frame_lock:
            if frame_buffer:
                # Get the most recent frame
                frame = frame_buffer[-1]
            else:
                frame = None

        if frame is None:
            video_placeholder.text("Initializing camera...")
            time.sleep(0.5)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # --- Analysis Logic (uses the most recent frame) ---
        if st.session_state.mode == "monitoring":
            current_time = time.time()
            if current_time - st.session_state.last_check_time > MONITORING_INTERVAL:
                st.session_state.last_check_time = current_time
                _, buffer = cv2.imencode('.jpg', frame)
                prompt = f"The user is looking for '{st.session_state.monitoring_target}'. Is this in the image? Answer with only 'yes' or 'no'."
                answer = analyze_image(buffer.tobytes(), prompt)
                if answer and 'yes' in answer.lower():
                    st.toast(f"I see '{st.session_state.monitoring_target}'!", icon="✅")
                    st.session_state.mode = "idle"
                    st.rerun()

        elif st.session_state.mode == "chat":
            if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
                user_question = st.session_state.chat_history[-1][1]
                _, buffer = cv2.imencode('.jpg', frame)
                with st.spinner('Thinking...'):
                    model_response = analyze_image(buffer.tobytes(), user_question)
                if model_response:
                    st.session_state.chat_history.append(("assistant", model_response))
                    st.rerun()

        time.sleep(0.01)