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
MONITOR_KEYWORDS = ["monitor for", "look for", "find a", "find the"]

# --- Thread-Safe Frame Buffer ---
frame_buffer = collections.deque(maxlen=1)
frame_lock = threading.Lock()
stop_event = threading.Event()

def camera_capture_thread():
    """Thread function to continuously capture frames from the camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Cannot use st.error here as this is a background thread
        print("Error: Could not open camera.")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                frame_buffer.append(frame)
        else:
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
        response = client.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt, 'images': [image_bytes]}])
        return response['message']['content']
    except Exception as e:
        st.sidebar.error(f"Ollama Error: {e}")
        return None

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("assistant", "Hi! Ask me a question about the camera feed, or tell me to monitor for something (e.g., 'monitor for a red bottle').")]
if "monitoring_target" not in st.session_state:
    st.session_state.monitoring_target = None
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0
if "processing_input" not in st.session_state:
    st.session_state.processing_input = False

# --- UI Layout ---
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.header("Conversation")
    # Monitoring status UI
    if st.session_state.monitoring_target:
        st.info(f"Monitoring for: **{st.session_state.monitoring_target}**")
        if st.button("Stop Monitoring"):
            st.session_state.monitoring_target = None
            st.session_state.chat_history.append(("assistant", "Okay, I've stopped monitoring."))
            st.rerun()
    
    # Chat history display
    chat_container = st.container(height=500)
    with chat_container:
        for author, message in st.session_state.chat_history:
            with st.chat_message(author):
                st.write(message)

# --- Main App Logic ---
with camera_manager():
    # Handle new user input from the chat box
    if prompt := st.chat_input("Your message..."):
        st.session_state.chat_history.append(("user", prompt))
        st.session_state.processing_input = True
        st.rerun()

    # Process the latest user input if needed
    if st.session_state.processing_input:
        user_prompt = st.session_state.chat_history[-1][1]
        is_monitor_command = any(user_prompt.lower().strip().startswith(kw) for kw in MONITOR_KEYWORDS)

        if is_monitor_command:
            # Extract the target from the command
            for kw in MONITOR_KEYWORDS:
                if user_prompt.lower().strip().startswith(kw):
                    target = user_prompt[len(kw):].strip()
                    break
            st.session_state.monitoring_target = target
            st.session_state.chat_history.append(("assistant", f"Okay, I'm now monitoring for: **{target}**"))
        else:
            # It's a regular chat question, get the current frame and analyze
            with frame_lock:
                if frame_buffer:
                    frame = frame_buffer[-1]
                    _, buffer = cv2.imencode('.jpg', frame)
                    with st.spinner("Thinking..."):
                        model_response = analyze_image(buffer.tobytes(), user_prompt)
                    if model_response:
                        st.session_state.chat_history.append(("assistant", model_response))
                else:
                    st.session_state.chat_history.append(("assistant", "Sorry, the camera feed is not available right now."))
        
        st.session_state.processing_input = False
        st.rerun()

    # Main loop for video display and monitoring
    while True:
        with frame_lock:
            if frame_buffer:
                frame = frame_buffer[-1]
            else:
                frame = None

        if frame is None:
            video_placeholder.text("Initializing camera...")
            time.sleep(0.5)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # --- Monitoring Logic ---
        if st.session_state.monitoring_target:
            MONITORING_INTERVAL = st.sidebar.slider("Monitoring Interval (sec)", 1, 10, 2, key="monitor_interval")
            current_time = time.time()
            if current_time - st.session_state.last_check_time > MONITORING_INTERVAL:
                st.session_state.last_check_time = current_time
                _, buffer = cv2.imencode('.jpg', frame)
                prompt = f"The user is looking for '{st.session_state.monitoring_target}'. Is this in the image? Answer with only 'yes' or 'no'."
                answer = analyze_image(buffer.tobytes(), prompt)
                if answer and 'yes' in answer.lower():
                    st.toast(f"I see '{st.session_state.monitoring_target}'!", icon="✅")
                    st.session_state.chat_history.append(("assistant", f"I've found it! I see **{st.session_state.monitoring_target}**."))
                    st.session_state.monitoring_target = None
                    st.rerun()
        
        time.sleep(0.01)