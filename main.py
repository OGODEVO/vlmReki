import streamlit as st
import cv2
import ollama
import time

# --- App Configuration ---
st.set_page_config(page_title="VLM Observer", layout="wide")
st.title("VLM Observer")

# --- VLM Configuration ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llava"
MONITOR_KEYWORDS = ["monitor for", "look for", "find a", "find the"]

# --- Helper Functions ---
def analyze_image(image_bytes, prompt):
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt, 'images': [image_bytes]}])
        return response['message']['content']
    except Exception as e:
        st.error(f"Ollama Error: {e}")
        return None

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("assistant", "Hi! Ask me a question or tell me to monitor for something.")]
if "monitoring_target" not in st.session_state:
    st.session_state.monitoring_target = None
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0

# --- UI Layout ---
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Live Camera Feed")
    video_placeholder = st.empty()
with col2:
    st.header("Conversation")
    if st.session_state.monitoring_target:
        st.info(f"Monitoring for: **{st.session_state.monitoring_target}**")
        if st.button("Stop Monitoring"):
            st.session_state.monitoring_target = None
            st.session_state.chat_history.append(("assistant", "Okay, I've stopped monitoring."))
            st.rerun()

    chat_container = st.container(height=500)
    with chat_container:
        for author, message in st.session_state.chat_history:
            with st.chat_message(author):
                st.write(message)

# --- Main App Logic ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not open camera. Please check permissions and ensure it's not in use.")
    st.stop()

# Store the prompt from the chat input
if prompt := st.chat_input("Your message..."):
    st.session_state.last_prompt = prompt
    st.rerun()

# Check if there is a new prompt to process
if "last_prompt" in st.session_state and st.session_state.last_prompt:
    user_prompt = st.session_state.last_prompt
    st.session_state.chat_history.append(("user", user_prompt))
    
    is_monitor_command = any(user_prompt.lower().strip().startswith(kw) for kw in MONITOR_KEYWORDS)
    
    if is_monitor_command:
        for kw in MONITOR_KEYWORDS:
            if user_prompt.lower().strip().startswith(kw):
                target = user_prompt[len(kw):].strip()
                break
        st.session_state.monitoring_target = target
        st.session_state.chat_history.append(("assistant", f"Okay, I'm now monitoring for: **{target}**"))
    else:
        st.session_state.monitoring_target = None
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame for analysis.")
        else:
            with st.spinner("Thinking..."):
                _, buffer = cv2.imencode('.jpg', frame)
                model_response = analyze_image(buffer.tobytes(), user_prompt)
                if model_response:
                    st.session_state.chat_history.append(("assistant", model_response))
    
    # Clear the prompt after processing
    st.session_state.last_prompt = None
    st.rerun()

# --- Main Loop for Video and Monitoring ---
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read frame from camera. The feed has been lost.")
        break

    if st.session_state.monitoring_target:
        MONITORING_INTERVAL = st.sidebar.slider("Monitoring Interval (sec)", 1, 10, 2, key="monitor_interval")
        current_time = time.time()
        if current_time - st.session_state.last_check_time > MONITORING_INTERVAL:
            st.session_state.last_check_time = current_time
            _, buffer = cv2.imencode('.jpg', frame)
            monitor_prompt = f"The user is looking for '{st.session_state.monitoring_target}'. Is this in the image? Answer with only 'yes' or 'no'."
            # This analysis runs in the background of the video feed
            answer = analyze_image(buffer.tobytes(), monitor_prompt)
            if answer and 'yes' in answer.lower():
                st.toast(f"I see '{st.session_state.monitoring_target}'!", icon="✅")
                st.session_state.chat_history.append(("assistant", f"I've found it! I see **{st.session_state.monitoring_target}**."))
                st.session_state.monitoring_target = None
                st.rerun()

    # --- Default Video Display ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    time.sleep(0.01)

cap.release()