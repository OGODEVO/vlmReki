import cv2
import ollama
import time

# --- Configuration ---
MODEL = "llava"
MONITORING_INTERVAL = 1  # Seconds between checks when monitoring
# ---------------------

def analyze_image(image_bytes, prompt):
    """Sends an image and a prompt to the Ollama model and returns the response."""
    try:
        response = ollama.chat(
            model=MODEL,
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
        print(f"Error communicating with Ollama: {e}")
        print("Is Ollama running? Have you pulled the model with `ollama run llava`?")
        return None

def main():
    """Main function to run the camera observer application."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("--- Initializing System ---")
    print("Performing initial camera check...")

    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive initial frame.")
        cap.release()
        return

    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    initial_prompt = "Describe what you see in this image in one sentence."
    description = analyze_image(image_bytes, initial_prompt)

    if description:
        print(f"Initial check complete. Model sees: {description}")
    else:
        print("Initial check failed. Exiting.")
        cap.release()
        return

    try:
        while True:
            print("\n----------------------------------------------------")
            instruction = input("What should I look for? (or type 'quit' to exit):")
            if instruction.lower() == 'quit':
                break

            print(f"\nMonitoring for: '{instruction}'. I will let you know when I see it.")
            print("(Press Ctrl+C to stop monitoring and enter a new instruction.)")

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Can't receive frame.")
                        break

                    _, buffer = cv2.imencode('.jpg', frame)
                    image_bytes = buffer.tobytes()

                    monitor_prompt = f"The user is looking for '{instruction}'. Is this in the image? Answer with only the word 'yes' or 'no'."
                    answer = analyze_image(image_bytes, monitor_prompt)

                    if answer and 'yes' in answer.lower():
                        print(f"\nFound it! I see '{instruction}'.")
                        break  # Stop monitoring and ask for new instruction
                    
                    # Wait before checking again
                    time.sleep(MONITORING_INTERVAL)

            except KeyboardInterrupt:
                print("\nStopped monitoring.")
                continue # Go back to asking for instructions

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")

    finally:
        cap.release()
        print("Camera released. Exiting.")

if __name__ == "__main__":
    main()
