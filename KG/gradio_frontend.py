import gradio as gr
from kg_query import get_chatbot_response  # Your (prompt) -> response function

# Format function for the chatbot
def respond(message, history):
    response = get_chatbot_response(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history

# Set up the chatbot
chat_interface = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(label="Medical KG Assistant", type="messages"),
    title="🧠 Medical Diagnosis Assistant",
    description="Describe your symptoms, lab results, and demographics. The assistant will analyze your input using a medical knowledge graph.",
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    chat_interface.launch()
