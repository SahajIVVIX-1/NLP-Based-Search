import os
import gradio as gr
from src.preprocessor import ensure_nltk_resources
from src.ui.views import AppState, create_search_app

# 1. Ensure required NLTK resources are present
ensure_nltk_resources()

# 2. Initialize application state and load sample corpus
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_docs")
app_state = AppState(sample_dir=sample_dir)

# 3. Create the Gradio app instance
demo = create_search_app(app_state)

if __name__ == "__main__":
    # Launch application
    server_port = int(os.environ.get("PORT", 7860))
    print(f"Starting NLP Document Search Engine on port {server_port}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        show_error=True,
    )
