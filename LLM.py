import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="pt")


# Define enhanced Streamlit app
def main():
    # Improved page configuration
    st.set_page_config(
        page_title="Language Model Deployment",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    st.markdown(
        """
        <style>
            .main {background-color: #f0f2f6;}
            h1 {text-align: center; color: #333;}
            .stTextArea>div>div>textarea {background-color: #fafafa;}
            .stButton>button {border: 2px solid #4CAF50; border-radius: 5px;}
        </style>
        """, unsafe_allow_html=True
    )

    # Header section with improved styling
    st.markdown(
        """
        <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px;">
            <h1 style="color: white;">Streamlit App with a Hugging Face LLM Model</h1>
        </div>
        <br>
        """, unsafe_allow_html=True
    )

    # Introduction section with detailed instructions
    st.markdown(
        """
        <div style="padding: 20px; border-radius: 10px;">
            <p>This app showcases the innovative capabilities of a cutting-edge language model for text generation. Leveraging a model trained on a diverse corpus, it excels in crafting detailed, context-aware responses, ranging from casual conversations to intricate narratives.</p>
            <p>Simply type a prompt into the text box below and press <span style="font-weight: bold;">Generate</span> to unveil the model's creativity. Whether you're looking for a continuation of a story, a simulated dialogue, or an imaginative exploration based on your input, the possibilities are vast. Feel the model's flair by experimenting with various prompts and witness how it weaves narratives, conjures dialogues, or delves into topics with remarkable coherence and inventiveness.</p>
            <p><strong>Note:</strong> The generated text may sometimes venture into unexpected territories or adopt unique narrative styles, reflecting the model's design to simulate a wide range of human-like text based on the given prompt.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # User input for text generation
    text_input = st.text_area("Enter your text here to start generating:", height=150,
                              placeholder="Type your prompt here...")

    if st.button("Generate"):
        if text_input:
            with st.spinner('Generating...'):
                # Generate slightly more text to ensure complete sentences
                generated_texts = text_generator(text_input, max_length=120, do_sample=True)
                generated_text = generated_texts[0]['generated_text']

                # Trim to the last complete sentence
                last_period_index = generated_text.rfind('. ')
                if last_period_index != -1:
                    generated_text = generated_text[:last_period_index + 1]

                st.markdown("### Generated Text:")
                st.write(generated_text)
        else:
            st.error("Please enter some text to generate.")


if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
