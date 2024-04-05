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
            <p>This app demonstrates the capabilities of a state-of-the-art language model for text generation, with a focus on creating a more engaging user interface and improved chatbot performance.</p>
            <p>Enter a prompt in the text box below and click <span style="font-weight: bold;">Generate</span> to see the model's response. Explore different prompts to discover the model's versatility!</p>
        </div>
        """, unsafe_allow_html=True
    )

    # User input for text generation
    text_input = st.text_area("Enter your text here to start generating:", height=150,
                              placeholder="Type your prompt here...")

    if st.button("Generate"):
        if text_input:
            with st.spinner('Generating...'):
                generated_text = text_generator(text_input, max_length=100, do_sample=True)[0]['generated_text']
            st.markdown("### Generated Text:")
            st.write(generated_text)
        else:
            st.error("Please enter some text to generate.")


if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
