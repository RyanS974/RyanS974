import gradio as gr

def create_main_screen():
    """
    Create the main landing screen with app description and navigation
    
    Returns:
        tuple: (welcome_msg, about_info, get_started_btn)
    """
    with gr.Column() as main_screen:
        welcome_msg = gr.Markdown(
            """
            # LLM Response Comparator
            
            ## Analyze and Compare Responses from Different LLMs on Political Topics
            """
        )
        
        about_info = gr.Markdown(
            """
            ### About This Tool
            
            This application allows you to compare how different Large Language Models (LLMs) respond 
            to the same political prompts or questions. Using various NLP techniques, the tool analyzes:
            
            - **Topic Modeling**: What key topics do different LLMs emphasize?
            - **N-gram Analysis**: What phrases and word patterns are characteristic of each LLM?
            - **Bias Detection**: Are there detectable biases in how LLMs approach political topics?
            - **Text Classification**: How do responses cluster or differentiate?
            - **Key Differences**: What specific content varies between models?
            
            ### How to Use
            
            1. Navigate to the **Dataset Input** tab
            2. Enter prompts and corresponding LLM responses, or load an example dataset
            3. Run various analyses to see how the responses compare
            4. Explore visualizations of the differences
            5. Generate a comprehensive report of findings
            
            This tool is for educational and research purposes to better understand how LLMs handle 
            politically sensitive topics.
            """
        )
        
        get_started_btn = gr.Button("Get Started", variant="primary", size="large")
    
    return welcome_msg, about_info, get_started_btn
