def update_report(report_data, options):
    """
    Update the report based on analysis results and options
    
    Args:
        report_data (dict): Comprehensive report data from all analyses
        options (dict): Report generation options
        
    Returns:
        tuple: (report_state, updated_report_display)
    """
    # Check if report data is valid
    if not report_data or "analyses" not in report_data:
        error_report = {
            "error": "No analysis results available for report generation"
        }
        return error_report, "### Error\nNo analysis results available for report generation."
    
    # Get selected sections
    include_sections = options.get("include_sections", [])
    report_format = options.get("report_format", "Markdown")
    
    # Generate report content
    report_content = []
    
    # Title and timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content.append(f"# LLM Response Comparison Report")
    report_content.append(f"*Generated on {timestamp}*")
    report_content.append("")
    
    # Executive Summary
    if "Executive Summary" in include_sections:
        report_content.append("## Executive Summary")
        
        # Count the prompts and models
        prompts = report_data["analyses"].keys()
        all_models = set()
        for prompt in prompts:
            for result_type, results in report_data["analyses"][prompt].items():
                if isinstance(results, dict) and "models" in results:
                    all_models.update(results["models"])
        
        report_content.append(f"This report analyzes responses from {len(all_models)} different LLMs across {len(prompts)} political prompts.")
        report_content.append("Key findings include:")
        
        # Simulate some findings
        report_content.append("- Significant differences in topic emphasis between models")
        report_content.append("- Varying levels of partisan leaning detected")
        report_content.append("- Distinct linguistic patterns in response structure")
        report_content.append("")
    
    # For each prompt, add relevant sections
    for prompt_idx, (prompt, analyses) in enumerate(report_data["analyses"].items()):
        report_content.append(f"## Prompt {prompt_idx + 1}: \"{truncate_text(prompt, 100)}\"")
        report_content.append("")
        
        # Topic Analysis
        if "Topic Analysis" in include_sections and "topic_modeling" in analyses:
            report_content.append("### Topic Analysis")
            report_content.append("The following topics were identified across LLM responses:")
            
            # Simulated topic data
            topics = ["Economy", "Foreign Policy", "Social Issues", "Healthcare"]
            report_content.append("| Model | Primary Topics |")
            report_content.append("| ----- | ------------- |")
            
            # Generate simulated model-topic data
            model_topics = {
                "GPT-4": "Economy (45%), Social Issues (30%)",
                "Claude-3": "Social Issues (40%), Healthcare (35%)",
                "Llama-3": "Foreign Policy (50%), Economy (25%)"
            }
            
            for model, topic_dist in model_topics.items():
                report_content.append(f"| {model} | {topic_dist} |")
            
            report_content.append("")
        
        # N-gram Analysis
        if "N-gram Analysis" in include_sections and "ngram_analysis" in analyses:
            report_content.append("### N-gram Analysis")
            report_content.append("Most distinctive phrases by model:")
            
            # Simulated n-gram data
            ngram_data = {
                "GPT-4": ["economic growth", "policy implications", "research suggests"],
                "Claude-3": ["on the other hand", "it's important to note", "multiple perspectives"],
                "Llama-3": ["statistical evidence", "historical context", "according to data"]
            }
            
            for model, ngrams in ngram_data.items():
                report_content.append(f"**{model}**: {', '.join(ngrams)}")
            
            report_content.append("")
        
        # Bias Analysis
        if "Bias Analysis" in include_sections and "bias_detection" in analyses:
            report_content.append("### Bias Analysis")
            
            # Simulated bias data
            report_content.append("#### Partisan Leaning")
            report_content.append("| Model | Leaning Score | Classification |")
            report_content.append("| ----- | ------------- | -------------- |")
            report_content.append("| GPT-4 | 0.02 | Neutral |")
            report_content.append("| Claude-3 | -0.15 | Slight Liberal Lean |")
            report_content.append("| Llama-3 | 0.08 | Slight Conservative Lean |")
            
            report_content.append("")
            report_content.append("#### Sentiment Analysis")
            report_content.append("| Model | Positive | Neutral | Negative |")
            report_content.append("| ----- | -------- | ------- | -------- |")
            report_content.append("| GPT-4 | 25% | 65% | 10% |")
            report_content.append("| Claude-3 | 30% | 60% | 10% |")
            report_content.append("| Llama-3 | 20% | 70% | 10% |")
            
            report_content.append("")
        
        # Similarity Metrics
        if "Similarity Metrics" in include_sections and "similarity_metrics" in analyses:
            report_content.append("### Similarity Metrics")
            report_content.append("How similar are the responses between models?")
            
            # Simulated similarity data
            report_content.append("| Model Pair | Cosine Similarity | Semantic Similarity |")
            report_content.append("| ---------- | ----------------- | ------------------- |")
            report_content.append("| GPT-4 vs Claude-3 | 0.78 | 0.85 |")
            report_content.append("| GPT-4 vs Llama-3 | 0.72 | 0.79 |")
            report_content.append("| Claude-3 vs Llama-3 | 0.76 | 0.82 |")
            
            report_content.append("")
    
    # Classification Results
    if "Classification Results" in include_sections:
        report_content.append("## Classification Results")
        
        # Simulated classification results
        report_content.append("### Response Style Classification")
        report_content.append("| Model | Predicted Style | Confidence |")
        report_content.append("| ----- | --------------- | ---------- |")
        report_content.append("| GPT-4 | Balanced | 85% |")
        report_content.append("| Claude-3 | Factual | 78% |")
        report_content.append("| Llama-3 | Balanced | 62% |")
        
        report_content.append("")
    
    # Key Differences
    if "Key Differences" in include_sections:
        report_content.append("## Key Differences Between Models")
        
        # Simulated key differences
        report_content.append("### Content Emphasis")
        report_content.append("- **GPT-4**: Tends to emphasize economic implications and policy considerations")
        report_content.append("- **Claude-3**: Places more emphasis on presenting multiple perspectives and ethical considerations")
        report_content.append("- **Llama-3**: Focuses more on historical context and statistical evidence")
        report_content.append("")
        
        report_content.append("### Response Structure")
        report_content.append("- **GPT-4**: Typically provides more structured responses with clear sections")
        report_content.append("- **Claude-3**: Often includes explicit acknowledgment of different viewpoints")
        report_content.append("- **Llama-3**: Generally more concise with a higher density of factual statements")
        report_content.append("")
    
    # Combine all content
    full_report = "\n".join(report_content)
    
    # Return the report
    return {"content": full_report, "format": report_format}, full_report

def update_with_llm_analysis(report, llm_analysis):
    """
    Update report with LLM meta-analysis
    
    Args:
        report (dict): The current report
        llm_analysis (str): Meta-analysis from LLM
        
    Returns:
        tuple: (updated_report, updated_display)
    """
    if not report or "content" not in report:
        return report, "Error: No report to analyze"
    
    # Add the LLM analysis to the report
    report_content = report["content"]
    
    # Add a section for LLM meta-analysis
    llm_section = [
        "## LLM Meta-Analysis",
        "",
        "*The following analysis was generated by an LLM based on the report findings:*",
        "",
        llm_analysis,
        ""
    ]
    
    # Combine the original report with the LLM analysis
    updated_content = report_content + "\n" + "\n".join(llm_section)
    updated_report = {
        "content": updated_content,
        "format": report.get("format", "Markdown"),
        "llm_analysis": llm_analysis
    }
    
    return updated_report, updated_content

def truncate_text(text, max_length=100):
    """
    Truncate text to specified length with ellipsis
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."import gradio as gr
import json
import datetime

def create_report_screen():
    """
    Create the report screen interface
    
    Returns:
        tuple: (report_options, generate_report_btn, llm_analysis_btn, export_btn, report_output)
    """
    with gr.Column() as report_screen:
        gr.Markdown("## Analysis Report")
        gr.Markdown("Generate a comprehensive report of all analyses and visualizations.")
        
        # Report options
        with gr.Group() as report_options:
            include_sections = gr.CheckboxGroup(
                choices=[
                    "Executive Summary",
                    "Topic Analysis",
                    "N-gram Analysis",
                    "Bias Analysis",
                    "Similarity Metrics",
                    "Classification Results",
                    "Key Differences",
                    "Visualizations"
                ],
                value=[
                    "Executive Summary",
                    "Topic Analysis",
                    "Bias Analysis",
                    "Classification Results",
                    "Key Differences"
                ],
                label="Include in Report"
            )
            
            report_format = gr.Radio(
                choices=["Markdown", "HTML"],
                value="Markdown",
                label="Report Format"
            )
        
        # Action buttons
        with gr.Row():
            generate_report_btn = gr.Button("Generate Report", variant="primary")
            llm_analysis_btn = gr.Button("Add LLM Meta-Analysis", variant="secondary")
            export_btn = gr.Button("Export Report", variant="secondary")
        
        # Report output
        with gr.Group() as report_output:
            report_markdown = gr.Markdown()
    
    return report_options, generate_report_btn, llm_analysis_btn, export_btn, report_output

def update_report(report_data, options):
    """
    Update the report based on analysis results and options
    
    Args:
        report_data (dict): Comprehensive report data from all analyses
        options (dict): Report generation options
        
    Returns:
        tuple: (report_state, updated_report_display)
    """
    # Check if report data is valid
    if not report_data or "analyses" not in report_data:
        error_report = {
            "error": "No analysis results available for report generation"
        }
        return error_report, "### Error\nNo analysis results available for report generation."
    
    # Get selected sections
    include_sections = options.get("include_sections", [])
    report_format = options.get("report_format", "Markdown")
    
    # Generate report content
    report_content = []
    
    # Title and timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content.append(f"# LLM Response Comparison Report")
    report_content.append(f"*Generated on {timestamp}*")
    report_content.append("")
    
    # Executive Summary
    if "Executive Summary" in include_sections:
        report_content.append("## Executive Summary")
        
        # Count the prompts and models
        prompts = report_data["analyses"].keys()
        all_models = set()
        for prompt in prompts:
            for result_type, results in report_data["analyses"][prompt].items():
                if isinstance(results, dict) and "models" in results:
                    all_models.update(results["models"])
        
        report_content.append(f"This report analyzes responses from {len(all_models)} different LLMs across {len(prompts)} political prompts.")
        report_content.append("Key findings include:")
        
        # Simulate some findings
        report_content.append("- Significant differences in topic emphasis between models")
        report_content.append("- Varying levels of partisan leaning detected")
        report_content.append("- Distinct linguistic patterns in response structure")
        report_content.append("")
    
    # For each prompt, add relevant sections
    for prompt_idx, (prompt, analyses) in enumerate(report_data["analyses"].items()):
        report_content.append(f"## Prompt {prompt_idx + 1}: \"{truncate_text(prompt, 100)}\"")
        report_content.append("")
        
        # Topic Analysis
        if "Topic Analysis" in include_sections and "topic_modeling" in analyses:
            report_content.append("### Topic Analysis")
            report_content.append("The following topics were identified across LLM responses:")
            
            # Simulated topic data
            topics = ["Economy", "Foreign Policy", "Social Issues", "Healthcare"]
            report_content.append("| Model | Primary Topics |")
            report_content.append("| ----- | ------------- |")
            
            # Generate simulated model-topic data
            model_topics = {
                "GPT-4": "Economy (45%), Social Issues (30%)",
                "Claude-3": "Social Issues (40%), Healthcare (35%)",
                "Llama-3": "Foreign Policy (50%), Economy (25%)"
            }
            
            for model, topic_dist in model_topics.items():
                report_content.append(f"| {model} | {topic_dist} |")
            
            report_content.append("")
        
        # N-gram Analysis
        if "N-gram Analysis" in include_sections and "ngram_analysis" in analyses:
            report_content.append("### N-gram Analysis")
            report_content.append("Most distinctive phrases by model:")
            
            # Simulated n-gram data
            ngram_data = {
                "GPT-4": ["economic growth", "policy implications", "research suggests"],
                "Claude-3": ["on the other hand", "it's important to note", "multiple perspectives"],
                "Llama-3": ["statistical evidence", "historical context", "according to data"]
            }
            
            for model, ngrams in ngram_data.items():
                report_content.append(f"**{model}**: {', '.join(ngrams)}")
            
            report_content.append("")
        
        # Bias Analysis
        if "Bias Analysis" in include_sections and "bias_detection" in analyses:
            report_content.append("### Bias Analysis")
            
            # Simulated bias data
            report_content.append("#### Partisan Leaning")
            report_content.append("| Model | Leaning Score | Classification |")
            report_content.append("| ----- | ------------- | -------------- |")
            report_content.append("| GPT-4 | 0.02 | Neutral |")
            report_content.append("| Claude-3 | -0.15 | Slight Liberal Lean |")
            report_content.append("| Llama-3 | 0.08 | Slight Conservative Lean |")
            
            report_content.append("")
            report_content.append("#### Sentiment Analysis")
            report_content.append("| Model | Positive | Neutral | Negative |")
            report_content.append("| ----- | -------- | ------- | -------- |")
            report_content.append("| GPT-4 | 25% | 65% | 10% |")
            report_content.append("| Claude-3 | 30% | 60% | 10% |")
            report_content.append("| Llama-3 | 20% | 70% | 10% |")
            
            report_content.append("")
        
        # Similarity Metrics
        if "Similarity Metrics" in include_sections and "similarity_metrics" in analyses:
            report_content.append("### Similarity Metrics")
            report_content.append("How similar are the responses between models?")
            
            # Simulated similarity data
            report_content.append("| Model Pair | Cosine Similarity | Semantic Similarity |")
            report_content.append("| ---------- | ----------------- | ------------------- |")
            report_content.append("| GPT-4 vs Claude-3 | 0.78 | 0.85 |")
            report_content.append("| GPT-4 vs Llama-3 | 0.72 | 0.79 |")
            report_content.append("| Claude-3 vs Llama-3 | 0.76 | 0.82 |")
            
            report_content.append("")
    
    # Classification Results
    if "Classification Results" in include_sections:
        report_content.append("## Classification Results")
        
        # Simulated classification results
        report_content.append("### Response Style Classification")
        report_content.append("| Model | Predicted Style | Confidence |")
        report_content.append("| ----- | --------------- | ---------- |")
        report_content.append("| GPT-4 | Balanced | 85% |")
        report_content.append("| Claude-3 | Factual | 78% |")
        report_content.append("| Llama-3 | Balanced | 62% |")
        
        report_content.append("")
    
    # Key Differences
    if "Key Differences" in include_sections:
        report_content.append("## Key Differences Between Models")
        
        # Simulated key differences
        report_content.append("### Content Emphasis")
        report_content.appen