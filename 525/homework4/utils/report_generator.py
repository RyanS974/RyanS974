import json
import datetime
import os
import markdown

def create_report(analysis_results, classification_results=None):
    """
    Create comprehensive analysis report
    
    Args:
        analysis_results (dict): Results from all analyses
        classification_results (dict): Results from classification
        
    Returns:
        dict: Structured report
    """
    # Start with an empty report
    report = {
        "title": "LLM Response Comparison Report",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sections": []
    }
    
    # Check if we have valid analysis results
    if not analysis_results or "analyses" not in analysis_results:
        report["sections"].append({
            "title": "Error",
            "content": "No analysis results available for report generation."
        })
        return report
    
    # Add executive summary
    executive_summary = create_executive_summary(analysis_results, classification_results)
    report["sections"].append(executive_summary)
    
    # Add sections for each prompt
    for prompt_idx, (prompt, analyses) in enumerate(analysis_results["analyses"].items()):
        prompt_section = {
            "title": f"Prompt {prompt_idx + 1}: '{truncate_text(prompt, 100)}'",
            "content": f"Analysis of LLM responses to: '{prompt}'",
            "subsections": []
        }
        
        # Add topic analysis if available
        if "topic_modeling" in analyses:
            topic_section = create_topic_section(analyses["topic_modeling"])
            prompt_section["subsections"].append(topic_section)
        
        # Add n-gram analysis if available
        if "ngram_analysis" in analyses:
            ngram_section = create_ngram_section(analyses["ngram_analysis"])
            prompt_section["subsections"].append(ngram_section)
        
        # Add bias analysis if available
        if "bias_detection" in analyses:
            bias_section = create_bias_section(analyses["bias_detection"])
            prompt_section["subsections"].append(bias_section)
        
        # Add similarity metrics if available
        if "similarity_metrics" in analyses:
            similarity_section = create_similarity_section(analyses["similarity_metrics"])
            prompt_section["subsections"].append(similarity_section)
        
        # Add bag of words analysis if available
        if "bag_of_words" in analyses:
            bow_section = create_bow_section(analyses["bag_of_words"])
            prompt_section["subsections"].append(bow_section)
        
        # Add difference highlighting if available
        if "difference_highlighting" in analyses:
            diff_section = create_diff_section(analyses["difference_highlighting"])
            prompt_section["subsections"].append(diff_section)
        
        report["sections"].append(prompt_section)
    
    # Add classification results if available
    if classification_results and "type" in classification_results:
        class_section = create_classification_section(classification_results)
        report["sections"].append(class_section)
    
    return report

def create_executive_summary(analysis_results, classification_results=None):
    """
    Create executive summary section
    
    Args:
        analysis_results (dict): Results from all analyses
        classification_results (dict): Results from classification
        
    Returns:
        dict: Executive summary section
    """
    # Count prompts and models
    prompts = analysis_results["analyses"].keys()
    all_models = set()
    
    for prompt in prompts:
        for result_type, results in analysis_results["analyses"][prompt].items():
            if isinstance(results, dict) and "models" in results:
                all_models.update(results["models"])
    
    summary_content = [
        f"This report analyzes responses from {len(all_models)} different LLMs across {len(prompts)} political prompts.",
        "",
        "Key findings include:"
    ]
    
    # Add placeholder findings (these would be generated based on actual analysis in a full implementation)
    summary_content.extend([
        "- Significant differences in topic emphasis between models",
        "- Varying levels of partisan leaning detected in responses",
        "- Distinctive linguistic patterns identified for each model",
        "- Different framing approaches to political topics",
        ""
    ])
    
    return {
        "title": "Executive Summary",
        "content": "\n".join(summary_content)
    }

def create_topic_section(topic_results):
    """
    Create topic analysis section
    
    Args:
        topic_results (dict): Results from topic modeling
        
    Returns:
        dict: Topic analysis section
    """
    content = ["The following topics were identified across LLM responses:"]
    
    # Add topic keywords
    if "topics" in topic_results:
        content.append("\nTopics and their keywords:")
        for topic_id, topic_info in topic_results["topics"].items():
            content.append(f"- **{topic_id}**: {', '.join(topic_info['keywords'])}")
    
    # Add primary topics by model
    if "primary_topics" in topic_results:
        content.append("\nPrimary topic for each model:")
        for model, primary_topic in topic_results["primary_topics"].items():
            content.append(f"- **{model}**: {primary_topic['topic']} (weight: {primary_topic['weight']:.2f})")
    
    # Add topic distribution by model
    if "model_topics" in topic_results:
        content.append("\nTopic distribution for each model:")
        for model, topics in topic_results["model_topics"].items():
            topic_strs = []
            for topic_id, weight in topics.items():
                topic_strs.append(f"{topic_id}: {weight:.2f}")
            content.append(f"- **{model}**: {', '.join(topic_strs)}")
    
    return {
        "title": "Topic Analysis",
        "content": "\n".join(content)
    }

def create_ngram_section(ngram_results):
    """
    Create n-gram analysis section
    
    Args:
        ngram_results (dict): Results from n-gram analysis
        
    Returns:
        dict: N-gram analysis section
    """
    content = [f"Analysis of {ngram_results['n']}-gram patterns in responses:"]
    
    # Add top n-grams by model
    if "top_ngrams" in ngram_results:
        content.append("\nMost common n-grams by model:")
        for model, ngrams in ngram_results["top_ngrams"].items():
            ngram_strs = []
            for ngram_info in ngrams:
                ngram_strs.append(f"\"{ngram_info['ngram']}\" ({ngram_info['count']})")
            content.append(f"- **{model}**: {', '.join(ngram_strs)}")
    
    # Add unique n-grams by model
    if "unique_ngrams" in ngram_results:
        content.append("\nUnique n-grams by model:")
        for model, ngrams in ngram_results["unique_ngrams"].items():
            if not ngrams:
                content.append(f"- **{model}**: No unique n-grams found")
                continue
                
            ngram_strs = []
            for ngram_info in ngrams:
                ngram_strs.append(f"\"{ngram_info['ngram']}\" ({ngram_info['count']})")
            content.append(f"- **{model}**: {', '.join(ngram_strs)}")
    
    # Add similarity between models
    if "similarities" in ngram_results:
        content.append("\nN-gram similarity between models:")
        for pair, similarity in ngram_results["similarities"].items():
            content.append(f"- **{pair}**: Jaccard similarity = {similarity['jaccard_similarity']:.2f}")
    
    return {
        "title": "N-gram Analysis",
        "content": "\n".join(content)
    }

def create_bias_section(bias_results):
    """
    Create bias analysis section
    
    Args:
        bias_results (dict): Results from bias detection
        
    Returns:
        dict: Bias analysis section
    """
    content = ["Analysis of potential biases in responses:"]
    
    # Add sentiment analysis if available
    if "sentiment" in bias_results:
        content.append("\n### Sentiment Analysis")
        content.append("| Model | Compound Score | Classification |")
        content.append("| ----- | -------------- | -------------- |")
        
        for model, sentiment in bias_results["sentiment"].items():
            content.append(f"| {model} | {sentiment['compound']:.2f} | {sentiment['classification']} |")
    
    # Add partisan leaning if available
    if "partisan_leaning" in bias_results:
        content.append("\n### Partisan Leaning")
        content.append("| Model | Lean Score | Liberal Terms | Conservative Terms | Classification |")
        content.append("| ----- | ---------- | ------------- | ------------------ | -------------- |")
        
        for model, partisan in bias_results["partisan_leaning"].items():
            content.append(f"| {model} | {partisan['lean_score']:.2f} | {partisan['liberal_terms']} | {partisan['conservative_terms']} | {partisan['classification']} |")
    
    # Add framing analysis if available
    if "framing" in bias_results:
        content.append("\n### Framing Analysis")
        content.append("| Model | Economic | Moral/Ethical | Security | Primary Frame |")
        content.append("| ----- | -------- | ------------- | -------- | ------------ |")
        
        for model, framing in bias_results["framing"].items():
            content.append(f"| {model} | {framing['economic_framing']:.0%} | {framing['moral_framing']:.0%} | {framing['security_framing']:.0%} | {framing['primary_frame']} |")
    
    return {
        "title": "Bias Analysis",
        "content": "\n".join(content)
    }

def create_similarity_section(similarity_results):
    """
    Create similarity metrics section
    
    Args:
        similarity_results (dict): Results from similarity metrics
        
    Returns:
        dict: Similarity metrics section
    """
    content = ["How similar are the responses between models?"]
    
    content.append("\n| Model Pair | Cosine Similarity | Semantic Similarity |")
    content.append("| ---------- | ----------------- | ------------------- |")
    
    for pair, metrics in similarity_results.items():
        cosine = metrics.get("cosine_similarity", "N/A")
        semantic = metrics.get("semantic_similarity", "N/A")
        
        if isinstance(cosine, float):
            cosine = f"{cosine:.2f}"
        if isinstance(semantic, float):
            semantic = f"{semantic:.2f}"
        
        content.append(f"| {pair} | {cosine} | {semantic} |")
    
    return {
        "title": "Similarity Metrics",
        "content": "\n".join(content)
    }

def create_bow_section(bow_results):
    """
    Create bag of words analysis section
    
    Args:
        bow_results (dict): Results from bag of words analysis
        
    Returns:
        dict: Bag of words section
    """
    content = ["Analysis of word usage patterns:"]
    
    # Add important words by model
    if "important_words" in bow_results:
        content.append("\n### Most important words by model:")
        for model, words in bow_results["important_words"].items():
            word_strs = []
            for word_info in words[:10]:  # Limit to top 10 for readability
                word_strs.append(f"{word_info['word']} ({word_info['count']})")
            content.append(f"- **{model}**: {', '.join(word_strs)}")
    
    # Add differential words
    if "differential_words" in bow_results and "word_count_matrix" in bow_results:
        content.append("\n### Words with biggest usage differences between models:")
        content.append("| Word | " + " | ".join(bow_results["models"]) + " |")
        content.append("| ---- | " + " | ".join(["-" * len(model) for model in bow_results["models"]]) + " |")
        
        for word in bow_results["differential_words"][:10]:  # Limit to top 10
            if word in bow_results["word_count_matrix"]:
                counts = bow_results["word_count_matrix"][word]
                count_strs = [str(counts.get(model, 0)) for model in bow_results["models"]]
                content.append(f"| {word} | {' | '.join(count_strs)} |")
    
    return {
        "title": "Word Usage Analysis",
        "content": "\n".join(content)
    }

def create_diff_section(diff_results):
    """
    Create difference highlighting section
    
    Args:
        diff_results (dict): Results from difference highlighting
        
    Returns:
        dict: Difference highlighting section
    """
    content = ["Analysis of content differences between responses:"]
    
    for pair, diff in diff_results.items():
        content.append(f"\n### {pair}")
        content.append(f"Content overlap: {diff['content_overlap']:.0%}")
        
        # Add significant words
        content.append("\nSignificant words unique to each:")
        if diff["significant_words_first"]:
            content.append(f"- First model: {', '.join(diff['significant_words_first'])}")
        else:
            content.append("- First model: None found")
            
        if diff["significant_words_second"]:
            content.append(f"- Second model: {', '.join(diff['significant_words_second'])}")
        else:
            content.append("- Second model: None found")
        
        # Add sample unique content (limited for readability)
        content.append("\nSample unique content:")
        content.append("- First model:")
        if diff["unique_to_first"]:
            for sentence in diff["unique_to_first"][:3]:  # Limit to 3 examples
                content.append(f"  - \"{sentence}\"")
        else:
            content.append("  - No unique content found")
            
        content.append("- Second model:")
        if diff["unique_to_second"]:
            for sentence in diff["unique_to_second"][:3]:  # Limit to 3 examples
                content.append(f"  - \"{sentence}\"")
        else:
            content.append("  - No unique content found")
    
    return {
        "title": "Content Differences",
        "content": "\n".join(content)
    }

def create_classification_section(classification_results):
    """
    Create classification results section
    
    Args:
        classification_results (dict): Results from classification
        
    Returns:
        dict: Classification section
    """
    content = [f"Results from {classification_results['type']}:"]
    
    if "results" in classification_results:
        content.append("\n| Model | Predicted Class | Confidence |")
        content.append("| ----- | --------------- | ---------- |")
        
        for result in classification_results["results"]:
            confidence = result["confidence"]
            if isinstance(confidence, float):
                confidence = f"{confidence:.0%}"
            content.append(f"| {result['model']} | {result['class']} | {confidence} |")
    
    if "metrics" in classification_results:
        metrics = classification_results["metrics"]
        content.append("\n### Classification Performance")
        content.append(f"- Algorithm: {metrics.get('algorithm', 'N/A')}")
        content.append(f"- Feature type: {metrics.get('feature_type', 'N/A')}")
        
        if "accuracy" in metrics:
            content.append(f"- Accuracy: {metrics['accuracy']:.2f}")
        if "precision" in metrics:
            content.append(f"- Precision: {metrics['precision']:.2f}")
        if "recall" in metrics:
            content.append(f"- Recall: {metrics['recall']:.2f}")
        if "f1_score" in metrics:
            content.append(f"- F1 Score: {metrics['f1_score']:.2f}")
        
        if "class_distribution" in metrics:
            content.append("\n### Class Distribution")
            for cls, count in metrics["class_distribution"].items():
                content.append(f"- {cls}: {count}")
    
    return {
        "title": "Classification Results",
        "content": "\n".join(content)
    }

def format_for_display(report, format="markdown"):
    """
    Format report for Gradio display
    
    Args:
        report (dict): Report data
        format (str): Output format ("markdown" or "html")
        
    Returns:
        str: Formatted report
    """
    if format.lower() == "html":
        return convert_to_html(report)
    else:
        return convert_to_markdown(report)

def convert_to_markdown(report):
    """
    Convert report to Markdown format
    
    Args:
        report (dict): Report data
        
    Returns:
        str: Markdown formatted report
    """
    md_content = [f"# {report['title']}", f"*Generated on {report['generated_at']}*", ""]
    
    for section in report["sections"]:
        md_content.append(f"## {section['title']}")
        md_content.append(section["content"])
        
        if "subsections" in section:
            for subsection in section["subsections"]:
                md_content.append(f"### {subsection['title']}")
                md_content.append(subsection["content"])
                md_content.append("")
        
        md_content.append("")
    
    return "\n".join(md_content)

def convert_to_html(report):
    """
    Convert report to HTML format
    
    Args:
        report (dict): Report data
        
    Returns:
        str: HTML formatted report
    """
    # Convert Markdown to HTML
    md_content = convert_to_markdown(report)
    html_content = markdown.markdown(md_content, extensions=['tables'])
    
    # Add basic styling
    styled_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        {html_content}
    </div>
    """
    
    return styled_html

def export_report(report, format="md"):
    """
    Export report to file in specified format
    
    Args:
        report (dict): Report data
        format (str): Output format ("md", "html", "pdf")
        
    Returns:
        str: Path to exported file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "md" or format == "markdown":
        content = convert_to_markdown(report)
        filename = f"reports/llm_comparison_report_{timestamp}.md"
    elif format == "html":
        content = convert_to_html(report)
        filename = f"reports/llm_comparison_report_{timestamp}.html"
    elif format == "pdf":
        # In a real implementation, this would convert to PDF
        # For now, just create markdown with a note
        content = convert_to_markdown(report) + "\n\n*PDF conversion not implemented in this demo*"
        filename = f"reports/llm_comparison_report_{timestamp}.md"
    else:
        return "Unsupported format"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filename

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
    return text[:max_length-3] + "..."
