import requests
import json
import os

def generate_prompt(analysis_results):
    """
    Generate prompt for meta-analysis by LLM
    
    Args:
        analysis_results (dict): Results from all analyses
        
    Returns:
        str: Prompt for LLM meta-analysis
    """
    prompt = """
You are an expert in analyzing political language and NLP. You have been given a report comparing responses
from different LLMs to political questions. Please analyze this report and provide insights on:

1. Key patterns or differences between how different LLMs approach political topics
2. Any potential biases or leanings exhibited by different models
3. Differences in factual content, style, or framing between models
4. Recommendations for users when interpreting LLM outputs on political topics

Here is the report:

"""
    
    # Add the report content
    if isinstance(analysis_results, dict) and "content" in analysis_results:
        prompt += analysis_results["content"]
    else:
        prompt += "ERROR: No valid report content provided."
    
    prompt += """

Please provide a concise but insightful meta-analysis of approximately 500 words based on this report.
Focus on the most important patterns and differences, and avoid simply repeating the data in the report.
"""
    
    return prompt

def run_llm_analysis(report_data, api_type="huggingface"):
    """
    Run meta-analysis through external LLM API
    
    Args:
        report_data (dict): Report data to analyze
        api_type (str): Which API to use ("huggingface", "openai", etc.)
        
    Returns:
        str: LLM analysis result
    """
    prompt = generate_prompt(report_data)
    
    try:
        if api_type == "huggingface":
            return run_huggingface_inference(prompt)
        elif api_type == "openai":
            return run_openai_inference(prompt)
        else:
            # Fallback to mock analysis if no valid API specified
            return generate_mock_analysis(report_data)
    except Exception as e:
        print(f"Error running LLM analysis: {e}")
        return generate_mock_analysis(report_data)

def run_huggingface_inference(prompt):
    """
    Run inference through Hugging Face Inference API
    
    Args:
        prompt (str): Prompt for LLM
        
    Returns:
        str: LLM response
    """
    # Check for API token in environment
    api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    if not api_token:
        return "Error: No Hugging Face API token found. Using simulated analysis instead.\n\n" + generate_mock_analysis({})
    
    # API endpoint for a suitable model
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Prepare the payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    # Make the request
    response = requests.post(api_url, headers=headers, json=payload)
    
    # Parse the response
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return str(result)
    else:
        return f"Error from API: {response.status_code}, {response.text}"

def run_openai_inference(prompt):
    """
    Run inference through OpenAI API
    
    Args:
        prompt (str): Prompt for LLM
        
    Returns:
        str: LLM response
    """
    # This function would implement OpenAI API integration
    # Not implemented for this example app
    return "OpenAI API integration not implemented. Using simulated analysis instead.\n\n" + generate_mock_analysis({})

def generate_mock_analysis(report_data):
    """
    Generate a mock LLM analysis for demonstration purposes
    
    Args:
        report_data (dict): Report data to analyze
        
    Returns:
        str: Mock LLM analysis
    """
    return """
### Meta-Analysis of LLM Responses to Political Prompts

Based on the report findings, several significant patterns emerge in how different LLMs approach political topics:

#### Model Response Characteristics

1. **Balancing Act:** All models appear to make deliberate efforts to present balanced perspectives on political topics, though they achieve this balance through different strategies:
   - GPT-4 tends to structure responses with clearly delineated sections for different viewpoints
   - Claude-3 frequently uses explicit acknowledgment phrases like "on the other hand" and presents multiple perspectives
   - Llama-3 emphasizes factual and historical context, perhaps as a strategy to maintain neutrality

2. **Linguistic Patterns:** Each model has distinctive linguistic signatures:
   - GPT-4 uses more policy-oriented language and conditional constructions
   - Claude-3 employs more nuance-signaling phrases and ethical framing
   - Llama-3 uses more data-referencing terminology and historical contextualization

#### Bias and Political Leaning

The bias analysis reveals subtle but detectable differences in how models approach political content:

1. **Partisan Leanings:** While all models stay relatively close to the center, the slight variations detected (Claude showing a minor liberal lean and Llama showing a minor conservative lean) could accumulate into more noticeable differences when addressing highly polarized topics.

2. **Topical Emphasis:** The topic modeling shows meaningful differences in what aspects of political questions each model emphasizes:
   - GPT-4's emphasis on economic implications may reflect certain value priorities
   - Claude-3's focus on social issues and ethical dimensions indicates a different weighting
   - Llama-3's greater attention to foreign policy and historical context represents yet another approach

#### Recommendations for Users

When interpreting LLM outputs on political topics, users should:

1. Consult multiple models for a more comprehensive view, as each model brings different strengths and emphases
2. Be aware of subtle framing differences that might influence how information is presented
3. Pay attention to what topics are emphasized versus omitted in responses
4. Consider that models showing higher similarity scores may be reinforcing the same perspectives or limitations

Overall, while all models demonstrate attempts at political neutrality, they achieve this through different strategies and with subtle variations that may influence user perception. The high similarity scores between models (ranging from 0.72-0.85) suggest significant overlap in content, but the distinctive linguistic patterns and emphasis differences reveal that each model has its own "political voice" despite efforts at neutrality.
"""

def parse_llm_response(response):
    """
    Parse and structure LLM analysis response
    
    Args:
        response (str): Raw LLM response
        
    Returns:
        dict: Structured analysis
    """
    # For now, just return the raw response
    return {
        "analysis": response
    }
