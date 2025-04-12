# LLM Response Comparator

## Overview

This application allows you to compare responses from different Large Language Models (LLMs) on political topics. Using various NLP techniques, it analyzes differences in content, phrasing, bias, and overall approach between models.

## Features

- **Topic Modeling**: Identifies key topics emphasized by different LLMs
- **N-gram Analysis**: Analyzes characteristic phrases and patterns of each model
- **Bias Detection**: Detects potential biases in how models approach political topics
- **Bag of Words Analysis**: Compares word usage across different models
- **Similarity Metrics**: Quantifies how similar responses are between models
- **Difference Highlighting**: Identifies specific content that varies between models
- **Classification**: Groups responses based on various characteristics
- **Visualizations**: Visual representation of analysis results
- **Report Generation**: Comprehensive report of all findings
- **LLM Meta-Analysis**: Optional analysis of results by another LLM

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/llm-response-comparator.git
cd llm-response-comparator
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

## Usage

Run the application:
```
python app.py
```

The application will launch a Gradio interface with the following tabs:

1. **Home**: Introduction to the application
2. **Dataset Input**: Enter prompts and LLM responses or load example datasets
3. **Analysis**: Run various analysis methods on the dataset
4. **Visualization**: View visualizations of analysis results
5. **Classification**: Classify responses based on various characteristics
6. **Report**: Generate a comprehensive report of all findings

## Project Structure

```
/llm_compare_app/
├── app.py                  # Main Gradio application
├── ui/                     # UI components
│   ├── main_screen.py      # Home screen UI
│   ├── dataset_input.py    # Dataset input screen UI
│   ├── analysis_screen.py  # Analysis options screen UI
│   ├── visualization_screen.py  # Visualization screen UI
│   ├── classification_screen.py # Classification screen UI
│   └── report_screen.py    # Report screen UI
├── processors/             # Analysis modules
│   ├── topic_modeling.py   # Topic modeling analysis
│   ├── ngram_analysis.py   # N-gram extraction and comparison
│   ├── bias_detection.py   # Bias detection in responses
│   ├── bow_analysis.py     # Bag of words implementation
│   ├── metrics.py          # Similarity metrics
│   └── diff_highlighter.py # Highlighting differences
├── visualizers/            # Visualization modules
├── utils/                  # Utility functions
│   ├── llm_analyzer.py     # LLM meta-analysis
│   └── report_generator.py # Report generation
├── models/                 # Classification models
├── examples/               # Example datasets
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation
```

## Example Dataset

The application comes with example datasets containing responses from popular LLMs (GPT-4, Claude-3, Llama-3) to political questions such as:

- Views on US presidential policies
- Universal healthcare
- Gun control
- Immigration policy
- Foreign relations

Each dataset includes the original prompt, model names, and complete responses.

## Dependencies

- gradio: Web interface
- nltk: Natural language processing
- scikit-learn: Machine learning and text processing
- matplotlib: Visualization
- numpy: Numerical computing
- pandas: Data manipulation

## Limitations

- The bias detection uses simple lexicon-based approaches that may miss subtle forms of bias
- Topic modeling on short texts can be imprecise
- Classification accuracy depends on the size and quality of input dataset
- Semantic similarity is approximated using statistical methods rather than deep learning

## Future Improvements

- Add support for more sophisticated bias detection models
- Integrate embedding-based semantic similarity
- Add support for more languages
- Expand classification capabilities
- Add export options for visualizations
- Create persistent database of analyses

## License

MIT

## Author

Your Name - [your.email@example.com](mailto:your.email@example.com)