#!/usr/bin/env python3
# bias.py

from gensim.models import KeyedVectors
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
import pandas as pd
import numpy as np

# Function to evaluate bias using the WEAT metric
def evaluate_bias():
    # Load word embedding models using Gensim
    model_paths = [
        "models/cbow_model.model", 
        "models/skipgram_model.model", 
        "models/glove-wiki-gigaword-100.txt", 
        "models/fasttext-wiki-news-subwords-300.vec" 
    ]
    
    # Load models using Gensim
    gensim_models = []
    model_names = ["CBOW", "Skip-gram", "GloVe", "FastText"]
    
    # for each model path, try to load the model
    for i, path in enumerate(model_paths):
        try:
            if path.endswith('.model'):
                # For trained models
                model = KeyedVectors.load(path)
                # If it's a Word2Vec model, get the word vectors
                if hasattr(model, 'wv'):
                    model = model.wv
                gensim_models.append(model)
            elif 'fasttext' in path:
                # For FastText model - try binary first, if that fails, try non-binary
                try:
                    gensim_models.append(KeyedVectors.load_word2vec_format(path, binary=True))
                except:
                    gensim_models.append(KeyedVectors.load_word2vec_format(path, binary=False))
            else:
                # For GloVe and other text-based models
                gensim_models.append(KeyedVectors.load_word2vec_format(path, binary=False))
            print(f"Successfully loaded: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            print("Skipping this model and continuing...")
            continue
    
    # Check if any models were successfully loaded
    if not gensim_models:
        print("No models were successfully loaded. Cannot evaluate bias.")
        return
    
    # Wrap Gensim models with WEFE's WordEmbeddingModel
    models = []
    for i, model in enumerate(gensim_models):
        try:
            # Check the actual signature of WordEmbeddingModel
            try:
                # Try with model_name parameter
                models.append(WordEmbeddingModel(model, model_name=model_names[i]))
            except TypeError:
                # If that fails, try without model_name parameter and set it after
                wem = WordEmbeddingModel(model)
                wem.name = model_names[i]
                models.append(wem)
            print(f"Successfully wrapped model: {model_names[i]}")
        except Exception as e:
            print(f"Error wrapping model {i+1} ({model_names[i]}): {e}")
            continue
    
    if not models:
        print("No models could be wrapped with WEFE. Cannot evaluate bias.")
        return
    
    # Define the target and attribute word sets for the WEAT test
    # Use more common words that are likely to be in the vocabularies
    target_sets = [
        ["man", "boy", "father", "brother", "son", "husband", "uncle", "gentleman"],
        ["woman", "girl", "mother", "sister", "daughter", "wife", "aunt", "lady"]
    ]
    attribute_sets = [
        ["math", "science", "technology", "computer", "logic", "system", "engineer"],
        ["art", "music", "literature", "poetry", "dance", "creative", "design"]
    ]

    # Create a WEAT query
    query = Query(
        target_sets=target_sets,
        attribute_sets=attribute_sets,
        target_sets_names=["Male terms", "Female terms"],
        attribute_sets_names=["Science terms", "Arts terms"]
    )

    # Initialize the WEAT metric
    weat = WEAT()

    # Store results for all models
    all_results = []
    
    # Process each model individually
    for model in models:
        try:
            print(f"Evaluating bias for model: {model.name}")
            
            # Increase the lost words threshold to avoid rejecting results
            wefe_result = weat.run_query(
                query, 
                model,
                lost_vocabulary_threshold=0.7,  # Allow more words to be missing
                lost_words_threshold=0.7        # Allow more words to be missing
            )
            
            # Check the type of the result and process accordingly
            if wefe_result is not None:
                if isinstance(wefe_result, dict):
                    # New API returns a dictionary
                    if 'weat' in wefe_result:
                        result_value = wefe_result['weat']
                        if isinstance(result_value, float) and not np.isnan(result_value):
                            all_results.append({
                                'Model': model.name,
                                'WEAT Score': result_value,
                                'Interpretation': interpret_weat_score(result_value)
                            })
                            print(f"  WEAT Score: {result_value:.4f}")
                        else:
                            print(f"  Invalid WEAT score for {model.name}: {result_value}")
                    else:
                        print(f"  No 'weat' key in results for {model.name}. Available keys: {wefe_result.keys()}")
                elif hasattr(wefe_result, 'iloc'):
                    # Old API returns a DataFrame
                    result_value = wefe_result.iloc[0, 0]
                    if not np.isnan(result_value):
                        all_results.append({
                            'Model': model.name,
                            'WEAT Score': result_value,
                            'Interpretation': interpret_weat_score(result_value)
                        })
                        print(f"  WEAT Score: {result_value:.4f}")
                    else:
                        print(f"  NaN result for {model.name}")
                else:
                    print(f"  Unexpected result type for {model.name}: {type(wefe_result)}")
                    print(f"  Result content: {wefe_result}")
            else:
                print(f"  No results returned for {model.name}")
        except Exception as e:
            print(f"Error during WEAT analysis for {model.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Display combined results in a table
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\nWEAT Bias Analysis Results:")
        print("-----------------------------")
        print(results_df.to_string(index=False))
        print("\nHigher positive scores indicate stronger association of males with science (and females with arts).")
        print("Negative scores indicate the opposite association.")
    else:
        print("No results were obtained from any model.")

    # Print vocabulary coverage information
    print("\nVocabulary Coverage Check:")
    print("-------------------------")
    
    # Check using the original gensim models instead of WEFE models
    for i, model in enumerate(gensim_models):
        if i >= len(model_names):
            continue
            
        model_name = model_names[i]
        print(f"\nModel: {model_name}")
        print("Target and attribute words not in vocabulary:")
        
        for set_name, word_set in zip(["Male terms", "Female terms", "Science terms", "Arts terms"],
                                     [target_sets[0], target_sets[1], attribute_sets[0], attribute_sets[1]]):
            missing_words = []
            for word in word_set:
                if word not in model:
                    missing_words.append(word)
            
            if missing_words:
                print(f"  {set_name}: {', '.join(missing_words)}")
            else:
                print(f"  {set_name}: All words found in vocabulary")

# Helper function to interpret the WEAT score
def interpret_weat_score(score):
    """Provide a simple interpretation of the WEAT score"""
    if score > 1.0:
        return "Strong male-science bias"
    elif score > 0.5:
        return "Moderate male-science bias"
    elif score > 0.1:
        return "Slight male-science bias"
    elif score > -0.1:
        return "Neutral"
    elif score > -0.5:
        return "Slight female-science bias"
    elif score > -1.0:
        return "Moderate female-science bias"
    else:
        return "Strong female-science bias"

# Dunder method to run the script
if __name__ == "__main__":
    evaluate_bias()