#!/usr/bin/env python3
# homework 3, word embeddings

import os
import dataset
import training
import comparison
import bias
import classification

PREPROCESSED_DIR = "preprocessed"

def main():
    ds = None
    pds = None
    while True:
        print("Please select an option:")
        print("1. Load dataset")
        print("2. Preprocess dataset")
        print("3. Train model")
        print("4. Compare models")
        print("5. Evaluate bias")
        print("6. Classify text")
        print("7. Display most similar words")
        print("8. Display vector arithmetic results")
        print("9. Display semantic field results")
        print("10. Display word similarity results")
        print("11. Display analogy results")
        print("12. Exit")
        option = input("Enter option: ")
        if option == "1":
            ds = dataset.load_data()
            dataset.download_embeddings()
        elif option == "2":
            pds = dataset.load_preprocessed_dataset(PREPROCESSED_DIR)
            if pds is None:
                if ds is not None:
                    pds = dataset.preprocess_dataset(ds)
                    dataset.save_preprocessed_dataset(pds, PREPROCESSED_DIR)
                else:
                    print("Please load the dataset first.")
            else:
                print("Preprocessed dataset loaded successfully.")
        elif option == "3":
            if pds is not None:
                skipgram_model = training.train_skipgram(pds)
                cbow_model = training.train_cbow(pds)
            else:
                print("Please preprocess the dataset first.")
        elif option == "4":
            comparison.compare()
        elif option == "5":
            bias.evaluate_bias()
        elif option == "6":
            classification.classify()
        elif option == "7":
            comparison.display_most_similar()
        elif option == "8":
            comparison.display_vector_arithmetic()
        elif option == "9":
            comparison.display_semantic_field()
        elif option == "10":
            comparison.display_word_similarity()
        elif option == "11":
            comparison.display_analogies()
        elif option == "12":
            return
        else:
            print("Invalid option. Please try again.")

# dunder main: if this script is run, call the main function
if __name__ == "__main__":
    main()