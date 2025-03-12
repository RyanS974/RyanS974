#!/usr/bin/env python3
# homework 3, word embeddings

import dataset
import training
import comparison
import bias
import classification

def main():
    while True:
        print("Please select an option:")
        print("1. Load dataset")
        print("2. Preprocess dataset")
        print("3. Train model")
        print("4. Compare models")
        print("5. Evaluate bias")
        print("6. Classify text")
        print("7. Exit")
        option = input("Enter option: ")
        if option == "1":
            ds = dataset.load_data()
            dataset.download_embeddings()
        elif option == "2":
            if ds is not None:
                pds = dataset.preprocess_dataset(ds)
            else:
                print("Please load the dataset first.")
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
            return
        else:
            print("Invalid option. Please try again.")

# dunder main: if this script is run, call the main function
if __name__ == "__main__":
    main()