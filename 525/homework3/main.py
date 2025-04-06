#!/usr/bin/env python3
# main.py for the fourth homework assignment

import sys

import dataset
import fine_tune
import zero_shot
import baselines

# main function
def main():
    '''
    Main function for the homework assignment.
    '''

    print("Welcome to my homework assignment on 'Pre-trained Transformer-based Models'")

    # main loop
    while True:
        # print menu
        print("Main Menu:")
        print("0. * Overview *")
        print("1. Dataset")
        print("2. Fine-tuning pre-trained models")
        print("3. Zero-shot classification")
        print("4. Baselines")
        print("5. ** EXIT **")
        choice = input("Please enter your choice: ")
        
        # 2 checks
        # check if choice is valid
        if choice not in ['0', '1', '2', '3', '4', '5']:
            print("Invalid choice. Please try again.")
            continue
        # check if choice is exit
        if choice == '5':
            print("Goodbye")
            sys.exit(0)
        
        # call functions based on choice
        # choice 1, Dataset
        if choice == '1':
            while True:
                # dataset menu
                print("The dataset is the SMS Spam Detection dataset from HuggingFace.")
                print("Dataset menu:")
                print("1. Load dataset")
                print("2. Verify dataset")
                print("3. ** Return to previous menu **")
                choice = input("Please enter your choice: ")

                # check if choice is valid
                if choice not in ['1', '2', '3']:
                    print("Invalid choice. Please try again.")
                    continue
                # check if choice is return to previous menu
                if choice == '3':
                    print("Returning to previous menu...")
                    break

                # call functions based on choice
                # choice 1, load dataset
                if choice == '1':
                    # load dataset
                    dataset.app_data = dataset.loaddataset()

                # choice 2, verify dataset
                if choice == '2':
                    # verify dataset
                    dataset.verifydataset()

        # choice 2, fine-tuning pre-trained models
        elif choice == '2':
            while True:
                # fine-tuning menu
                print("Fine Tuning menu:")
                print("1. Fine-tune DistilBERT")
                print("2. Test DistilBERT")
                print("3. Results of DistilBERT")
                print("   *****")
                print("4. Fine-tune T5")
                print("5. Test T5")
                print("6. Results of T5")
                print("   *****")
                print("7. Compare DistilBERT and T5 results")
                print("8. ** Return to previous menu **")
                choice = input("Please enter your choice: ")

                # check if choice is valid
                if choice not in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    print("Invalid choice. Please try again.")
                    continue

                # check if choice is return to previous menu
                if choice == '8':
                    print("Returning to previous menu...")
                    break

                # call functions based on choice
                # choice 1, fine-tune DistilBERT
                if choice == '1':
                    # fine-tune DistilBERT
                    fine_tune.fine_tune_distilbert(dataset.app_data)

                # choice 2, test DistilBERT
                elif choice == '2':
                    # test DistilBERT
                    fine_tune.test_distilbert(dataset.app_data)

                # choice 3, results of DistilBERT
                elif choice == '3':
                    # results of DistilBERT
                    fine_tune.results_distilbert(dataset.app_data)

                # choice 4, fine-tune T5
                elif choice == '4':
                    # fine-tune T5
                    fine_tune.fine_tune_t5(dataset.app_data)

                # choice 5, test T5
                elif choice == '5':
                    # test T5
                    fine_tune.test_t5(dataset.app_data)

                # choice 6, results of T5
                elif choice == '6':
                    # results of T5
                    fine_tune.results_t5(dataset.app_data)

                # choice 7, compare DistilBERT and T5 results
                elif choice == '7':
                    # compare DistilBERT and T5 results
                    fine_tune.compare_results(dataset.app_data)

        # choice 3, zero-shot classification
        elif choice == '3':
            while True:
                # zero-shot menu
                print("Zero-shot classification menu:")
                print("0. Overview")
                print("   *****")
                print("1. Zero-shot classification with Exaone3.5")
                print("2. Results of Exaone3.5")
                print("   *****")
                print("3. Zero-shot classification with Granite3.2")
                print("4. Results of Granite3.2")
                print("   *****")
                print("5. Comparison of Exaone3.5 and Granite3.2 results")
                print("6. ** Return to previous menu **")
                choice = input("Please enter your choice: ")

                # check if choice is valid
                if choice not in ['0', '1', '2', '3', '4', '5', '6']:
                    print("Invalid choice. Please try again.")
                    continue
                # check if choice is return to previous menu
                if choice == '6':
                    print("Returning to previous menu...")
                    break

                # call functions based on choice
                # choice 0, overview
                if choice == '0':
                    # overview
                    zero_shot.overview(dataset.app_data)

                # choice 1, zero-shot classification with Exaone3.5
                elif choice == '1':
                    # zero-shot classification with Exaone3.5
                    zero_shot.zero_shot_exaone(dataset.app_data)

                # choice 2, results of Exaone3.5
                elif choice == '2':
                    # results of Exaone3.5
                    zero_shot.results_exaone(dataset.app_data)

                # choice 3, zero-shot classification with Granite3.2
                elif choice == '3':
                    # zero-shot classification with Granite3.2
                    zero_shot.zero_shot_granite(dataset.app_data)

                # choice 4, results of Granite3.2
                elif choice == '4':
                    # results of Granite3.2
                    zero_shot.results_granite(dataset.app_data)

                # choice 5, comparison of Exaone3.5 and Granite3.2 results
                elif choice == '5':
                    # comparison of Exaone3.5 and Granite3.2 results
                    zero_shot.compare_results(dataset.app_data)

        # choice 4, baselines
        elif choice == '4':
            while True:
                # baselines menu
                print("Baselines menu:")
                print("1. BoW") # convert the dataset to a BoW representation
                print("2. TF-IDF") # convert the dataset to a TF-IDF representation
                print("   *****")
                print("3. Train Logistic Regression classifier") # train a logistic regression classifier using the TF-IDF representations
                print("4. Test the Logistic Regression classifier") # test the logistic regression classifier on the test set
                print("5. Results of the Logistic Regression classifier") # print the results of the logistic regression classifier
                print("   *****")
                print("6. Train Random Baseline")
                print("7. Test Random Baseline")
                print("8. Results of Random Baseline")
                print("   *****")
                print("9. Train Majority Class Baseline")
                print("10. Test Majority Class Baseline")
                print("11. Results of Majority Class Baseline")
                print("   *****")
                print("12. Compare Baselines")
                print("   *****")
                print("13. Other Logistic Regression classifiers (BoW and Combination with TF-IDF)") # extra feature of other LR classifiers
                print("14. ** Return to previous menu **")
                choice = input("Please enter your choice: ")

                # check if choice is valid
                if choice not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']:
                    print("Invalid choice. Please try again.")
                    continue
                # check if choice is return to previous menu
                if choice == '14':
                    print("Returning to previous menu...")
                    break

                # call functions based on choice
                # choice 1, BoW
                if choice == '1':
                    # convert the dataset to a BoW representation
                    baselines.bow_representation(dataset.app_data)
                # choice 2, TF-IDF
                elif choice == '2':
                    # convert the dataset to a TF-IDF representation
                    baselines.tfidf_representation(dataset.app_data)
                # choice 3, train Logistic Regression classifier
                elif choice == '3':
                    # train a logistic regression classifier using TF-IDF representations
                    baselines.train_logistic_regression(dataset.app_data)
                # choice 4, test Logistic Regression classifier
                elif choice == '4':
                    # test the logistic regression classifier on the test set
                    baselines.test_logistic_regression(dataset.app_data)
                # choice 5, results of Logistic Regression classifier
                elif choice == '5':
                    # print the results of the logistic regression classifier
                    baselines.results_logistic_regression(dataset.app_data)
                # choice 6, train Random Baseline
                elif choice == '6':
                    # train Random Baseline
                    baselines.train_random_baseline(dataset.app_data)
                # choice 7, test Random Baseline
                elif choice == '7':
                    # test Random Baseline
                    baselines.test_random_baseline(dataset.app_data)
                # choice 8, results of Random Baseline
                elif choice == '8':
                    # print the results of the random baseline
                    baselines.results_random_baseline(dataset.app_data)
                # choice 9, train Majority Class Baseline
                elif choice == '9':
                    # train Majority Class Baseline
                    baselines.train_majority_class_baseline(dataset.app_data)
                # choice 10, test Majority Class Baseline
                elif choice == '10':
                    # test Majority Class Baseline
                    baselines.test_majority_class_baseline(dataset.app_data)
                # choice 11, results of Majority Class Baseline
                elif choice == '11':
                    # print the results of the majority class baseline
                    baselines.results_majority_class_baseline(dataset.app_data)
                # choice 12, compare Baselines
                elif choice == '12':
                    # compare Baselines
                    baselines.compare_baselines(dataset.app_data)
                elif choice == '13':
                    # extra feature of other LR classifiers
                    while True:
                        print("Other Logistic Regression classifiers menu:")
                        print("1. Train Logistic Regression (BoW)")                        
                        print("2. Test Logistic Regression (BoW)")
                        print("3. Results of Logistic Regression (BoW)")
                        print("   *****")
                        print("4. Train Logistic Regression (Combination of BoW and TF-IDF)")
                        print("5. Test Logistic Regression (Combination of BoW and TF-IDF)")
                        print("6. Results of Logistic Regression (Combination of BoW and TF-IDF)")
                        print("   *****")
                        print("7. Compare all three LR classifiers, BoW, TF-IDF and Combination")
                        print("8. ** Return to previous menu **")

                        choice = input("Please enter your choice: ")

                        # check if choice is valid
                        if choice not in ['1', '2', '3', '4', '5', '6', '7', '8']:
                            print("Invalid choice. Please try again.")
                            continue
                        # check if choice is return to previous menu
                        if choice == '8':
                            print("Returning to previous menu...")
                            break

                        # call functions based on choice
                        # choice 1, train Logistic Regression (BoW)
                        if choice == '1':
                            # train Logistic Regression (BoW)
                            baselines.train_logistic_regression_bow(dataset.app_data)
                        # choice 2, test Logistic Regression (BoW)
                        elif choice == '2':
                            # test Logistic Regression (BoW)
                            baselines.test_logistic_regression_bow(dataset.app_data)
                        # choice 3, results of Logistic Regression (BoW)
                        elif choice == '3':
                            # results of Logistic Regression (BoW)
                            baselines.results_logistic_regression_bow(dataset.app_data)
                        # choice 4, train Logistic Regression (Combination of BoW and TF-IDF)
                        elif choice == '4':
                            # train Logistic Regression (Combination of BoW and TF-IDF)
                            baselines.train_logistic_regression_combination(dataset.app_data)
                        # choice 5, test Logistic Regression (Combination of BoW and TF-IDF)
                        elif choice == '5':
                            # test Logistic Regression (Combination of BoW and TF-IDF)
                            baselines.test_logistic_regression_combination(dataset.app_data)
                        # choice 6, results of Logistic Regression (Combination of BoW and TF-IDF)
                        elif choice == '6':
                            # results of Logistic Regression (Combination of BoW and TF-IDF)
                            baselines.results_logistic_regression_combination(dataset.app_data)
                        # choice 7, compare all three LR classifiers, BoW, TF-IDF and Combination
                        elif choice == '7':
                            # compare all three LR classifiers, BoW, TF-IDF and Combination
                            baselines.compare_all_logistic_regression(dataset.app_data)                        

# dunder name
if __name__ == "__main__":
    # call main function
    main()