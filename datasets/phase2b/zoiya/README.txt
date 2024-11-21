# Dataset Description:
HIPA-AI includes 100 Reddit posts, including:
80 posts in a labeled test.csv
10 posts in a labeled dev.csv
10 posts in an unlabeled train_data.csv

This task uses binary classification to predict whether something is HIPAA or not. No need to change the predictions to 1's and 0's--this competition uses "yes" and "no." It also uses "no" as it's scoreboard metric to calculate scores. 

A 2000-post unlabeled dataset and annotation instructions are available if you would like more data. 


# Tips and Tricks
Here are some tips and tricks for uploading your submission:

1. Make sure your prediction is called preds.txt and is inside a zipped folder called submission.zip. For a quick and easy method, just download my sample submission and delete my preds.txt and swap for your own. If you do create your own, check for nested folders and try zipping and renaming the preds.txt file. 

2. Due to my baseline only predicting "no," the scoring looks at the F1-scores/accuracy of the "no" label rather than yes. If your model only predicts "yes," it may give an error. 

3. Make sure your preds.txt does not have headers. Also make sure yes and no are lowercase. 

4. Download the scoring output to see a more detailed list of your results. Also download the scoring error log if you're stuck. 

5. Feel free to reach out if you need anything!

Have fun!