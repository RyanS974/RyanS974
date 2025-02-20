# datashare.py
# at the moment, I am storing all results in corpus. if I need more data sharing variables I will put them here.

class DataShare():
    overview_assignment = '''
    The second homework assignment for ARI 525 NLP, due on 2/18/25.    
    
    The report is in a markdown file in the same directory as this file, titled REPORT.md.
    The README.md is also in this directory.
    '''

    # init
    def __init__(self):
        self.info = "default information string"

        # dataset
        self.corpus = {}
