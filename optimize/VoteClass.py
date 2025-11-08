from collections import Counter
from scipy.stats import mode
import numpy as np

class Voter:
    def __init__(self, model_list, X_test, vote_type='s'):
        self.model_list = model_list
        self.vote_type = vote_type
        self.X_test = X_test

    def vote_soft(self):
        """
        sums the predic_proba across models and pick the highest. models must support predict_proba.
        """
        prob_sum = np.sum([model.predict_proba(self.X_test) for model in self.model_list], axis=0)
        soft_vote_pred = np.argmax(prob_sum, axis=1)
        # Error fixed with chatpgt
        soft_vote_preds = self.model_list[0].classes_[np.argmax(prob_sum, axis=1)]
        return soft_vote_preds

    def vote_hard(self):
        # Vote majority across models
        all_preds = np.array([model.predict(self.X_test) for model in self.model_list])
        all_preds = np.array(all_preds)
        final_predic = []

        num_sample = all_preds.shape[1]

        for i in range(num_sample):
            sample_pred = all_preds[:, i]
            #Counts occurances of each label.
            labels, counts = np.unique(sample_pred, return_counts=True)

            # Get the label with the highest count.
            majority_label = labels[np.argmax(counts)]
            final_predic.append(majority_label)

        return np.array(final_predic)

    def choose_vote(self):
        if self.vote_type.lower() == 's':
            return self.vote_soft()

        elif self.vote_type.lower() == 'h':
            return self.vote_hard()

        elif self.vote_type.lower() == 'a':
            return [self.vote_soft(), self.vote_hard()]

        else:
            print("Error, wrong input choose s for soft, h for hard or a for all")