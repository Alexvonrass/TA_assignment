# TA_assignment

## This is Alexander Rass' submission for the TravelAudience test assignment

### How to run the code

`git clone https://github.com/Alexvonrass/TA_assignment.git`

To clone my conda env, you can run 

`conda env create -f ta_assignment.yml`

in your local environment to ensure reproducibility, then run the code

`python train.py`

### Additional comments

The dataset is quite small and the last few days are quite different from the previous ones, you can see 
how I arrived at that conclusion in the Archieve/Adversarial_d2d_comparison.ipynb.

Because of that, and since the instruction mentioned focusing on the engineering part over the metric performance, I created
features that made sense to me from a logical standpoint. They are:

1) Distance between origin and destination
2) Origin and destination as categories directly using LightGBM categorical feature
3) Time-based features (hour, day of week e.t.c.)
4) Features based on each user's previous searches - i.e. how many searches in the last X minutes by this user
5) Origin-based search features - similar to 4) but based on origin instead of user
6) I did not do this for destination-based searches because it's too similar to origin-based
7) Total searches / mean of target on the platform in the last X minutes

Other features that could be added
8) More specific user-based features, i.e. standard deviation of distance between origin and destination in previous searches



