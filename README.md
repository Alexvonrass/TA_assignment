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
features that made sense to me from a logical standpoint. Here they are with some motivation behind each feature:

1) Distance between origin and destination - distance containts 
2) Origin and destination as categories directly using LightGBM categorical feature - Some origin/destination pairs will have a much higher probability of booking
3) Time-based features (hour, day of week e.t.c.) and time-delta features (length of stay, time until trip): 
Potentially people could look through potential bookings in the morning and do the actual booking in the evening, so this is useful information. 
4) Features based on each user's previous searches - i.e. how many searches in the last X minutes by this user: If a person is searching for a while it might mean that they're dissatisfied with something or, in contrast, fine-tuning small details and are definitely looking to book
5) Origin-based search features - similar to 4) but based on origin instead of user, see 6)
6) I did not do this for destination-based searches because it's too similar to origin-based, but in hindsight destination-based is probably more interesting, for example if a champions league tie finishes and we know that an important match will take place in a city it might drive bookings
7) Total searches / mean of target on the platform in the last X minutes: this is general information about the platform. In the past days we can see that a sudden increase in searches drives more bookings, but not always (perhaps this is fraudulent traffic or just of less quality)

Other features that could be added
8) More specific user-based features, i.e. standard deviation of distance between origin and destination in previous searches



