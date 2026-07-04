make sure all of the ablations aren't inefficient and rerun all of them to ensure they work


make sure that the telescope perplexity is or is not equivalent to the skip the first 20 tokens version and see whether or not we should mention that in the code/ paper? They are not equivalent. Make sure that this is noted somewhere that token skipping may slightly increase performance because it cuts through some of the noise early on in the text. Oftentimes early token information is just noise since there is less information to work off of.



Make sure all of the licenses are properly attributed, the texts are included in the licenses folder, and all attributions (such as mentions of the authors and links to original works etc) are all included. (ie https://www.reddit.com/r/COPYRIGHT/comments/vfo54v/question_about_cc_by_40_license/)



Make a pip package for telescope so its easy to import and easy to use


Where is the reuters conversion script?

rename all of the dataset creation scripts and make sure they all work


Put the ghostbuster-data into another huggingface repository and make sure it is cloned in the installation instructions


make all of the plots in the repository follow the config.yaml for all of the plot coloring where it makes

make sure to upload the new version of the experiment_results