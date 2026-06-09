


### 2026-06-09

The current version of this repo has rather bloated code, and needs some
refactoring.

Additionally, some conceptual choices could be reconsidered.


#### Refactoring

- Removing unused code
- Redesigning how test and training set are defined (without metadata xlsx)

See also: [refactoring_202606.md](refactoring_202606.md).

#### Conceptual choices

- Rescaling per channel is probably not very detrimental, but unclear
which goal it precisely achieves. 
    - Currently baseline is subtracted per channel.
    - This doesn't add information.
    - Goal should be to equalize the data among all the different images.
    - Perhpas test performance with different settings??
    - Allow the rescaling function to be chosen by user. Supply a list of some good options.
- Previously, I used tiling to get more variety in training data, and avoiding
having to annotate large image files.
    - This is however baked into the core of the repo, and this feature
    is not used any more. 
    - Should be relocated to a pre-processing step.

