


### 2026-06-09

The current version of this repo has rather bloated code, and needs some
refactoring.

Additionally, some conceptual choices could be reconsidered.


#### Refactoring

- Removing unused code
    - [x] File cleanup 1: see [refactoring_202606_A.md](refactoring_202606_A.md).
    - [ ] File cleanup 2: probably there are still unused functions/functions
    that should be moved outside of cheeky-cell main scope.
- [x] Redesigning how test and training set are defined (without metadata xlsx)
    - Done, but kept xlsx because it allows easy re-assignment of test/train
    and user-friendlyness. See [refactoring_202606_B.md](refactoring_202606_B.md).
- (**conceptual**) Rescaling per channel is probably not very detrimental, but unclear
which goal it precisely achieves. 
    - Currently baseline is subtracted per channel.
    - This doesn't add information.
    - Goal should be to equalize the data among all the different images.
    - Perhpas test performance with different settings??
    - Allow the rescaling function to be chosen by user. Supply a list of some good options.
- (**conceptual**) Previously, I used tiling to get more variety in training data, and avoiding
having to annotate large image files.
    - This is however still baked into the core of the repo, and this feature
    is not used any more. 
    - Should be relocated to a pre-processing step.





