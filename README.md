# AI in Medical Imaging Project: Few-shot Learning for the MONKEY Challenge

We explored Prototypical Networks for few-shot classification of lymphocytes and monocytes in PAS-stained kidney biopsy images from the MONKEY Challenge (Studer et al., 2025).

## Running the Project
1. Run ```dataset_prep.ipynb```, a preprocessing pipeline based on MONKEY Github tutorial (ThibautGoldsborough, 2025), to generate an .h5 file containing: 128x128 image patches centered on annotated nuclei, the class labels per patch (0 = lymphocyte, 1 = monocyte, 2 = other), and a binary nucleus mask per patch as the fourth channel.
2. Configure training parameters in ```config.py```
3. Train model by running ```train.py```
4. Evaluate model by running ```evaluate.py```

## References
Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. Advances
in neural information processing systems, 30.

L. Studer, D. Ayatollahi van Midden, F. Hilbrands, J. L., Kers, and J van der Laak. Monkey
challenge: Detection of inflammation in kidney biopsies. grand challenge., 2025. URL https://monkey.grand-challenge.org/. Accessed: May 28, 2025.

Pachetti, E., & Colantonio, S. (2024). A systematic review of few-shot learning in medical
imaging. Artificial intelligence in medicine, 102949.

ThibautGoldsborough. (2025). GitHub - ThibautGoldsborough/instanseg-monkey-challenge.
GitHub. https://github.com/ThibautGoldsborough/instanseg-monkey-challenge
