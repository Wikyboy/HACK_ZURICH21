# HACK ZURICH 2021
## DunGANs and Dragons (GOTY 2021)
We have developed this virtual game board to allow people to play Dungeons and Dragons together in a shared virtual game environment. The game is run through Unity, with the models and image generation controlled with external python scripts.
On starting the game, players can enter a description of their character. The Dungeon Master has control over the environment, where they can describe the next scene that the players will encounter. This will trigger the generation of new images by the Generative Adversarial Network working in the background. The game will read the Dungeon Master's input description and identify the creatures they are describing. It will then generate new game scenes that are representative of these creatures to share with the players.
## Usage
Start the GAN by running `main.py` (WARNING: it may take a while to load the first time!) and then start the game using `HACK_ZURICH_2D.exe`.
Once it is running, try out the following descriptions as the DM:
* A wizard god smirks at you
* The elf troll goblin ate the man.
* Deep sea monster emerges from the depths.
* You see a scorpion spider in the undergrowth.
* A mutant tiger sleeps nearby.
* A gazelle with inferno in its eyes.
* A koala kangaroo cross breed.
* A gigantic crab awaits in the depths.
* A bustard whale.
* A rabid meerkat approaches you.
## Authors and Acknowledgment
* Mihnea Romanovschi
* Alex Saoulis
* Kathryn Baker
Huge thanks are extended to the Google Developers responsible for [BigGAN](https://arxiv.org/abs/1809.11096 "link to paper related to BigGAN"), as well as those responsible for the pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/ "link to GloVe datasets") word vector datasets.
