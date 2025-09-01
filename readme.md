
# Objectif de ce code

BONADA Nathan - LAURENT Clément

À partir d'une vidéo d'une personne source et d'une autre personne, notre objectif est de générer une nouvelle vidéo de la cible effectuant les mêmes mouvements que la source. 

[Allez voir le sujet du TP ici](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/)

# User Guide
First you need to generate the images and skeletons from the video you want using VideoSkeleton.py
To launch the project you need to select an other video on DanceDemo.py and select the right Gentype depending on what you want.
To close a video in process press ctrl+c on the terminal.

Gentype 1 :
*GenNearest*, take picture to picture and compare skeleton distance showing the closest skeleton.

Gentype 2 and 3 :
*GenVanillaNN*, from a skeleton create new images that matches the video
The train is launched if the file DanceGenVanillaFromSke.pth doesn't exist.
It is launched on 15 epochs which is enough to see something, after that the neural network doesn't learn a lot.

Gentype 4 :
*GenGan* approach goes beyond GenVanillaNN by leveraging adversarial training, where a generator and discriminator compete to produce more realistic and detailed images, compared to GenVanillaNN, which directly maps skeletons to images without this adversarial refinement.
To train you need to launch the file GenGan.py and you can select the epochs number.
