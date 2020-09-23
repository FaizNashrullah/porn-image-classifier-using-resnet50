Porn Image Classifier Using ResNet-50
=====================================

This is a project aimed at classifying pornographic images. Keras
framework is used and the code is composed with full of python
languange. We use CNN with **ResNet-50** as architecture because it can
provide the shortcut connections concept that used to avoid a vanishing
gradient problem that often happen in deep convoluitional layer cases.

Getting Started
===============

If you are not familiar with this kind of project, you can follow the
several guides below to run the project properly. Basically u can run
this project using command control but some IDE like spyder, vsscode,
etc. is preferred.

Prerequisites
-------------

-   For the dataset, we used **NPDI dataset** for this project. We take
    2 out of 3 class from the dataset, that is npe(non porn easy) and
    porn images. You can change the dataset according to your preference
    but it may take a different result.
-   Its recommended to make the **Environment** of your project.
-   Library installed, check on **library.txt**.

Preparation
-----------

-   Create and environment . \>If the environment already exists, you
    can install the requirement libraries on the library.txt using pip
    method or conda install if you are working with anaconda navigator.
    (Install the library on your environment directory path, you can use
    cmd, terminal, anaconda prompt, etc.)
-   Download all the files in the repo, put in one folder and place that
    folder inside your environment folder. download the trained\_model
    folder if )
-   Make a datasets folder and place the file. You can change the
    dataset, number of class, and anything else as your preference. \>
    In this project the class is divided by 2, **np (non porn)** and
    **porn**. Non porn class images is taken from npe class and porn
    class images from porn class of NPDI dataset.

    ``` {.yaml}
    datasets/
        NPDI/
            test/
                np/
                    np01.jpg
                    np02.jpg
                    ...
                porn/
                    porn01.jpg
                    porn02.jpg
                    ...
                ...
             train/
                np/
                    ...
                porn/
                    ...
                ...
             val/
                np/
                    ...
                porn/
                    ...
                ...
    ```

Training part
-------------

Open the **train.py** on IDE and run it. The batch size, learning rate,
and other parameters or hyper-parameters can be adjusted according to
your need. If the training part already conducted, you'll get some
auto-saved results in your directory which contains:

``` {.yaml}
- histoy.xlsx
- total of computational time.xlsx
- train vs val acc.png
- train vs val loss.png
- etc...
```

This part will produce resnet50.h5 model of your resnet that will saved
on folder named models.

If you want to skip the training process and try the **trained models**,
you can download the model that was trained with learning rate 0.005,
batch size 32, and epoch 100 in this link.

``` {.yaml}
https://drive.google.com/drive/folders/1BhiOagqZXmKLhRIAlunoFArPfKjKQO-t?usp=sharing
```

Testing Part
------------

If the model has been saved, then run the **predict.py** to make a
prediction of images that stored in test folder in datasets directory.
The result will be stored in several excel files:

``` {.yaml}
- acc.xlsx (accuracy)
- conf matrix.xlsx (confusion matrix)
- Precision recall fscore support.xlsx
- and other prediction result files
```

Built with
----------

-   Python
-   Keras
-   TensorFlow
-   Pandas
-   Sklearn
-   NumPy
-   Matplotlib

