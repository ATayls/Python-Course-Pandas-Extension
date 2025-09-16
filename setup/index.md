---
layout: default
title: Setup Instructions
---

# Setup Instructions

This lesson is designed to be run on a personal computer.
All the software used is freely available online.
Here you will find instructions on how to set up your environment for the workshop.


## Install Python
In this workshop, we will be using Python 3 along with several popular libraries.
While you can install a plain-vanilla Python and all required libraries manually,
we recommend installing Anaconda, a Python distribution that includes everything needed for this lesson.
Detailed installation instructions for various operating systems can be found in the [Anaconda documentation](https://docs.anaconda.com/anaconda/install/).

### Step-by-Step Installation Guide:
1. **Download Anaconda**: Visit the [Anaconda distribution page](https://www.anaconda.com/products/distribution) and download the installer suitable for your operating system.
2. **Install Anaconda**:
    - **Windows**: Double-click the downloaded `.exe` file and follow the instructions.
    - **macOS**: Open the downloaded `.pkg` file and follow the installation prompts.
    - **Linux**: Open a terminal and run the following commands:
        ```bash
        bash ~/Downloads/Anaconda3-<version>-Linux-x86_64.sh
        ```
        Follow the on-screen instructions.

## Launch Python Interface
To start working with Python, we need to launch a program that will interpret and execute our Python commands. Below we list several options. If you don’t have a preference, proceed with the top option in the list that is available on your machine. Otherwise, you may use any interface you prefer.

### Option A: Jupyter Notebook
A Jupyter Notebook provides a browser-based interface for working with Python, which is particularly useful for web scraping tasks. If you installed Anaconda, you can launch a notebook in two ways:

<iframe width="560" height="315" src="https://www.youtube.com/embed/WUeBzT43JyY" frameborder="0" allowfullscreen></iframe>

1. **Anaconda Navigator**:
    - Open Anaconda Navigator from your applications menu.
    - Click on the "Launch" button under the Jupyter Notebook icon.

2. **Command Line (Terminal)**:
    - Open a terminal or command prompt.
    - Run the command:
        ```bash
        jupyter notebook
        ```
    - This will open the Jupyter Notebook interface in your default web browser.

### Option B: IPython Interpreter
IPython is an enhanced interactive Python interpreter that offers additional features compared to the plain-vanilla Python interpreter. It is included with Anaconda.

To start using IPython, execute the following command in your terminal or command prompt:
```bash
ipython
```

### Option C: Plain-Vanilla Python Interpreter
To launch a plain-vanilla Python interpreter, execute:
```bash
python
```
If you are using Git Bash on Windows, you need to call Python via `winpty`:
```bash
winpty python
```

## Additional Setup for Web Scraping
For this workshop, we will be using several Python libraries that are commonly used for web scraping. These include `requests`, `BeautifulSoup`, and `pandas`. You can install these libraries using `pip`, which is included with Anaconda.

### Optional: Anaconda environments
If you are using Anaconda, you can install the required libraries in your base environment. Alternatively, you can create a new environment for this workshop by running the following command in your terminal or command prompt:
see the [Anaconda documentation](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/) for more information on managing environments.
```bash
conda create --name web-scraping python=3.8
conda activate web-scraping
```

### Installing Required Libraries

Run the following command in your terminal or command prompt to install these libraries:
```bash
pip install pandas statsmodels
```

## Verifying the Installation
To ensure everything is set up correctly, create a new Jupyter Notebook and run the following code snippet:
```python
import pandas as pd
import statsmodels.api as sm

print("Setup is complete!")
```
If you see the message "Setup is complete!" printed in the output, you are ready to start the workshop.

## Conclusion
If you encounter any issues during setup, refer to the Anaconda documentation or please get in touch prior to the workshop starting to help resolve the issues.

