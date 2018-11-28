# Emerging Tech Assignment

The four folders in this repo contain the following:

<ul>
  <li>numpy-random: A jupyter notebook explaining the numpy.random package and demonstrating some if it's functions</li>
  <li>iris-dataset: A jupyter notebook explaining the iris dataset and containing a neural network which attempts to predict the different species of      flowers input</li>
  <li>mnist: A jupyter notebook explaining the mnist dataset and showing how to open and read zipped datasets into memory</li>
  <li>digit-recognition: A jupyter notebook explaining the digitrec.py script contained within the folder, and showing how the neural network can read and recognise different handwritten digits, taken from the mnist dataset</li>
</ul>

## Using these notebooks

In order to use these notebooks and python script, the following steps must be taken:

<ol>
  <li>Clone this repo to your local machine</li>
  <li>Download and install Anaconda, this should install all the nessessary python packages and jupyter</li>
  <li>Open your console, navigate to the downloaded repo on your machine, and run "jupyter lab"</li>
  <li>Once jupyter lab has opened in your browser, open the desired notebook from the left, and then open the 'Run' tab along the top and select 'Run All'</li> 
</ol>

## Running the python script

The python script, contained in the digit-recognition folder, can be run either from the top of the digit-recognition notebook, or can be run from console by navigating into the digit-recognition folder, and running "python digitrec.py"

The program will explain itself, giving the user the option to either train the model more, or use it in it's current state.
It will also offer to either use random digits from the test dataset, or allow the user to select there own image and test from that.

*Note: There are handwritten digit images included in a seperate folder, if the user doesn't have there own images to test with*
