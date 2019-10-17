# SyntObj: An object focused image synthesizer

Created by Alassane Watt at [JRL-Lab](http://jrl-umi3218.github.io/) at AIST Tsukuba Research and CentraleSupelec University (https://www.centralesupelec.fr/).

### Introduction

This code allows to synthesize images where the object of interest is generated in different arbitrary poses.


### Citation

If you find SyntObj useful in your research, please consider citing:

    @inproceedings{tawlas,
        Author = {Watt, Alassane},
        Title = {SyntObj: An object focused image synthesizer }
    }


### Installation

	Tested environment:
		- Ubuntu 16.04
		- CUDA 9.0
		- gcc 4.8


In the sequel (Please change $ROOT variable to the path of the project root)

1. Install dependencies:
	- [Pangolin](https://github.com/stevenlovegrove/Pangolin) {tested branch: c2a6ef524401945b493f14f8b5b8aa76cc7d71a9}
	- [Eigen](https://eigen.tuxfamily.org) {tested version: 3.3.7}
	- [boost](https://www.boost.org/) {tested version: 1.66.0}
	- [Sophus](https://github.com/strasdat/Sophus) {tested branch: ef9551ff429899b5adae66eabd5a23f165953199}
	- [nanoflann](https://github.com/jlblancoc/nanoflann) {tested commit: 49cd1120247ceaab93c1aee89fb647d831f9c9ba}
	- libsuitesparse-dev

	Change hard coded pathes in CMakeLists.txt

2.  Build the kinect fusion library:

	```Shell
	cd $ROOT/lib/kinect_fusion
	mkdir build
	cd build
	cmake ..
	make
	```

3. Build the synthesize library
	```Shell
	cd $ROOT/lib/synthesize
	mkdir build
	cd build
	cmake ..
	make
	```

	Add the path of the built libary libsynthesizer.so to python path
	```Shell
	export PYTHONPATH=$ROOT/lib/synthesize/build:$PYTHONPATH
	```


### Running the synthesizer

1. Provide the right arguments to the bash script file  ( in $ROOT/synthesis/scripts/synthesis.sh)
	* xyz points of the object model ( "*.xyz" 3-column file)
	* obj model file (tested with '.obj' format) with complementary files ('.mtl' etc..) in the same folder.
	* poses (7-column file)
	* extent file (3-column file)
	* Nb of images to generate

2. run the following script from the root folder ($ROOT/)
    ```Shell
    ./synthesis/scripts/synthesis.sh
    ```


### Output
The output is in hdf5 format. It contains the images and their metadata.