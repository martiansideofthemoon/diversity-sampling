## Instructions

This directory contains the source code to generate multiple diverse solutions for the interactive binary segmentation experiment.
The demo requires TRW-S to be installed to run correctly. The directory ./intseg/utils/imrender/ contains the source code for TRW-S. Please recompile the source if the existing binary files are incompatible with your system.

The repository also contains data corresponding to an example image within the 'voctest50data' directory. The entire database (corresponding to PASCAL VOC2007 val set) can be downloaded from ( https://filebox.ece.vt.edu/~vittal/embr/voctest50data.tar ). Please replace the downloaded directory with the 'voctest50data' directory. You would need to write your own wrapper to loop through the entire dataset.

The demo can be seen by running the demo_DivMBest_intseg.m script.

## Acknowledgements

    @article{kolmogorov2006convergent,
      title={Convergent tree-reweighted message passing for energy minimization},
      author={Kolmogorov, Vladimir},
      journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on},
      volume={28},
      number={10},
      pages={1568--1583},
      year={2006},
      publisher={IEEE}
    }
    
We thank [Oliver Woodford](http://www.robots.ox.ac.uk/~ojw/) for providing matlab wrappers of the TRW-S implementation.
