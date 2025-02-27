# Current Considerations

## Porting Code and Model onto Arduino

1. Basic C code for feature extraction with Signal read added - need to consider sampling rate and inference frequency based on calculation speed and microcontroller capacity (empirical testing)

2. Model options - primary consideration is speed - only inference can be run on edge, so directly implementing most ML techniques is not feasible. Possible alternatives:
-   Current (Least feasible) Training simple ML models using scikit-learn / python and porting to C/C++ code using libraries like [cONNXr](https://github.com/alrevuelta/cONNXr) or [libonnx](https://github.com/xboot/libonnx)
-   Choosing a classifier that can be better implemented with more efficient inference times than KNN using C++ libraries like [mlpack](https://www.mlpack.org/doc/index.html#classification-algorithms) or [shark](http://image.diku.dk/shark/sphinx_pages/build/html/index.html) - More libraries that can be explored at [C++ Libraries for ML](https://github.com/josephmisiti/awesome-machine-learning?tab=readme-ov-file#cpp)
-   Very very very basic multi-layer perceptron (2-3 layers with very few nodes) trained and then exported as TFLite with relatively few calculations anticipated when running inference. 

    Choosing one or the other from the above should be based on accuracy and speed considered together. 

## Speeding Up Feature Extraction

1. O(n) functions for features extraction - may not be possible to improve as linear data and all signals must be processed, however, refactoring for more efficiency may be an option.

2. Exploring other alternatives to this - maybe using some linear algebra techniques (?) 