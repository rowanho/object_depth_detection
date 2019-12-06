# Computer Vision Assignment

Instructions For Running the Program

In main.py:

Change variable "master_path_to_dataset" to the path of the intended dataset.

The following files also need to be in the directory:
	yolov3.cfg
	yolo3.weights


To run the program:
```
> python main.py
```
There are also additional command line flags to use alternative implementation techniques.

To use the sparse disparity method instead of dense:
```
> python3 main.py --is_sparse=true
```

To use MOG2 background subtraction:
```
> python3 main.py --use_fg_mask=true
```

Both flags also work together:
```
> python3 main.py --is_sparse=true --use_fg_mask=true
```

