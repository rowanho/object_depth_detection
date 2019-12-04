Instructions For Running the Program

In main.py:

Change variable "master_path_to_dataset" to the path of the intended dataset.

On lab linux environment:

	> opencv4.1.1.init	
	> python3 main.py

There are also additional command line flags to use alternative implementation techniques.

Run:
	> python3 main.py --is_sparse=true

To use the sparse disparity method instead of dense

Run    > python3 main.py --use_fg_mask=true

To use MOG2 background subtraction

Both flags also work together:
	> python3 main.py --is_sparse=true --use_fg_mask=true
