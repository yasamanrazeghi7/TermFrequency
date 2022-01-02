# TYM
## aws scripts pipeline
The entire pipeline for running and processing the data is in scripts folder.
need do ```./run_pipeline``` after assigning the desired flags and codes.
### Pipeline
The `scripts/run_pipeline.sh` script
creates aws clusters (`scripts/create_cluters.sh`), distributes 
the files to clusters (`scripts/run_files`), then clean up the results,
and finally, terminates the clusters (`scripts/terminate_clusters`).
### Running the pipeline
In  `scripts/run_pipeline.sh` script assign 
- ``CLUSTER_COUNT``
- ``TOTAL_FILES``
- ``CLUSTER_PREFIX``
- ``RUN_RANGE_FILE``
- ``CLEANUP_FILE`` 
 
to run the code, you need to change directory to `scripts` then run the scripts.
## tests
to run the unit tests
python -m unittest regex_unit_test.py 


