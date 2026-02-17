Follow instructions from Ray's official documentation and install ray on all nodes in your cluster
# pip install -U "ray[default]"

Then start ray on one node with `ray start --head --port=6379 --dashboard-host=0.0.0.0`
To add other nodes to the same ray cluster, use `ray start --address='<head_node_ip:port>'` which is shown after the cluster is started on head node

To use Ray in your python code:

1. `import ray`
2. `ray.init(address="auto")`
3. To mark as a Ray function use `@ray.remote ` before the function definition. This lets Ray schedule and distribute this function later.
Eg: 
`@ray.remote`
`def predict_batch():`

4. To run the function, use `ray.get()g`. Eg:
`result = predict_batch()`
`ray.get(result)`

5. To shutdown ray, use `ray.shutdown()`
6. To submit Ray job, `ray job submit --submission-id <name> --working-dir ,dir_path> -- python <py_filename>`
