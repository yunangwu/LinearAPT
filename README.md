### Environment

Run the following code to setup environment:

```
conda create --name env_name_here --file requirements.txt
```

### Reproduce the Result

First, change the name of the `/run` directory:

```
mv run run_backup
```

Then, run the `run.sh` shell script:

```
mkdir run
sh run.sh
```
