# CS4278_Project

This project requires the `gym-duckietown` dependency 
from [here](https://github.com/AdaCompNUS/CS4278-5478-Project-Materials).

```bash
$ git clone https://github.com/AdaCompNUS/CS4278-5478-Project-Materials.git 
$ cd CS4278-5478-Project-Materials
$ pip install -e gym-duckietown
$ cd ..
$ git clone https://github.com/teoha/CS4278_Project.git
```

## Generate Control Files

```bash
$ cd code
$ python run_tests.py
```
Files are saved to `./control_files`.

## Run Single Map

```bash
$ cd code
$ python -m iil-dagger.run -m $MAP_NAME -s $SEED -st $START -gt $GOAL
```