# Basic Usage

This script provides a basic example of how to make use of the package. The idea here, is that training loops and such are customised to your needs; dependent on the environment. But you utilise the library for the RL side of things; selecting an action & training the policy. The package provides additional utilities (memory buffer, record) in order to make other parts of the training easier.

Run the script
```bash
python3 basic_usage.py
```

Thanks to `Record` the results of our training is saved in the `./global_logs` directory.