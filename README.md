# mnist-pngs


Sometimes it is convenient to have png files of MNIST ready to go for teaching & tutorials. This is essentially just a copy of MNIST in png format.



You can clone it directly in Python via the following:



```python
import os
from git import Repo

if not os.path.exists('mnist-pngs'):
    Repo.clone_from("https://github.com/rasbt/mnist-pngs", "mnist-pngs")
```



(Note that it requires `gitpython`, which can be installed via `pip install gitpython`).

