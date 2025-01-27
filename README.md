# Gradual
PyTorch inspired tensor/autograd library.
Heavily abuses std::shared_ptr because I wanted to learn cpp while avoiding memory issues. 

# Setup
Install googles GTest testing framework (might not be important because i use fetchcontent now idk tho)
```shell
sudo apt install libgtest-dev
```

# Build
```shell
./build.sh build --debug
```

# Run 
```shell 
./build.sh build --debug && ./build.sh test
```


# Sources

https://cs231n.stanford.edu/handouts/derivatives.pdf
https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf