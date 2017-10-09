# Pointer Sentinel Mixture Models
To implement the ICLR 2017 paper: *Pointer Sentinel Mixture Models* in Pytorch.

## Some tricks

- The sentinel vector should be initialized properly. Otherwise, g will be extremely close to 1 at first so that the Pointer Network could not be trained normally(no gradient).
- For convenience, I use a large tensor(`length * batch_size * vocab_size`) to compute `p_ptr`. It's a heavy load for GPU and will be removed later.
- Dropout is not required for this model.

## TODO

- [ ] Truncated BPTT
