# Todo : Scions

### 29 Feb 2024

- [x] Write topological sort
- [x] Make DAG based on ID rather than index

### 1-12 March 2024

- [x] Topological sort based on effective ranks
- [x] Regeneration of indices after sort
- [ ] Labels for ease of selecting input and output ?
  > Doesn't seem possible as storing string_views will be difficult for comptime
- [x] Compute Graph :
  - [x] Sub Graph Generation
  - [x] Denoting Input and Output Nodes

### 13-18 March 2024

- [ ] **Tests** for Manifold
- [ ] **Auto diff** implementation :
  - [x] System to figure out derivative OP
  - [x] Expression Grouping
  - [x] Expression Group -> Expression Group
  - [x] Pinning of Group sequence
  - [x] Default group? (disables first operation to be a group)
  - [ ] Reverse Mode AD (this is going to be hard)
    - [ ] Reverse Count (Count the number of ops and tensors required for AD) ? probaby just do it
- [x] Fix sorting: Use dependency based topological sorting rather than naive
- [x] Fix Dot to use groups (kinda done, I just label the group and pinned)

### 20-31 March 2024

- [ ] Write Memory Manager for CPU
- [ ] write Code Gen logic for CPU
- [ ] Support Expressions containing expressions

## Ideas?

- **Strides** for Tensors? _Will be kind of difficult to implement effectively_
