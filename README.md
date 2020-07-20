# Latent-mixture
The focus on this branch:  
- [x] Using prior that is clustering focus, Gaussian mixture in particular.  
     - Works well for first and second task, but fails subsequently.  
     - Model not learning subsequent task.  
     - The model, p(x|z) = g(z), z from clustering prior.
- [] Initialization of new Zs such that the new Zs are intialized away from previous clusters.  
- [] Limit the proposals to be away from previously learned clusters.
- [] Regularize the movement of previously learned cluster?
