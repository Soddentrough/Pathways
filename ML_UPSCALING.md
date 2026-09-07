We have a path tracer which produces noisy output but we can control resolution, sample counts, and accumulation time. 

Let's use the most advanced, cutting edge research from the CV field to build a relatively lightweight machine learning denoiser/upscaler.

We can capture grainy, 1080p sequences along with clean high resolution ground truth sequences and use that data to train a bespoke model to post process the original input frame(s) into a pleasing final image.
