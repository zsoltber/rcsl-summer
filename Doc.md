# Docs for my 6 week research work at RCSL

## Algorithms

### Pruning-based Destructive Weight Identification

This algorithm is based on the heuristic idea that through randomised pruning of weights and running accuracy delta evaluations with these new weights, over a few random passes, weight candidates can be identified that hurt the performance of the model on the novel data. Changing these weights under various modification paradigms has shown the ability to increase accuracy of detections without any off-device or backpropagation component.

(Done on the PYNQ Z2 on the PS part: ARM Cortex A9 32-bit armv7l)