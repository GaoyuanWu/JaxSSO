# CHANGELOG

## 0.0.1
First release.

## 0.0.2
Similar to 0.0.1

## 0.0.3
Add method `'Sens_C_Coord'` in `Model_Sens`, which is more efficient than `'Sens_k_Coord'`.
This method calculates the gradient of strain energy directly, given the displacement vector.


## 0.0.6
Separate FEA analysis module and Sensitivity analysis model.

## 1.0.0
Major update.
Support beams and quadrilateral shell elements based on MITC-4 shell.
Support shape, size, topology optimization.
Include five examples:
* Shape optimization of gridshell
* Shape optimization of continuous shell
* Size optimization of continuous shell
* Two examples of simultaneous shape and topology optimization
* Integration with Neural Networks for structural optimization
