## Important
### outdated way of getting dependencies
- pyproject.toml is outdated for now
### Steps Before Testing Your `.pt` File Model
1. Adjust the variables responsible for reading the existing model.
2. Ensure your model is in the correct directory.
3. Disable the specific code line, otherways an error will be generated.
<br>**Line 21:**  
```python
   training_results, validation_results = train_model(model)
```
### Where Will the Output Go?
- The output and input files are located in the `test` folder.
